import math


class PoseAvoidance:
    def __init__(self, rigs, targets, chains, motion_limits=None):
        from mathutils import Vector

        self.rigs, self.targets, self.chains = rigs, targets, chains
        self.motion_limits = motion_limits or {}
        self.step_dt = 0.
        self.step_wrists = {side: Vector() for side in rigs}
        self.wrists = {side: Vector() for side in rigs}
        self.angles = {side: Vector() for side in rigs}
        self.fingers = {key: Vector() for key in targets}
        self.curls = {key: Vector() for key in targets}
        self.cups = {key: Vector() for key in targets}
        self.previous_contacts = {}
        self.failed_signature = None
        self.stats = dict(samples=0, trials=0, accepted=0, cached_failures=0,
                          max_wrist_shift_mm=0., max_finger_shift_mm=0.)

    def separate_airborne_order(self, samples, clearance=.003):
        """Separate adjacent evaluated finger chains, moving only airborne joints.

        Work in armature coordinates so a turning wrist cannot reverse the test.
        Preserve the evaluated IK pose before changing the MCP spread; active
        fingers and their targets are never edited by this pass.
        """
        import bpy
        from mathutils import Vector

        changed = 0
        pairs = (('index', 'middle'), ('middle', 'ring'), ('ring', 'little'))
        for _ in range(4):
            corrected = False
            for side, arm in self.rigs.items():
                scale = max(abs(v) for v in arm.scale)
                for first, second in pairs:
                    if first not in self.chains or second not in self.chains:
                        continue
                    a, b = self.chains[first], self.chains[second]
                    sign = 1 if arm.data.bones[b[0]].head_local.x > arm.data.bones[a[0]].head_local.x else -1
                    # Compare corresponding joints. Projecting entire curved
                    # chains onto X overstates overlap at different depths.
                    pairs_at = [(arm.pose.bones[x].tail.copy(), arm.pose.bones[y].tail.copy())
                                for x, y in zip(a, b)]
                    pairs_at.append((arm.pose.bones[a[0]].head.copy(), arm.pose.bones[b[0]].head.copy()))
                    first_point, second_point = min(pairs_at, key=lambda p: sign * (p[1].x - p[0].x))
                    gap = sign * (second_point.x - first_point.x) * scale
                    missing = clearance - gap
                    free = [(side, f) for f in (first, second)
                            if (side, f) in samples and not samples[side, f][2]
                            and samples[side, f][1] < .95]
                    if missing <= .00001 or not free:
                        continue
                    derivatives = []
                    for key in free:
                        bone = arm.pose.bones[self.chains[key[1]][0]]
                        point = first_point if key[1] == first else second_point
                        reference = (bone.parent.matrix @ bone.parent.bone.matrix_local.inverted()
                                     @ bone.bone.matrix_local) if bone.parent else bone.bone.matrix_local
                        axis = (reference.to_3x3() @ Vector((0., 0., 1.))).normalized()
                        derivative = axis.cross(point - bone.head).x * scale * sign * (-1 if key[1] == first else 1)
                        if abs(derivative) > 1e-6:
                            derivatives.append((key, bone, derivative))
                    denominator = sum(d * d for _, _, d in derivatives)
                    for key, bone, derivative in derivatives:
                        self.curl_finger(key, Vector())
                        step = missing * derivative / denominator
                        limit = getattr(self, 'finger_lateral_limit', None)
                        angle = bone.rotation_euler.z + max(-.04, min(.04, step))
                        bone.rotation_euler.z = max(-limit, min(limit, angle)) if limit is not None else angle
                        corrected = True
                        changed += 1
                    if derivatives:
                        bpy.context.view_layer.update()
            if not corrected:
                break
        self.stats['airborne_order_adjustments'] = self.stats.get('airborne_order_adjustments', 0) + changed
        return changed

    def move_wrist(self, side, delta, angles, roots, samples, project=False):
        from mathutils import Euler, Matrix, Vector

        arm = self.rigs[side]
        pivot = roots[side].copy()
        rotation = Euler(angles).to_matrix().to_4x4()
        if project:
            wanted = pivot + delta
            disks = []
            for key, (pos, _, down) in samples.items():
                if key[0] != side or not down:
                    continue
                chain = self.chains[key[1]]
                base = rotation.to_3x3() @ (arm.matrix_world @ arm.pose.bones[chain[0]].head - pivot)
                length = sum(arm.data.bones[name].length for name in chain) * max(abs(v) for v in arm.scale) * .97
                wanted.z = min(wanted.z, pos[2] - base.z + length * .97)
                disks.append((Vector(pos), base, length))
            for _ in range(12):
                for pos, base, length in disks:
                    radius = math.sqrt(max(1e-8, length ** 2 - (wanted.z + base.z - pos.z) ** 2))
                    center = Vector((pos.x - base.x, pos.y - base.y, wanted.z))
                    lateral = wanted - center
                    if lateral.length > radius:
                        wanted = center + lateral.normalized() * radius
            delta = wanted - pivot
            if self.motion_limits and self.motion_limits.get('speed_limits', True):
                # A collision proposal is a per-sample correction, not a new
                # destination that may bypass the baker's wrist speed limit.
                change = self.wrists[side] + delta - self.step_wrists[side]
                budget = self.motion_limits.get('wrist_speed', .18) * self.step_dt
                if change.length > budget:
                    change *= budget / change.length
                delta = self.step_wrists[side] + change - self.wrists[side]
        transform = Matrix.Translation(pivot + delta) @ rotation @ Matrix.Translation(-pivot)
        old_euler = arm.rotation_euler.copy()
        arm.matrix_world = transform @ arm.matrix_world
        arm.rotation_euler = arm.matrix_world.to_euler('XYZ', old_euler)
        roots[side] += delta
        for key, (pos, weight, down) in samples.items():
            if key[0] == side and not down:
                samples[key] = (transform @ Vector(pos), weight, down)
                self.targets[key].location = transform @ self.targets[key].location
        return delta

    def move_finger(self, key, delta, samples):
        from mathutils import Vector

        pos, weight, down = samples[key]
        assert not down, 'Avoidance must not move a held contact'
        arm = self.rigs[key[0]]
        tip = arm.pose.bones[self.chains[key[1]][-1]]
        if tip.constraints[0].influence < .999:
            # FK may intentionally keep this airborne finger curled while its
            # inactive IK target points toward a distant future note. Enabling
            # that target directly straightened the finger for a single frame.
            pad_offset = Vector(pos) - self.targets[key].location
            self.targets[key].location = arm.matrix_world @ tip.tail
            pos = self.targets[key].location + pad_offset
        samples[key] = (Vector(pos) + delta, weight, down)
        self.targets[key].location += delta
        tip.constraints[0].influence = 1.

    def curl_finger(self, key, delta):
        bones = [self.rigs[key[0]].pose.bones[name] for name in self.chains[key[1]]]
        if bones[-1].constraints[0].influence > .001:
            # Preserve the evaluated pose when handing control back from IK to
            # explicit joint rotations, then apply only the requested curl.
            rotations = [bone.bone.convert_local_to_pose(bone.matrix, bone.bone.matrix_local,
                parent_matrix=bone.parent.matrix, parent_matrix_local=bone.parent.bone.matrix_local,
                invert=True).to_euler('XYZ', bone.rotation_euler) for bone in bones]
            for bone, rotation in zip(bones, rotations):
                bone.rotation_euler = rotation
        bones[-1].constraints[0].influence = 0.
        for index, (bone, angle) in enumerate(zip(bones, delta)):
            bone.rotation_euler.x = max(-1.5, min(.8 if index == 0 else 0., bone.rotation_euler.x + angle))

    def cup_palm(self, key, angles):
        # These metacarpal/opposition bones move the knuckle, while contact IK keeps the pad down.
        bone = self.rigs[key[0]].pose.bones[self.chains[key[1]][0]].parent
        if key[1] != 'thumb' and getattr(self, 'finger_lateral_limit', None) is not None:
            angles = (angles[0], 0., 0.)
        bone.rotation_euler = tuple(a + b for a, b in zip(bone.rotation_euler, angles))

    def apply(self, roots, samples, dt):
        """Carry the previous solution forward with a gradual return to the planned pose."""
        contacts = {key: tuple(pos) for key, (pos, _, down) in samples.items() if down}
        steady = bool(contacts) and contacts.keys() == self.previous_contacts.keys()
        decay = 1. if steady else math.exp(-max(0., dt) / .22)
        self.previous_contacts = contacts
        changed = False
        for side in self.rigs:
            self.wrists[side] *= decay
            self.angles[side] *= decay
            if self.wrists[side].length + self.angles[side].length > 1e-8:
                self.move_wrist(side, self.wrists[side], self.angles[side], roots, samples)
                changed = True
        for key in self.fingers:
            if key in samples:
                self.cups[key] *= decay
                if self.cups[key].length > .001:
                    self.cup_palm(key, self.cups[key])
                    changed = True
            if key not in samples or samples[key][2]:
                self.fingers[key].zero()
                self.curls[key].zero()
                continue
            self.fingers[key] *= decay
            if self.fingers[key].length > .00005:
                self.move_finger(key, self.fingers[key], samples)
                changed = True
            self.curls[key] *= decay
            if self.curls[key].length > .001:
                self.curl_finger(key, self.curls[key])
                changed = True
        return changed

    def prepare(self, roots, samples, dt, pads, fit):
        """Keep a previous offset only while the current contacts remain reachable."""
        from mathutils import Vector

        self.step_dt = max(0., dt)
        self.step_wrists = {side: value.copy() for side, value in self.wrists.items()}
        original = self.snapshot(roots, samples)
        limits = {key: max(.0007, (pads[key] - Vector(pos)).length + .00005)
                  for key, (pos, _, down) in samples.items() if down}
        for attempt in range(4):
            if attempt:
                self.restore(original, roots, samples)
                for name, speed in (('wrists', .18), ('angles', 2.), ('fingers', .45), ('curls', 2.5), ('cups', 1.2)):
                    for value in getattr(self, name).values():
                        if self.motion_limits and self.motion_limits.get('speed_limits', True):
                            reduction = value * (1 - .5**attempt)
                            budget = speed * self.step_dt
                            if reduction.length > budget:
                                reduction *= budget / reduction.length
                            value -= reduction
                        else:
                            value *= .5 ** attempt
            if not self.apply(roots, samples, dt):
                return pads
            result = fit(samples)
            if all((result[k] - Vector(samples[k][0])).length <= limit for k, limit in limits.items()):
                return result
        self.restore(original, roots, samples)
        for values in (self.wrists, self.angles, self.fingers, self.curls, self.cups):
            for value in values.values():
                value.zero()
        return fit(samples)

    def snapshot(self, roots, samples):
        from mathutils import Matrix, Vector

        def solved_basis(bone):
            parent = (dict(parent_matrix=bone.parent.matrix,
                           parent_matrix_local=bone.parent.bone.matrix_local) if bone.parent else {})
            solved = bone.bone.convert_local_to_pose(bone.matrix, bone.bone.matrix_local, invert=True, **parent)
            # This rig animates rotations only. Repeatedly restoring an evaluated
            # IK matrix feeds tiny scale/shear errors back into the next solve.
            # Keep the solved rotation while preserving rest translation/scale.
            rotation = solved.to_quaternion()
            rotation.normalize()
            return Matrix.LocRotScale(bone.location.copy(), rotation, bone.scale.copy())
        return dict(
            previous_contacts=dict(self.previous_contacts),
            arms={side: (arm.matrix_world.copy(), arm.rotation_euler.copy()) for side, arm in self.rigs.items()},
            bones={(side, bone.name): (solved_basis(bone), tuple(c.influence for c in bone.constraints))
                   for side, arm in self.rigs.items() for bone in arm.pose.bones},
            bone_channels={(side, bone.name): (bone.location.copy(), bone.scale.copy())
                           for side, arm in self.rigs.items() for bone in arm.pose.bones},
            targets={key: obj.location.copy() for key, obj in self.targets.items()},
            roots={side: pos.copy() for side, pos in roots.items()}, samples=dict(samples),
            wrists={side: pos.copy() for side, pos in self.wrists.items()},
            angles={side: angle.copy() for side, angle in self.angles.items()},
            fingers={key: pos.copy() for key, pos in self.fingers.items()},
            curls={key: pos.copy() for key, pos in self.curls.items()},
            cups={key: pos.copy() for key, pos in self.cups.items()})

    def restore(self, state, roots, samples):
        import bpy

        self.previous_contacts = dict(state['previous_contacts'])

        for side, (matrix, angles) in state['arms'].items():
            self.rigs[side].matrix_world = matrix
            self.rigs[side].rotation_euler = angles
        for (side, name), (matrix, influences) in state['bones'].items():
            bone = self.rigs[side].pose.bones[name]
            bone.matrix_basis = matrix
            bone.location, bone.scale = state['bone_channels'][side, name]
            for constraint, influence in zip(bone.constraints, influences):
                constraint.influence = influence
        for key, pos in state['targets'].items():
            self.targets[key].location = pos
        roots.update({side: pos.copy() for side, pos in state['roots'].items()})
        samples.clear()
        samples.update(state['samples'])
        for name in ('wrists', 'angles', 'fingers', 'curls', 'cups'):
            setattr(self, name, {key: value.copy() for key, value in state[name].items()})
        bpy.context.view_layer.update()

    def proposals(self, collisions, samples):
        from mathutils import Vector

        if any(first[0] != second[0] for _, first, second, _ in collisions):
            # A coupled move can leave a local minimum that neither wrist can escape alone.
            for upper in (1, -1):
                for height in (.018, .03, .045):
                    yield 'layer', upper, Vector((0., 0., height)), Vector()
            if all(any(key[0] == side and down for key, (_, _, down) in samples.items()) for side in (-1, 1)):
                for upper in (1, -1):
                    for behind in (.07, .09):
                        for height in (.07, .08):
                            yield 'arch', upper, Vector((behind, height, .03)), Vector()
        seen = set()
        for clearance, first, second, direction in collisions[:4]:
            distance = max(.004, min(.018, (.001 - clearance) * 2.5))
            for collider, sign in ((first, 1), (second, -1)):
                key = collider[:2]
                if collider[1] == 'palm' and 100 <= collider[2] < 105:
                    key = (collider[0], ('index', 'middle', 'ring', 'little', 'thumb')[collider[2] - 100])
                if key not in samples or key in seen:
                    continue
                seen.add(key)
                yield 'cup', key, Vector(), Vector((.14, 0., 0.))
                yield 'cup', key, Vector(), Vector((-.14, 0., 0.))
                yield 'cup', key, Vector(), Vector((0., 0., .18))
                yield 'cup', key, Vector(), Vector((0., 0., -.18))
                yield 'cup', key, Vector(), Vector((.06, 0., 0.))
                yield 'cup', key, Vector(), Vector((-.06, 0., 0.))
                yield 'cup', key, Vector(), Vector((0., 0., .08))
                yield 'cup', key, Vector(), Vector((0., 0., -.08))
                yield 'cup', key, Vector(), Vector((.06, 0., .08))
                yield 'cup', key, Vector(), Vector((.06, 0., -.08))
                if not samples[key][2]:
                    delta = direction * sign * distance
                    delta.z = max(.003, delta.z)
                    yield 'finger', key, delta, Vector()
                    yield 'finger', key, Vector((0., 0., distance)), Vector()
                    yield 'curl', key, Vector((.25, -.15, -.10)), Vector()
                    yield 'curl', key, Vector((.5, -.3, -.2)), Vector()
        sides = {key[0] for _, first, second, _ in collisions[:4] for key in (first, second)}
        # Move the less occupied hand first, with deterministic ties to avoid swapping layers.
        sides = sorted(sides, key=lambda side: (sum(down for key, (_, _, down) in samples.items() if key[0] == side), -side))
        for side in sides:
            active = [Vector(pos) for key, (pos, _, down) in samples.items() if key[0] == side and down]
            if active:
                # Raising a wrist also needs horizontal slack in its held fingers.
                # Move toward their mean contact while lifting, rather than stretching the chain.
                for forward, upward in ((.012, .008), (.024, .018), (.04, .03)):
                    yield 'wrist', side, Vector((0., forward, upward)), Vector()
                    yield 'wrist', side, Vector((0., -forward, -upward)), Vector()
            for clearance, first, second, direction in collisions[:2]:
                for collider, sign in ((first, 1), (second, -1)):
                    if collider[0] == side:
                        yield 'wrist', side, direction * sign * max(.002, min(.008, .001 - clearance)), Vector()
            yield 'wrist', side, Vector((0., 0., .003)), Vector()
            yield 'wrist', side, Vector((0., 0., -.003)), Vector()
            yield 'wrist', side, Vector((0., -.003, 0.)), Vector()
            yield 'wrist', side, Vector((0., .003, 0.)), Vector()
            yield 'wrist', side, Vector((0., 0., .008)), Vector()
            yield 'wrist', side, Vector((0., 0., .018)), Vector()
            yield 'wrist', side, Vector((0., 0., -.008)), Vector()
            yield 'wrist', side, Vector((0., 0., -.018)), Vector()
            yield 'wrist', side, Vector((side * .008, 0., .006)), Vector()
            yield 'wrist', side, Vector((-side * .008, 0., .006)), Vector()
            yield 'wrist', side, Vector((0., -.008, .006)), Vector()
            yield 'wrist', side, Vector((0., .008, 0.)), Vector()
            yield 'wrist', side, Vector(), Vector((0., side * .14, 0.))
            yield 'wrist', side, Vector(), Vector((.16, 0., 0.))
            yield 'wrist', side, Vector(), Vector((-.16, 0., 0.))
            yield 'wrist', side, Vector(), Vector((0., 0., side * .14))
            yield 'wrist', side, Vector(), Vector((0., 0., -side * .14))
            yield 'wrist', side, Vector(), Vector((0., 0., side * .05))
            yield 'wrist', side, Vector(), Vector((0., 0., -side * .05))
            yield 'wrist', side, Vector(), Vector((.05, 0., 0.))
            yield 'wrist', side, Vector(), Vector((-.05, 0., 0.))

    def resolve(self, roots, samples, pads, collisions_at, fit, clearance_at, scale=1.):
        """Bounded local search. Reject candidates that sacrifice contacts or screen clearance."""
        from mathutils import Vector

        collisions = collisions_at()
        if not collisions:
            self.failed_signature = None
            return pads, 0
        signature = (tuple((first, second, round(clearance * 1000, 1)) for clearance, first, second, _ in collisions),
                     tuple((side, *(round(v, 4) for v in root)) for side, root in roots.items()),
                     tuple((key, *(round(v, 4) for v in pos)) for key, (pos, _, down) in samples.items() if down))
        if signature == self.failed_signature:
            self.stats['cached_failures'] += 1
            return pads, 0
        self.stats['samples'] += 1
        limits = {key: max(.0007, (pads[key] - Vector(pos)).length + .00005)
                  for key, (pos, _, down) in samples.items() if down}
        floor = min(-.0002, clearance_at() - .00005)

        def cost(items):
            return sum(max(0., .001 - clearance) ** 2 for clearance, *_ in items)

        accepted = 0
        trials_left = 24
        for _ in range(2):
            original = self.snapshot(roots, samples)
            best, best_cost = None, cost(collisions)
            # Visit different remedies early so dense chords cannot spend hundreds
            # of mesh evaluations exhausting one family of unsuccessful guesses.
            families = {}
            for proposal in self.proposals(collisions, samples):
                families.setdefault(proposal[0], []).append(proposal)
            proposals = []
            for index in range(max(map(len, families.values()), default=0)):
                for values in families.values():
                    if index < len(values):
                        proposals.append(values[index])
            for kind, key, delta, angles in proposals:
                if trials_left <= 0:
                    break
                if (getattr(self, 'finger_lateral_limit', None) is not None and kind in ('finger', 'curl')
                        and key[1] != 'thumb'):
                    continue
                self.restore(original, roots, samples)
                delta = delta * scale if kind != 'curl' else delta
                if self.motion_limits and self.motion_limits.get('speed_limits', True):
                    if kind in ('finger', 'curl'):
                        budget = (.45 if kind == 'finger' else 2.5) * self.step_dt
                        if delta.length > budget:
                            delta *= budget / delta.length
                    angular_budget = (1.2 if kind == 'cup' else 2.) * self.step_dt
                    if angles.length > angular_budget:
                        angles *= angular_budget / angles.length
                if kind == 'arch':
                    for side in (key, -key):
                        held = [(k, Vector(pos)) for k, (pos, _, down) in samples.items() if k[0] == side and down]
                        center = sum((p for _, p in held), Vector()) / len(held)
                        arm = self.rigs[side]
                        base_x = sum((arm.matrix_world @ arm.pose.bones[self.chains[k[1]][0]].head).x
                                     - roots[side].x for k, _ in held) / len(held)
                        wanted = Vector((center.x - base_x, center.y - (delta.x if side == key else .14 * scale),
                                         delta.y if side == key else delta.z))
                        actual = self.move_wrist(side, wanted - roots[side], angles, roots, samples, project=True)
                        self.wrists[side] += actual
                    if any(v.length > .12 * scale for v in self.wrists.values()):
                        continue
                elif kind == 'cup':
                    if (self.cups[key] + angles).length > (.8 if key[1] == 'thumb' else .35):
                        continue
                    self.cups[key] += angles
                    self.cup_palm(key, angles)
                elif kind == 'layer':
                    for side, shift in ((key, delta), (-key, -delta * .5)):
                        actual = self.move_wrist(side, shift, angles, roots, samples, project=True)
                        self.wrists[side] += actual
                    if any(v.length > .09 * scale for v in self.wrists.values()):
                        continue
                elif kind == 'curl':
                    if (self.curls[key] + delta).length > 1.:
                        continue
                    self.curls[key] += delta
                    self.curl_finger(key, delta)
                elif kind == 'finger':
                    if (self.fingers[key] + delta).length > self.motion_limits.get('max_finger_shift', .010) * scale:
                        continue
                    self.fingers[key] += delta
                    self.move_finger(key, delta, samples)
                else:
                    if ((self.wrists[key] + delta).length > .07 * scale
                            or (self.angles[key] + angles).length > .7):
                        continue
                    self.angles[key] += angles
                    actual = self.move_wrist(key, delta, angles, roots, samples, project=True)
                    if (self.wrists[key] + actual).length > .09 * scale:
                        continue
                    self.wrists[key] += actual
                self.stats['trials'] += 1
                trials_left -= 1
                if self.motion_limits and any(v.length > self.motion_limits.get('max_wrist_shift', .025) * scale
                                              for v in self.wrists.values()):
                    continue
                trial_pads = fit(samples)
                if any((trial_pads[k] - Vector(samples[k][0])).length > limit for k, limit in limits.items()):
                    continue
                trial = collisions_at()
                value = cost(trial)
                if value >= best_cost - 1e-12 or clearance_at() < floor:
                    continue
                best, best_cost = self.snapshot(roots, samples), value
                if not trial or value < cost(collisions) * .25:
                    break
            self.restore(best or original, roots, samples)
            pads = fit(samples)
            if best is None:
                break
            accepted += 1
            self.stats['accepted'] += 1
            self.stats['max_wrist_shift_mm'] = max(self.stats['max_wrist_shift_mm'],
                                                  max(v.length for v in self.wrists.values()) * 1000)
            self.stats['max_finger_shift_mm'] = max(self.stats['max_finger_shift_mm'],
                                                   max(v.length for v in self.fingers.values()) * 1000)
            collisions = collisions_at()
            if not collisions:
                break
        self.failed_signature = signature if accepted == 0 else None
        return pads, accepted
