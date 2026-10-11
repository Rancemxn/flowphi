import json
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from flowphi.rendering.blender import camera_pixel, configure_camera, project_shadow_mesh, screen_rect
from flowphi.progress import report_progress


def chunk_command(job, start, count, left, top, hands, shadows, output):
    w, h, fps = job['width'], job['height'], job['fps']
    x, y, sw, sh = screen_rect(job)
    graph = (
        # FFV1/MKV timestamps use milliseconds; normalize before shortest overlays at 60 fps.
        f'[0:v]settb=1/{fps},setpts=N[screen];[1:v]settb=1/{fps},setpts=N[hands];'
        f'[2:v]settb=1/{fps},setpts=N[mask];[3:v]settb=1/{fps},setpts=N[background];'
        f'[background][screen]overlay={x}:{y}:shortest=1,'
        f'drawbox=x={x - 7}:y={y - 7}:w={sw + 14}:h={sh + 14}:color=0x1d2530:t=7[bg];'
        f'[mask]alphaextract,scale={w}:{h}:flags=bilinear,boxblur=3:1,lutyuv=y=val*0.20[alpha];'
        f'color=black:s={w}x{h}:r={fps},format=rgba[black];[black][alpha]alphamerge[shadow];'
        '[bg][shadow]overlay=0:0:shortest=1[under];'
        f'[under][hands]overlay={left}:{top}:shortest=1,format=yuv420p[video]'
    )
    return ['ffmpeg', '-v', 'warning', '-y', '-ss', str((start - 1) / fps), '-i', job['screen_video'],
            '-i', str(hands), '-i', str(shadows), '-loop', '1', '-framerate', str(fps), '-i', job['background'],
            '-filter_complex', graph, '-map', '[video]', '-an', '-c:v', 'libx264', '-preset', 'veryfast',
            '-crf', '20', '-frames:v', str(count), '-progress', 'pipe:1', '-nostats', str(output)]


def completed_chunk(path, job, count):
    if not path.is_file():
        return False
    try:
        info = json.loads(subprocess.check_output(
            ['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_streams', '-of', 'json', str(path)],
            text=True, stderr=subprocess.DEVNULL))['streams'][0]
        return (info['width'] == job['width'] and info['height'] == job['height']
                and info['r_frame_rate'] == f"{job['fps']}/1" and int(info['nb_frames']) == count)
    except (subprocess.CalledProcessError, ValueError, KeyError, IndexError):
        return False


def main(job, resume=False):
    import bpy
    from mathutils import Vector

    out = Path(job['output']).resolve()
    scene = bpy.context.scene
    w, h, fps = job['width'], job['height'], job['fps']
    scene.render.resolution_x, scene.render.resolution_y = w, h
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = 'FFMPEG'
    scene.render.ffmpeg.format, scene.render.ffmpeg.codec = 'MKV', 'FFV1'
    scene.render.ffmpeg.audio_codec = 'NONE'
    scene.render.image_settings.color_mode = 'RGBA'
    scene.render.image_settings.color_depth = '8'
    scene.render.film_transparent, scene.render.fps = True, fps
    configure_camera(scene, job)
    # The projected screen shadow is a separate pass; skip costly Workbench self-shadow maps.
    scene.display.shading.show_shadows = False
    meshes = [o for o in scene.objects if o.type == 'MESH']
    segments = out / 'segments'
    segments.mkdir(exist_ok=True)
    started = time.perf_counter()
    timings = dict(bulk_shadow=job.get('render_bulk_shadow', True), bounds_seconds=0., hands_seconds=0.,
                   shadow_seconds=0., shadow_update_seconds=0., shadow_updates=0, composite_seconds=0.)

    def update_shadow(scene):
        begin = time.perf_counter()
        for obj, shadow in projections:
            # Re-evaluate after replacing each shadow mesh: that mutation dirties the graph.
            evaluated = obj.evaluated_get(bpy.context.evaluated_depsgraph_get())
            mesh = bpy.data.meshes.new_from_object(evaluated)
            if job.get('render_bulk_shadow', True):
                project_shadow_mesh(mesh, evaluated.matrix_world)
            else:
                for vertex in mesh.vertices:
                    point = evaluated.matrix_world @ vertex.co
                    vertex.co = (point.x + .35 * point.z, point.y - .45 * point.z, 0.)
            old = shadow.data
            shadow.data = mesh
            bpy.data.meshes.remove(old)
        timings['shadow_update_seconds'] += time.perf_counter() - begin
        timings['shadow_updates'] += 1

    def report_frame(scene):
        completed = (start - 1) * 3 + pass_offset + scene.frame_current - start + 1
        print(f'HANDCAM_PROGRESS {completed}', flush=True)
        report_progress('video.hands' if pass_offset == 0 else 'video.shadows',
                        '渲染手部' if pass_offset == 0 else '渲染阴影',
                        done=scene.frame_current - start + 1, total=count, unit='帧',
                        detail=f"视频帧 {start}–{stop - 1} / {job['frames']}")

    files = []
    try:
        bpy.app.handlers.render_write.append(report_frame)
        for start in range(1, job['frames'] + 1, 150):
            stop = min(start + 150, job['frames'] + 1)
            count = stop - start
            final = segments / f'{start:05d}.mp4'
            if resume and completed_chunk(final, job, count):
                files.append(final)
                print(f'HANDCAM_PROGRESS {(stop - 1) * 3}', flush=True)
                report_progress('video.resume', '复用已完成片段', done=stop - 1, total=job['frames'], unit='帧', force=True)
                continue
            scene.frame_start, scene.frame_end = start, stop - 1
            # Native render borders skip transparent pixels without reducing hand resolution.
            begin = time.perf_counter()
            bounds = [w, h, 0, 0]
            report_progress('video.bounds', '计算渲染范围', done=0, total=count, unit='帧', force=True)
            for frame in range(start, stop):
                scene.frame_set(frame)
                for obj in meshes:
                    evaluated = obj.evaluated_get(bpy.context.evaluated_depsgraph_get())
                    for corner in evaluated.bound_box:
                        point = evaluated.matrix_world @ Vector(corner)
                        px, py = camera_pixel(scene, point)
                        bounds = [min(bounds[0], px), min(bounds[1], py), max(bounds[2], px), max(bounds[3], py)]
                report_progress('video.bounds', '计算渲染范围', done=frame - start + 1, total=count, unit='帧')
            left, top = max(0, int(bounds[0] - 16) // 2 * 2), max(0, int(bounds[1] - 16) // 2 * 2)
            right, bottom = min(w, int(bounds[2] + 18) // 2 * 2), min(h, int(bounds[3] + 18) // 2 * 2)
            if right <= left or bottom <= top:
                left, top, right, bottom = 0, 0, w, h
            scene.render.use_border = scene.render.use_crop_to_border = True
            scene.render.border_min_x = (left + .001) / w
            scene.render.border_max_x = (right + .001) / w if right < w else 1.
            scene.render.border_min_y = (h - bottom + .001) / h
            scene.render.border_max_y = (h - top + .001) / h if top else 1.
            scene.render.resolution_percentage = 100
            hands, shadows = out / 'hands-chunk.mkv', out / 'shadow-chunk.mkv'
            pass_offset = 0
            scene.render.filepath = str(hands)
            timings['bounds_seconds'] += time.perf_counter() - begin
            begin = time.perf_counter()
            report_progress('video.hands', '渲染手部', done=0, total=count, unit='帧',
                            detail=f"视频帧 {start}–{stop - 1} / {job['frames']}", force=True)
            bpy.ops.render.render(animation=True)
            timings['hands_seconds'] += time.perf_counter() - begin

            projections = []
            for obj in meshes:
                shadow = bpy.data.objects.new('Projected hand', bpy.data.meshes.new('Shadow'))
                scene.collection.objects.link(shadow)
                shadow.color = (0, 0, 0, 1)
                projections.append((obj, shadow))
                obj.hide_render = True
            scene.display.shading.light, scene.display.shading.show_cavity = 'FLAT', False
            # The soft shadow is blurred on output; half resolution preserves the approved look.
            scene.render.use_border, scene.render.resolution_percentage = False, 50
            bpy.app.handlers.frame_change_post.append(update_shadow)
            try:
                begin = time.perf_counter()
                scene.frame_set(start)
                pass_offset = count
                scene.render.filepath = str(shadows)
                report_progress('video.shadows', '渲染阴影', done=0, total=count, unit='帧',
                                detail=f"视频帧 {start}–{stop - 1} / {job['frames']}", force=True)
                bpy.ops.render.render(animation=True)
                timings['shadow_seconds'] += time.perf_counter() - begin
            finally:
                bpy.app.handlers.frame_change_post.remove(update_shadow)
                for obj, shadow in projections:
                    mesh = shadow.data
                    bpy.data.objects.remove(shadow, do_unlink=True)
                    bpy.data.meshes.remove(mesh)
                    obj.hide_render = False
                scene.display.shading.light, scene.display.shading.show_cavity = 'STUDIO', True

            temporary = final.with_suffix('.partial.mp4')
            command = chunk_command(job, start, count, left, top, hands, shadows, temporary)
            begin = time.perf_counter()
            report_progress('video.composite', '合成视频画面', done=0, total=count, unit='帧', force=True)
            with subprocess.Popen(command, stdout=subprocess.PIPE, text=True) as process:
                try:
                    for line in process.stdout:
                        if line.startswith('frame='):
                            completed = (start - 1) * 3 + count * 2 + min(count, int(line.split('=')[1]))
                            print(f'HANDCAM_PROGRESS {completed}', flush=True)
                            report_progress('video.composite', '合成视频画面', done=min(count, int(line.split('=')[1])),
                                            total=count, unit='帧')
                    if process.wait():
                        raise RuntimeError(f'Compositing failed at frame {start}')
                except BaseException:
                    if process.poll() is None:
                        process.terminate()
                    raise
            timings['composite_seconds'] += time.perf_counter() - begin
            temporary.replace(final)
            files.append(final)
            hands.unlink()
            shadows.unlink()
    finally:
        bpy.app.handlers.render_write.remove(report_frame)

    # Relative generated names avoid quoting user paths, and exclude stale chunks from older jobs.
    listing = segments / 'concat.txt'
    begin = time.perf_counter()
    report_progress('video.mux', '合并视频与音轨', force=True)
    listing.write_text(''.join(f"file '{p.name}'\n" for p in files), encoding='utf-8')
    subprocess.run(['ffmpeg', '-v', 'warning', '-y', '-f', 'concat', '-safe', '0', '-i', str(listing),
                    '-i', job['mixed_audio'], '-map', '0:v:0', '-map', '1:a:0', '-c:v', 'copy',
                    '-c:a', 'aac', '-b:a', '192k', '-t', str(job['duration']), '-movflags', '+faststart',
                    str(out / 'handcam.partial.mp4')], check=True)
    (out / 'handcam.partial.mp4').replace(out / 'handcam.mp4')
    for path in files:
        path.unlink()
    listing.unlink()
    timings['mux_seconds'] = time.perf_counter() - begin
    timings['total_seconds'] = time.perf_counter() - started
    (out / 'render-timings.json').write_text(json.dumps(timings, indent=2), encoding='utf-8')
    print('Full movie assembled', flush=True)
    report_progress('video.complete', '视频已生成', done=1, total=1, force=True)


if __name__ == '__main__':
    main(json.loads(Path(sys.argv[sys.argv.index('--') + 1]).read_text(encoding='utf-8')), '--resume' in sys.argv)
