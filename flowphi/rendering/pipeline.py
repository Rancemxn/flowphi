from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import struct
import subprocess
import zipfile

from rich.progress import BarColumn, MofNCompleteColumn, Progress, TimeElapsedColumn, TimeRemainingColumn

from flowphi.planning.touch import TouchAction, load_data
from flowphi.charts.model import NoteType
from flowphi.charts.loader import load_chart
from flowphi.charts.media import find_media
from flowphi.progress import report_log, report_progress, structured_output
from flowphi.runtime import NO_WINDOW


def progress_bar():
    return Progress('[progress.description]{task.description}', BarColumn(), '[progress.percentage]{task.percentage:>3.0f}%',
                    MofNCompleteColumn(), TimeElapsedColumn(), TimeRemainingColumn(), disable=structured_output())


def clip_frames(start, duration, fps, music_duration):
    count = math.ceil(duration * fps)
    audio_end = math.ceil(music_duration * fps) / fps
    if start + count / fps > audio_end + 1e-9:
        raise ValueError('Requested clip extends beyond the audio')
    return count


def run_blender(command, log_path, description, total, *, stage='animation'):
    """Retain the full child log and forward live progress to the desktop worker."""
    with log_path.open('w', encoding='utf-8') as log, progress_bar() as progress:
        task = progress.add_task(description, total=total)
        report_log('子进程日志：' + str(log_path))
        report_progress(stage, description, done=0, total=total, unit='帧', force=True)
        structured_seen = False
        with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              text=True, encoding='utf-8', errors='replace', bufsize=1,
                              creationflags=NO_WINDOW) as process:
            try:
                for line in process.stdout:
                    log.write(line)
                    log.flush()
                    if line.startswith('HANDCAM_PROGRESS '):
                        completed = int(line.split()[1])
                        progress.update(task, completed=min(completed, total))
                        if not structured_seen:
                            report_progress(stage, description, done=min(completed, total), total=total, unit='帧')
                    elif line.startswith('HANDCAM_STATUS '):
                        status = line.strip().split(' ', 1)[1]
                        progress.update(task, description=status)
                        joints = re.fullmatch(r'Baking joints (\d+)/(\d+)', status)
                        if joints:
                            report_progress(stage + '.joints', '写入关节关键帧',
                                            done=int(joints[1]), total=int(joints[2]), unit='组')
                        else:
                            report_progress(stage + '.save', '保存手部动画' if status == 'Saving animation' else status, force=True)
                    elif structured_output():
                        try:
                            entry = json.loads(line)
                        except ValueError:
                            entry = None
                        if isinstance(entry, dict) and entry.get('event') == 'progress':
                            structured_seen = True
                        print(line.rstrip(), flush=True)
                if process.wait():
                    raise RuntimeError(f'{description} failed; see {log_path}')
            except BaseException:
                if process.poll() is None:
                    process.terminate()
                raise
        progress.update(task, completed=total)
        if not structured_seen:
            report_progress(stage + '.complete', description + '完成', done=1, total=1, force=True)


def render_saved(out, blender, resume=False):
    job = json.loads((out / 'job.json').read_text(encoding='utf-8'))
    if Path(job['output']).resolve() != out:
        raise ValueError('Saved job belongs to a different output directory')
    for path in (out / 'handcam.blend', *(Path(job[key]) for key in ('screen_video', 'mixed_audio', 'background'))):
        if not path.is_file():
            raise ValueError(f'Render input missing: {path}')
    command = [blender, '--background', '--disable-autoexec', str(out / 'handcam.blend'),
               '--python-exit-code', '1', '--python', str(Path(__file__).with_name('video.py')),
               '--', str(out / 'job.json')]
    if resume:
        command.append('--resume')
    run_blender(command, out / ('render-resume.log' if resume else 'render.log'),
                '渲染与合成视频', job['frames'] * 3, stage='render')


def contacts_from_psap(content: bytes, time_offset: float = 0.) -> tuple[tuple[int, int], list[dict]]:
    if not math.isfinite(time_offset):
        raise ValueError('Chart offset must be finite')
    # Validate before the legacy reader, whose header check uses assert.
    if len(content) < 12 or content[:4] != b'PSAP':
        raise ValueError('Not a PSAP file')
    offset = 12
    while offset < len(content):
        if offset + 5 > len(content):
            raise ValueError('Truncated PSAP timestamp')
        count = content[offset + 4]
        if count > 127 or offset + 5 + count * 21 > len(content):
            raise ValueError('Invalid or truncated PSAP events')
        offset += 5 + count * 21
    screen, frames = load_data(content)
    if screen.width <= 0 or screen.height <= 0:
        raise ValueError('Invalid PSAP screen size')
    active, contacts = {}, []
    previous = -math.inf
    for timestamp, events in frames:
        if timestamp < previous:
            raise ValueError('PSAP timestamps must be ordered')
        previous = timestamp
        t = timestamp / 1000 + time_offset
        for event in events:
            pid, action = event.pointer_id, event.action
            x, y = event.pos.real / screen.width, event.pos.imag / screen.height
            if not all(math.isfinite(v) for v in (x, y)) or not (0 <= x <= 1 and 0 <= y <= 1):
                raise ValueError(f'Pointer {pid} has an invalid screen position at {t}s')
            if action == TouchAction.DOWN:
                if pid in active:
                    raise ValueError(f'Duplicate DOWN for pointer {pid} at {t}s')
                contact = {'pointer': pid, 'start': t, 'end': None, 'points': [[t, x, y]]}
                contacts.append(contact)
                active[pid] = contact
            elif action == TouchAction.MOVE:
                if pid not in active:
                    raise ValueError(f'MOVE without DOWN for pointer {pid} at {t}s')
                points = active[pid]['points']
                if points[-1][0] == t:
                    if points[-1][1:] != [x, y]:
                        raise ValueError(f'Conflicting positions at {t}s for pointer {pid}')
                else:
                    points.append([t, x, y])
            elif action in (TouchAction.UP, TouchAction.CANCEL):
                if pid not in active:
                    raise ValueError(f'UP without DOWN for pointer {pid} at {t}s')
                contact = active.pop(pid)
                contact['end'] = t
                # Some planners put a stale position in UP. Releasing does not teleport a fingertip.
            else:
                raise ValueError(f'Unsupported PSAP action: {action.name}')
    if active:
        raise ValueError(f'PSAP ends with unreleased pointers: {sorted(active)}')
    return (screen.width, screen.height), contacts


def extract_archive(source: Path, destination: Path):
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(source) as archive:
        # Extract only files inside this dedicated directory; reject traversal and zip bombs.
        root = destination.resolve()
        if sum(i.file_size for i in archive.infolist()) > 512 * 1024 * 1024:
            raise ValueError('Chart archive exceeds 512 MiB unpacked')
        for info in archive.infolist():
            target = (root / info.filename).resolve()
            if not target.is_relative_to(root):
                raise ValueError(f'Unsafe archive path: {info.filename}')
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(info) as src, target.open('wb') as dst:
                    shutil.copyfileobj(src, dst)


def unpack(source: Path, destination: Path) -> list[Path]:
    if not zipfile.is_zipfile(source):
        return [source]
    extract_archive(source, destination)
    candidates = sorted(p for p in destination.rglob('*') if p.suffix.lower() in ('.json', '.pec'))
    charts = []
    for path in candidates:
        try:
            load_chart(path.read_text(encoding='utf-8-sig'), source=path)
            charts.append(path)
        except (ValueError, KeyError, TypeError):
            continue
    if not charts:
        raise ValueError('压缩包中没有可读取的 JSON 或 PEC 谱面')
    return charts


def prepare_resources(source: Path, destination: Path):
    if source.is_dir():
        return source.resolve()
    if not source.is_file() or not zipfile.is_zipfile(source):
        raise ValueError(f'Resources must be a directory or ZIP: {source}')
    extract_archive(source, destination)
    packs = [p.parent for p in destination.rglob('info.yml') if (p.parent / 'click.png').is_file()]
    if len(packs) != 1:
        raise ValueError('Resource ZIP must contain one pack with info.yml and click.png')
    return packs[0].resolve()


def find_blender(value: str | None) -> str:
    if value:
        executable = shutil.which(value) or (str(Path(value).resolve()) if Path(value).is_file() else None)
    else:
        executable = shutil.which('blender')
        if not executable:
            installed = sorted(Path('C:/Program Files/Blender Foundation').glob('Blender */blender.exe'))
            executable = str(installed[-1]) if installed else None
    if not executable:
        raise ValueError('Blender not found; specify --blender /path/to/blender')
    return executable


def render_screen(chart, image_path, directory, job):
    import skia
    from flowphi.rendering.chart import PlayfieldRenderer, background_image

    directory.mkdir(parents=True, exist_ok=True)
    from flowphi.rendering.blender import screen_rect
    _, _, width, height = screen_rect(job)
    report_progress('screen.prepare', '准备谱面画面', detail=f'{width}×{height}', force=True)
    renderer = PlayfieldRenderer(chart, width, height, job['resources'], job['title'], job['level'], job['music_duration'])
    surface = skia.Surface(width, height)
    canvas = surface.getCanvas()
    background = background_image(image_path, width, height, job['background_dim'], height * .015)
    streaming = job.get('screen_video')
    selected = {1, job['frames'] // 4 + 1, job['frames'] // 2 + 1, job['frames'] * 3 // 4 + 1, job['frames']}
    process = None
    if streaming:
        process = subprocess.Popen(['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pixel_format', 'bgra',
                                    '-video_size', f'{width}x{height}', '-framerate', str(job['fps']), '-i', '-',
                                    '-an', '-c:v', 'libx264', '-preset', 'ultrafast', '-crf', '12', '-pix_fmt', 'yuv420p',
                                    str(directory.parent / 'screen.partial.mp4')], stdin=subprocess.PIPE,
                                    creationflags=NO_WINDOW)
    try:
        with progress_bar() as progress:
            task = progress.add_task('Chart video', total=job['frames'])
            report_progress('screen', '生成谱面视频', done=0, total=job['frames'], unit='帧', force=True)
            for index in range(job['frames']):
                canvas.drawImage(background, 0, 0)
                renderer.draw(canvas, job['start'] + index / job['fps'] - chart.offset)
                image = surface.makeImageSnapshot()
                if process:
                    process.stdin.write(image.tobytes())
                if not process or index + 1 in selected:
                    image.save(str(directory / f'{index + 1:05d}.png'))
                progress.update(task, advance=1)
                report_progress('screen', '生成谱面视频', done=index + 1, total=job['frames'], unit='帧')
    finally:
        if process:
            process.stdin.close()
            if process.wait():
                raise RuntimeError('Screen video encoding failed')
    if process:
        (directory.parent / 'screen.partial.mp4').replace(streaming)
    print(f'Screen: {job["frames"]} frames (Phigros Autoplay)', flush=True)
    return renderer


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('chart', type=Path, help='chart ZIP/PEZ, JSON or PEC')
    parser.add_argument('--chart-entry', help='path of the selected chart inside a ZIP/PEZ')
    parser.add_argument('--model', type=Path, required=True, help='Hands + armature.blend')
    parser.add_argument('--audio', type=Path, help='overrides the music in the chart archive')
    parser.add_argument('--psap', type=Path, required=True, help='Touch sequence exported by the editor')
    parser.add_argument('--motion-plan', type=Path, required=True, help='Matching authored motion-plan.json')
    parser.add_argument('--strict-psap', action='store_true', help='display every raw contact, including redundant overlaps')
    parser.add_argument('--video', type=Path, help='full-song Autoplay video, aligned with the music')
    parser.add_argument('--background', type=Path, help='illustration for the playfield and blurred surround')
    parser.add_argument('--resources', type=Path, required=True,
                        help='Phira resource ZIP, pack directory, or assets directory')
    parser.add_argument('--title', help='override song title from chart metadata')
    parser.add_argument('--level', help='override chart difficulty label')
    parser.add_argument('--screen-fill', type=float, default=.82, help='playfield fraction of output width (default: .82)')
    parser.add_argument('--background-dim', type=float, default=.45, help='playfield illustration dimming, 0–1')
    parser.add_argument('--hitsound-volume', type=float, default=.35, help='hit sound gain, 0 disables (default: .35)')
    parser.add_argument('--start', type=float, default=0., help='music time in seconds')
    parser.add_argument('--duration', type=float, default=10.)
    parser.add_argument('--full', action='store_true', help='render the complete music, including the final partial frame')
    parser.add_argument('--fps', type=int, default=30, help='output frame rate (default: 30)')
    parser.add_argument('--width', type=int, default=1280, help='output width in pixels (default: 1280)')
    parser.add_argument('--height', type=int, default=720, help='output height in pixels (default: 720)')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--blender')
    parser.add_argument('--engine', choices=('workbench', 'eevee', 'cycles'), default='workbench')
    parser.add_argument('--hand-scale', type=float, default=.27, help='model scale; source model units to metres (default: .27)')
    parser.add_argument('--contact-height', type=float, default=.0005, help='fingertip skin clearance in metres')
    parser.add_argument('--lift-height', type=float, help='maximum airborne finger lift in metres (default: .045)')
    parser.add_argument('--prepare-only', action='store_true', help='write PSAP, motion job and screen frames')
    parser.add_argument('--plan-only', action='store_true', help='write touch/motion plans and diagnostics without baking or rendering')
    parser.add_argument('--keyframes', action='store_true', help='render five inspection frames instead of the video')
    parser.add_argument('--bake-only', action='store_true', help='solve and save the full hand animation without rendering media')
    parser.add_argument('--warmup', type=float, default=1., help='seconds of motion history before a bake segment')
    parser.add_argument('--consistent-history', action='store_true',
                        help='replay from the shared -1 second origin and retain full contact context for parallel bakes')
    parser.add_argument('--screen-video', action='store_true', help='stream chart frames into video instead of saving every PNG')
    parser.add_argument('--stream', action='store_true', help='render a baked animation in small video chunks, saving disk space')
    parser.add_argument('--resume', action='store_true', help='resume video rendering from the saved job in --output; uses saved settings')
    args = parser.parse_args(argv)
    if args.lift_height is None:
        args.lift_height = .045
    if args.motion_plan and not args.psap:
        parser.error('--motion-plan requires its matching --psap')
    if not math.isfinite(args.warmup) or not 0 <= args.warmup <= 20:
        parser.error('--warmup must be between 0 and 20 seconds')
    if args.stream and (args.video or args.keyframes or args.bake_only or args.engine != 'workbench'):
        parser.error('--stream requires Workbench and built-in chart rendering; omit --video, --keyframes and --bake-only')
    if not all(math.isfinite(v) for v in (args.start, args.duration, args.hand_scale, args.contact_height, args.lift_height,
                                        args.screen_fill, args.background_dim, args.hitsound_volume)):
        parser.error('Times and scales must be finite')
    if args.start < 0 or args.duration <= 0 or args.hand_scale <= 0 or args.contact_height < 0 or args.lift_height < 0:
        parser.error('Invalid time, hand scale or contact height')
    if not (1 <= args.fps <= 120) or min(args.width, args.height) < 64 or args.width % 2 or args.height % 2:
        parser.error('Use 1–120 fps and even dimensions of at least 64 pixels')
    if not .4 <= args.screen_fill <= .95 or not 0 <= args.background_dim <= 1 or not 0 <= args.hitsound_volume <= 2:
        parser.error('Use screen fill .4–.95, background dim 0–1, and hit sound gain 0–2')
    try:
        if args.resume:
            out = args.output.resolve()
            render_saved(out, find_blender(args.blender), resume=True)
            print(f'Video: {out / "handcam.mp4"}', flush=True)
            return 0
        for path in (args.chart, args.model, args.audio, args.psap, args.motion_plan, args.video, args.background):
            if path is not None and not path.is_file():
                raise ValueError(f'File does not exist: {path}')
        if not shutil.which('ffmpeg') or not shutil.which('ffprobe'):
            raise ValueError('FFmpeg and ffprobe are required on PATH')
        blender = find_blender(args.blender)
        out = args.output.resolve()
        if args.model.resolve() == out / 'handcam.blend' or (args.video and args.video.resolve() == out / 'handcam.mp4'):
            raise ValueError('Output would overwrite an input model or video; choose another --output directory')
        out.mkdir(parents=True, exist_ok=True)
        report_progress('render.inputs', '准备渲染资源', force=True)
        charts = unpack(args.chart.resolve(), out / 'input')
        if args.chart_entry:
            chart_path = (out / 'input' / args.chart_entry).resolve()
            if chart_path not in charts:
                raise ValueError(f'压缩包中找不到所选谱面：{args.chart_entry}')
        elif len(charts) == 1:
            chart_path = charts[0]
        else:
            entries = '\n'.join(path.relative_to(out / 'input').as_posix() for path in charts)
            raise ValueError('压缩包包含多个谱面，请通过 --chart-entry 选择：\n' + entries)
        content = chart_path.read_text(encoding='utf-8-sig')
        chart = load_chart(content, source=chart_path)
        meta = json.loads(content).get('META', {}) if chart.format == 'rpe' else {}
        media_root = out / 'input' if zipfile.is_zipfile(args.chart) else None
        music, picture = find_media(chart_path, media_root, metadata=meta)
        picture = args.background.resolve() if args.background else picture
        music = args.audio.resolve() if args.audio else music
        report_log('曲绘：' + (str(picture) if picture else '未找到，将使用纯色背景'))
        if music is None:
            raise ValueError('Provide --audio, or a chart archive with one music file')
        probe = json.loads(subprocess.check_output([
            'ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'json', str(music),
        ], text=True, creationflags=NO_WINDOW))
        music_duration = float(probe['format']['duration'])
        if args.full:
            args.start, args.duration = 0., music_duration
        frame_count = clip_frames(args.start, args.duration, args.fps, music_duration)
        duration = frame_count / args.fps
        from flowphi.rendering.chart import background_image, mix_audio, read_info
        from flowphi.rendering.blender import fit_screen_in_frame
        metadata = read_info(chart_path.parent / 'info.yml') | read_info(chart_path.parent / 'info.txt')
        if chart.format == 'rpe':
            metadata = {key.lower(): value for key, value in meta.items()} | metadata
        if not math.isfinite(chart.offset):
            raise ValueError('Chart offset must be finite')
        content = args.psap.read_bytes()
        motion_plan = json.loads(args.motion_plan.read_text(encoding='utf-8'))
        # Convert chart time to music time once, including fractional offsets.
        dimensions, contacts = contacts_from_psap(content, chart.offset)
        if motion_plan:
            from flowphi.motion import attach_plan
            profile = motion_plan['profile']
            if not math.isclose(profile['scale'], args.hand_scale, rel_tol=1e-9):
                raise ValueError('Motion plan uses a different --hand-scale; generate a new draft in the editor')
            if profile.get('model_sha256') and profile['model_sha256'] != hashlib.sha256(args.model.read_bytes()).hexdigest():
                raise ValueError('Motion plan was measured for a different hand model; generate a new draft in the editor')
            expected_screen = [.28, .28 * chart.height / chart.width]
            if any(not math.isclose(a, b, rel_tol=1e-9) for a, b in zip(motion_plan['physical_screen'], expected_screen)):
                raise ValueError('Motion plan uses a different physical screen size')
            contacts = attach_plan(content, contacts, motion_plan, chart.offset)
            (out / 'motion-plan.json').write_text(json.dumps(motion_plan, indent=2), encoding='utf-8')
            (out / 'planning-diagnostics.json').write_text(json.dumps(motion_plan['diagnostics'], indent=2), encoding='utf-8')
            if motion_plan['diagnostics']['degraded']:
                print(f'Planning: {len(motion_plan["diagnostics"]["degraded"])} contacts exceed motion estimates; '
                      'see planning-diagnostics.json', flush=True)
        # A validated excerpt may contain only a rest (for example, the intro).
        # The authored plan still supplies the hand profile and resting poses.
        if not contacts and motion_plan.get('authoring', {}).get('time_range') is None:
            raise ValueError('The plan has no contacts')
        if not math.isclose(dimensions[0] / dimensions[1], chart.width / chart.height, rel_tol=.001):
            raise ValueError('PSAP and chart aspect ratios differ')
        (out / 'plan.psap').write_bytes(content)
        if args.plan_only:
            print(f'Plan: {out / "plan.psap"}' + (f' + {out / "motion-plan.json"}' if motion_plan else ''), flush=True)
            return 0
        if not motion_plan and not any(c['end'] >= args.start and c['start'] < args.start + duration for c in contacts):
            raise ValueError('No contacts overlap the requested clip')
        # Keep the authored rest schedule while solving only nearby fingers.
        if args.consistent_history:
            args.warmup = args.start + 1.
        else:
            contacts = [c for c in contacts if c['end'] >= args.start - args.warmup and c['start'] <= args.start + duration + 1]
        screen_w = .28
        screen_h = screen_w * chart.height / chart.width
        resources = prepare_resources(args.resources.resolve(), out / 'resources') if not args.video else args.resources.resolve()
        job = dict(start=args.start, duration=duration, frames=frame_count, fps=args.fps, warmup=args.warmup,
                   width=args.width, height=args.height, output=str(out), model=str(args.model.resolve()),
                   contacts=contacts, screen=[screen_w, screen_h], view_width=screen_w / args.screen_fill, camera_y=-.015,
                   resources=str(resources), illustration=str(picture) if picture else None,
                   title=args.title if args.title is not None else metadata.get('name', args.chart.stem),
                   level=args.level if args.level is not None else metadata.get('level', ''),
                   music_duration=music_duration, background_dim=args.background_dim,
                   renderer='external' if args.video else 'phigros-autoplay', hitsound_volume=args.hitsound_volume,
                   hold_starts=[n.seconds + chart.offset for line in chart.lines for n in line.notes if n.type == NoteType.HOLD],
                   hold_intervals=[(n.seconds + chart.offset, n.seconds + n.hold + chart.offset)
                                   for line in chart.lines for n in line.notes if n.type == NoteType.HOLD],
                   engine=args.engine, hand_scale=args.hand_scale, contact_height=args.contact_height,
                   lift_height=args.lift_height, strict_psap=args.strict_psap,
                   keyframes=args.keyframes, bake_only=args.bake_only or args.stream, audio=str(music), source=str(args.chart.resolve()),
                   algorithm=motion_plan.get('algorithm'))
        if motion_plan:
            rests = copy.deepcopy(motion_plan['hand_rest'])
            for gap in rests:
                gap['start'] += chart.offset
                gap['end'] += chart.offset
                for point in gap['knots']:
                    point[0] += chart.offset
            job.update(motion_plan_version=motion_plan['version'], hand_rest=rests,
                       hand_profile=motion_plan.get('profile'),
                       palm_motion=motion_plan.get('palm_motion'),
                       finger_motion='whole_finger' if motion_plan.get('palm_motion') == 'v4' else None,
                       palm_lift_ratio=.70 if motion_plan.get('palm_motion') == 'v4' else None,
                       palm_lift_mode='gentle' if motion_plan.get('palm_motion') == 'v4' else None,
                       palm_strike_speed=1.8, stroke_peak_phase=.62,
                       palm_lift_low=.042, palm_lift_high=.035,
                       stroke_gap_limit=1., transfer_sway=.004,
                       camera_projection='perspective' if motion_plan.get('palm_motion') == 'v4' else 'orthographic',
                       camera_height=.65,
                       camera_offset_y=-.15,
                       enable_thumb=True, pose_avoidance=True, render_speed_limits=False,
                       avoidance_clearance=0., airborne_order_clearance=.003,
                       wrist_speed=motion_plan['settings']['wrist_speed'])
            job['pose'] = copy.deepcopy(motion_plan.get('pose'))
            job['pose_time_offset'] = chart.offset
            guides = copy.deepcopy(motion_plan.get('pose_guides', []))
            for guide in guides:
                for name in ('prepare', 'start', 'end', 'release'):
                    guide[name] += chart.offset
            job['pose_guides'] = guides
            for name in ('palm_lift_low', 'palm_lift_high', 'palm_strike_speed', 'transfer_sway', 'finger_lateral_limit'):
                if name in motion_plan.get('pose_style', {}):
                    job[name] = motion_plan['pose_style'][name]
        _, _, sw, sh = fit_screen_in_frame(job)
        if not args.bake_only:
            report_progress('background', '准备曲绘背景', force=True)
            background_image(picture, args.width, args.height, .2, args.height * .045).save(str(out / 'background.png'))
        job['background'] = str(out / 'background.png')
        if (args.screen_video or args.stream) and not args.video:
            job['screen_video'] = str(out / 'screen.mp4')
        (out / 'job.json').write_text(json.dumps(job, indent=2), encoding='utf-8')
        if args.bake_only:
            pass
        elif args.video:
            (out / 'screen').mkdir(exist_ok=True)
            # A shorter new input must not silently reuse frames from a previous export.
            for frame in (out / 'screen').glob('[0-9][0-9][0-9][0-9][0-9].png'):
                frame.unlink()
            subprocess.run(['ffmpeg', '-v', 'error', '-y', '-ss', str(args.start), '-i', str(args.video),
                            '-vf', f'fps={args.fps},scale={sw}:{sh}', '-frames:v', str(frame_count),
                            str(out / 'screen/%05d.png')], check=True, creationflags=NO_WINDOW)
            if not (out / 'screen' / f'{frame_count:05d}.png').exists():
                raise ValueError('Autoplay video is shorter than the requested clip')
        else:
            renderer = render_screen(chart, picture, out / 'screen', job)
            report_progress('audio', '混合音乐与打击音', force=True)
            audio_stats = mix_audio(music, resources, renderer.hits, args.start, duration,
                                    args.hitsound_volume, out / 'audio.wav')
            job['mixed_audio'] = str(out / 'audio.wav')
            (out / 'audio-mix.json').write_text(json.dumps(audio_stats, indent=2), encoding='utf-8')
        (out / 'job.json').write_text(json.dumps(job, indent=2), encoding='utf-8')
        for warning in chart.warnings:
            print(f'Chart warning: {warning}', flush=True)
        print(f'Prepared {len(contacts)} contact lifetimes in {out}', flush=True)
        if args.prepare_only:
            return 0
        command = [blender, '--background', '--factory-startup', '--disable-autoexec',
                   str(args.model.resolve()), '--python-exit-code', '1', '--python',
                   str(Path(__file__).with_name('blender.py').resolve()), '--', str(out / 'job.json')]
        run_blender(command, out / 'blender.log', '计算手部动作', job['frames'])
        diagnostics = json.loads((out / 'diagnostics.json').read_text(encoding='utf-8'))
        if diagnostics['shared_events']:
            print(f'Animation: {len(diagnostics["shared_events"])} redundant events shared by an existing contact; '
                  'see motion.json, or use --strict-psap to keep every raw contact', flush=True)
        if diagnostics.get('idle_releases'):
            print(f'Animation: {len(diagnostics["idle_releases"])} idle pointer intervals released; see motion.json', flush=True)
        if diagnostics['contact_errors_over_1mm']:
            report_log(f'指腹接触偏差最大 {diagnostics["max_error_mm"]:.2f} mm，'
                       f'{len(diagnostics["contact_errors_over_1mm"])} 个采样超过 1 mm详情见 {out / "diagnostics.json"}', 'warning')
        if diagnostics['collision_errors']:
            report_log(f'手部仍有未解决的碰撞，最大深度 {diagnostics["max_collision_depth_mm"]:.2f} mm'
                       f'详情见 {out / "diagnostics.json"}', 'warning')
        if diagnostics['min_screen_clearance_mm'] is not None and diagnostics['min_screen_clearance_mm'] < 0:
            report_log(f'手部表面最多穿过屏幕 {-diagnostics["min_screen_clearance_mm"]:.2f} mm，请检查接触位置和手部姿态', 'warning')
        if args.keyframes:
            print(f'Inspection frames: {out / "inspection"}', flush=True)
            return 0
        if args.bake_only:
            print(f'Animation: {out / "handcam.blend"}', flush=True)
            return 0
        if args.stream:
            render_saved(out, blender)
        else:
            from flowphi.rendering.blender import composite_command
            subprocess.run(composite_command(job), check=True, creationflags=NO_WINDOW)
            (out / 'handcam.partial.mp4').replace(out / 'handcam.mp4')
        print(f'Video: {out / "handcam.mp4"}', flush=True)
        return 0
    except (ValueError, OSError, RuntimeError, struct.error, subprocess.CalledProcessError) as error:
        parser.exit(1, f'flowphi video export: {error}\n')


if __name__ == '__main__':
    raise SystemExit(main())
