from __future__ import annotations

import math

from PySide6.QtCore import QSettings


# These are the same units and choices exposed by the editor's controls.
PREFERENCES = {
    'preview_fps': (30, 10, 60), 'preview_width': (960, 480, 1920),
    'lookahead': (8, 2, 30), 'timeline_span': (12, 2, 120),
    'replan_beam': (32, 1, 128), 'replan_contacts': (1200, 32, 5000),
    'history_mib': (16, 4, 128), 'ui_scale': (100, 50, 200),
    'solver_beam': (8, 1, 32), 'solver_candidates': (3, 1, 8),
    'generator_fingers': ('all', ['index', 'index_middle', 'all']),
    'fingering_objective': ('balanced', ['balanced', 'load']),
    'speed_limits': (True,), 'judgement_windows': (False,), 'keep_unplanned': (True,),
    'clip_resolution': ('640x360', ['640x360', '1280x720', '1920x1080']),
    'render_resolution': ('1280x720', ['640x360', '1280x720', '1920x1080']),
    'clip_fps': (30, [15, 30, 60]), 'render_fps': (30, [15, 30, 60]),
    'playback_rate': (1., [.1, .25, .5, .75, 1., 1.5, 2.]),
    'range_start': (0., 0., 86400.), 'range_end': (5., 0., 86400.),
    'loop': (False,), 'compare': (False,), 'pose_view': ('angle', ['angle', 'top', 'side']),
    'auto_check': (False,), 'model': ('',), 'resources': ('',), 'blender': ('',),
}


def validate_preferences(values):
    if not isinstance(values, dict) or set(values) - PREFERENCES.keys():
        raise ValueError('包含未知的编辑器设置')
    for key, value in values.items():
        default, *limits = PREFERENCES[key]
        valid = type(value) is type(default) or (type(default) is float and type(value) is int)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            valid = valid and math.isfinite(value)
        if len(limits) == 1:
            valid = valid and value in limits[0]
        elif len(limits) == 2:
            valid = valid and limits[0] <= value <= limits[1]
        if not valid:
            raise ValueError(f'设置 {key} 的类型、范围或选项无效')
    if 'range_start' in values and 'range_end' in values and values['range_end'] <= values['range_start']:
        raise ValueError('片段结束时间必须晚于开始时间')
    return values


def read_preferences(settings):
    values = {}
    for key, (default, *_) in PREFERENCES.items():
        try:
            value = settings.value(key, default, type=type(default))
            validate_preferences({key: value})
        except (TypeError, ValueError, OverflowError):
            value = default
        values[key] = value
    if values['range_end'] <= values['range_start']:
        values.update(range_start=0., range_end=5.)
    return values


def current_preferences(editor):
    return read_preferences(editor.settings) | editor.performance | dict(
        playback_rate=editor.speed.currentData(), loop=editor.loop.isChecked(),
        range_start=editor.range_start.value(), range_end=editor.range_end.value(),
        compare=editor.compare_action.isChecked(), pose_view=editor.pose_view.currentData())


def save_preferences(editor):
    for key, value in current_preferences(editor).items():
        editor.settings.setValue(key, value)
    editor.settings.sync()
    if editor.settings.status() != QSettings.Status.NoError:
        raise OSError('无法保存编辑器设置')


def apply_preferences(editor, values):
    values = validate_preferences(current_preferences(editor) | values)
    for key, value in values.items():
        editor.settings.setValue(key, value)
    editor.performance.update({key: values[key] for key in editor.performance})
    editor._apply_performance()
    editor.speed.setCurrentIndex(editor.speed.findData(values['playback_rate']))
    editor.loop.setChecked(values['loop'])
    editor.range_start.setValue(values['range_start'])
    editor.range_end.setValue(values['range_end'])
    editor.pose_view.setCurrentIndex(editor.pose_view.findData(values['pose_view']))
    editor.compare_baseline(values['compare'])
    editor.auto_check_action.setChecked(values['auto_check'])
    save_preferences(editor)
