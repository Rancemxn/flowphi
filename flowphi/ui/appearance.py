import os
import sys

from PySide6.QtCore import QSettings

from flowphi.config import WORKSPACE


def app_settings():
    if getattr(sys, 'frozen', False):
        return QSettings(str(WORKSPACE / '.local' / 'settings.ini'), QSettings.IniFormat)
    return QSettings('flowphi', 'FingeringEditor')


def interface_scale(settings=None):
    settings = settings if settings is not None else app_settings()
    try:
        value = int(settings.value('ui_scale', 100))
    except (ValueError, TypeError):
        return 100
    return value if 50 <= value <= 200 else 100


def apply_startup_scale(settings=None):
    scale = interface_scale(settings)
    os.environ['QT_SCALE_FACTOR'] = f'{scale / 100:g}'
    return scale
