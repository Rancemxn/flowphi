from __future__ import annotations

import os
import sys


WORKER_MODULES = frozenset(('flowphi.ui.worker', 'flowphi.rendering.pipeline'))


def module_command(module, *arguments):
    if module not in WORKER_MODULES:
        raise ValueError(f'Unsupported worker module: {module}')
    prefix = (['--run-module', module] if getattr(sys, 'frozen', False)
              else ['-u', '-X', 'utf8', '-m', module])
    return [sys.executable, *prefix, *map(str, arguments)]


def restore_standard_streams():
    for name, handle_id, mode in (('stdin', -10, 'r'), ('stdout', -11, 'w'), ('stderr', -12, 'w')):
        if getattr(sys, name) is not None:
            continue
        stream = None
        if sys.platform == 'win32':
            import ctypes
            from ctypes import wintypes
            import msvcrt

            get_handle = ctypes.WinDLL('kernel32', use_last_error=True).GetStdHandle
            get_handle.argtypes = [wintypes.DWORD]
            get_handle.restype = wintypes.HANDLE
            handle = get_handle(handle_id)
            if handle not in (None, wintypes.HANDLE(-1).value):
                try:
                    flags = os.O_RDONLY if mode == 'r' else os.O_WRONLY
                    descriptor = msvcrt.open_osfhandle(handle, flags | os.O_BINARY)
                    stream = open(descriptor, mode, encoding='utf-8', errors='replace', buffering=1)
                except OSError:
                    pass
        if stream is None:
            stream = open(os.devnull, mode, encoding='utf-8')
        setattr(sys, name, stream)
        setattr(sys, '__' + name + '__', stream)
