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
        stream = getattr(sys, name)
        if stream is not None:
            if mode == 'w' and hasattr(stream, 'reconfigure'):
                # Frozen Python can keep the Windows code page even with PYTHONUTF8 set.
                stream.reconfigure(encoding='utf-8', errors='replace', line_buffering=True)
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


def configure_logging():
    import logging
    from logging.handlers import RotatingFileHandler
    import threading

    from flowphi.config import APP_LOG

    try:
        APP_LOG.parent.mkdir(parents=True, exist_ok=True)
        handler = RotatingFileHandler(APP_LOG, maxBytes=5 * 1024 * 1024, backupCount=2,
                                      encoding='utf-8', errors='backslashreplace')
    except OSError:
        logging.basicConfig(level=logging.INFO)
        logging.exception('无法创建程序日志：%s', APP_LOG)
    else:
        logging.basicConfig(level=logging.INFO, handlers=[handler],
                            format='%(asctime)s %(levelname)s [%(threadName)s] %(name)s: %(message)s')
    logging.captureWarnings(True)
    previous_hook = sys.excepthook

    def exception_hook(kind, error, traceback):
        logging.critical('未处理异常', exc_info=(kind, error, traceback))
        previous_hook(kind, error, traceback)

    sys.excepthook = exception_hook
    threading.excepthook = lambda args: exception_hook(args.exc_type, args.exc_value, args.exc_traceback)
    logging.info('FlowPhi 启动 · Python %s · %s', sys.version.split()[0], sys.executable)
