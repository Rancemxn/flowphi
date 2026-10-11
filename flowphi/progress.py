from __future__ import annotations

import json
import os
import time


_last = {}


def structured_output():
    return os.environ.get('FLOWPHI_PROGRESS_JSON') == '1'


def report_progress(stage, message, *, done=None, total=None, unit='', detail='', force=False):
    if not structured_output():
        return
    now = time.monotonic()
    previous = _last.get(stage)
    signature = message, total
    complete = total is not None and done is not None and done >= total
    if previous and not force and not complete and signature == previous[1] and now - previous[0] < .2:
        return
    _last[stage] = now, signature
    entry = dict(event='progress', stage=stage, message=message, unit=unit, detail=detail)
    if done is not None and total is not None:
        entry.update(done=done, total=total)
    print(json.dumps(entry, ensure_ascii=False), flush=True)


def report_log(message, level='info'):
    if structured_output():
        print(json.dumps(dict(event='log', level=level, message=message), ensure_ascii=False), flush=True)
    else:
        print(message, flush=True)
