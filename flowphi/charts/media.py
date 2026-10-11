from __future__ import annotations

import csv
import json
from pathlib import Path


def read_info(path):
    path = Path(path)
    if not path.is_file():
        return {}
    result = {}
    for row in path.read_text(encoding='utf-8-sig').splitlines():
        if ':' not in row or row.lstrip().startswith('#'):
            continue
        key, value = row.split(':', 1)
        value = value.strip()
        try:
            value = json.loads(value)
        except ValueError:
            value = int(value, 16) if value.startswith('0x') else value.strip('\"\'')
        result[key.strip().lower()] = value
    return result


def find_media(chart_path, directory=None, *, metadata=None):
    chart_path = Path(chart_path).resolve()
    root = Path(directory).resolve() if directory is not None else chart_path.parent
    files = sorted(p for p in (root.rglob('*') if directory is not None else root.iterdir()) if p.is_file())
    by_name = {}
    for path in files:
        by_name.setdefault(path.name.casefold(), []).append(path)
    info = {str(k).lower(): v for k, v in (metadata or {}).items()}
    for folder in dict.fromkeys((root, chart_path.parent)):
        for name in ('info.yml', 'info.yaml', 'info.txt'):
            info.update(read_info(folder / name))
        table = folder / 'info.csv'
        if table.is_file():
            for row in csv.DictReader(table.read_text(encoding='utf-8-sig').splitlines()):
                row = {k.lower(): v for k, v in row.items() if k is not None}
                if Path(row.get('chart', '')).name.casefold() == chart_path.name.casefold():
                    info.update(row)
                    break

    def named(value, suffixes):
        if not isinstance(value, str) or not value:
            return None
        for folder in (chart_path.parent, root):
            candidate = (folder / value).resolve()
            if candidate.is_relative_to(root) and candidate.is_file() and candidate.suffix.lower() in suffixes:
                return candidate
        matches = [p for p in by_name.get(Path(value).name.casefold(), ()) if p.suffix.lower() in suffixes]
        return matches[0] if len(matches) == 1 else None

    def choose(keys, suffixes, preferred):
        for key in keys:
            if candidate := named(info.get(key), suffixes):
                return candidate
        for name in preferred:
            if candidate := named(name, suffixes):
                return candidate
        candidates = [p for p in files if p.suffix.lower() in suffixes]
        return candidates[0] if len(candidates) == 1 else None

    audio_types = ('.wav', '.ogg', '.mp3', '.flac')
    image_types = ('.png', '.jpg', '.jpeg', '.webp')
    music = choose(('song', 'music', 'audio'), audio_types,
                   [stem + ext for stem in ('music', chart_path.stem) for ext in audio_types])
    picture = choose(('illustration', 'picture', 'image', 'background'), image_types,
                     [stem + ext for stem in ('illustration', chart_path.stem, 'background', 'cover', 'defaultImage')
                      for ext in image_types])
    return music, picture
