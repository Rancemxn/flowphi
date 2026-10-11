import os
import sys
from pathlib import Path

PACKAGE = Path(__file__).resolve().parent
ROOT = Path(sys.executable).resolve().parent if getattr(sys, 'frozen', False) else PACKAGE.parent
WORKSPACE = Path(os.environ.get('FLOWPHI_WORKSPACE', str(ROOT))).expanduser().resolve()
DEFAULT_CACHE = WORKSPACE / '.local' / 'editor-cache'
ASSETS = WORKSPACE / 'assets'
PROJECTS = WORKSPACE / 'projects'
OUTPUTS = WORKSPACE / 'output'
