from __future__ import annotations

import sys
import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
