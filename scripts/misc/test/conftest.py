"""Put scripts/misc and its tooling on sys.path; stdlib + pytest only (no PyAuto install)."""

import sys
from pathlib import Path

MISC = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(MISC), str(MISC / "tooling")]
