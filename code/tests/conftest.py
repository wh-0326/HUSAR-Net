"""Use only code included in this repository."""
from pathlib import Path
import sys

CODE = Path(__file__).resolve().parents[1]
HUSARNET = str(CODE / "husarnet")
for directory in (HUSARNET, str(CODE / "legacy")):
    sys.path.insert(0, directory)
