# tests/conftest.py
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]   # project root
SRC = ROOT / "src"

# Put src at the *front* of sys.path so local code is found first
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
