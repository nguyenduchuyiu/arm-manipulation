"""Regression entry point for the physical-contact pick oracle."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.oracle_grasp import main


if __name__ == "__main__":
    main()
