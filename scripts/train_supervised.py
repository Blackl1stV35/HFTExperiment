#!/usr/bin/env python3
"""Train the dual-branch model.

Usage:
    python scripts/train_supervised.py model=dual_branch data=xauusd
    python scripts/train_supervised.py data.labeling.method=hybrid training.learning_rate=0.0002
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.training.train_supervised import main

if __name__ == "__main__":
    main()
