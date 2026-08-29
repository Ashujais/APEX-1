from __future__ import annotations

import sys

from apex.cli import main
from apex.compute import KaggleProvider

if __name__ == "__main__":
    KaggleProvider().require_training_runtime()
    sys.argv.insert(1, "evaluate")
    main()
