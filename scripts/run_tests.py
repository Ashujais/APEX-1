from __future__ import annotations

from pathlib import Path

import pytest


def main() -> int:
    temp_parent = Path(".test-tmp")
    temp_parent.mkdir(exist_ok=True)
    return pytest.main(["-ra", "-p", "no:cacheprovider", "--basetemp", str(temp_parent / "pytest")])


if __name__ == "__main__":
    raise SystemExit(main())
