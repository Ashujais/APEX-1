from __future__ import annotations

import json

from apex.hardware import HardwareAdvisor, detect_hardware


def main() -> None:
    report = detect_hardware()
    print(
        json.dumps(
            {"hardware": report.to_dict(), "advice": HardwareAdvisor().recommend(report).to_dict()},
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
