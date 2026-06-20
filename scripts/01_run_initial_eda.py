"""Run the first descriptive analysis."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

matplotlib.use("Agg")

from credit_default.eda import run_initial_eda  # noqa: E402


def main() -> None:
    outputs = run_initial_eda()
    for name, path in outputs.items():
        print(f"{name}: {path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
