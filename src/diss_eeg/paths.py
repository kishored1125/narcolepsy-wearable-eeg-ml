from __future__ import annotations

from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[2]
ROOT_DIR = PROJECT_DIR.parent
OUTPUT_DIR = PROJECT_DIR / "outputs"
TABLE_DIR = OUTPUT_DIR / "tables"
FIGURE_DIR = OUTPUT_DIR / "figures"
DOC_DIR = PROJECT_DIR / "docs"


def ensure_output_dirs() -> None:
    for path in [OUTPUT_DIR, TABLE_DIR, FIGURE_DIR, DOC_DIR]:
        path.mkdir(parents=True, exist_ok=True)

