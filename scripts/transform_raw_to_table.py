import json
import sys
from pathlib import Path

import pandas as pd

from _path_utils import safe_run_id


def get_project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def load_raw_jsons(raw_dir: Path) -> list[dict]:
    if not raw_dir.is_dir():
        raise FileNotFoundError(f"Raw directory not found: {raw_dir}")

    paths = list(raw_dir.glob("*.json"))
    if not paths:
        raise FileNotFoundError(f"No JSON files found in {raw_dir}")

    def sort_key(p: Path) -> int:
        return int(p.stem) if p.stem.isdigit() else -1

    rows = []
    for path in sorted(paths, key=sort_key):
        with open(path, encoding="utf-8") as f:
            rows.append(json.load(f))
    return rows


def transform_to_table(rows: list[dict]) -> pd.DataFrame:

    return pd.DataFrame(rows)


def main(run_id: str, project_root: Path | None = None) -> Path:
   
    if project_root is None:
        project_root = get_project_root()

    base = project_root / "artifacts" / safe_run_id(run_id)
    raw_dir = base / "raw"
    rows = load_raw_jsons(raw_dir)
    df = transform_to_table(rows)

    output_path = base / "dataset.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8")

    return output_path


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python transform_raw_to_table.py <run_id> [project_root]")
        sys.exit(1)

    run_id = sys.argv[1]
    project_root = Path(sys.argv[2]) if len(sys.argv) > 2 else None

    out = main(run_id, project_root)
    print(out)
