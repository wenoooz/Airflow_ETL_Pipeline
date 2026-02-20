import json
import sys
from pathlib import Path

import pandas as pd


def get_project_root() -> Path:
    """Resolve project root (parent of scripts/)."""
    return Path(__file__).resolve().parent.parent


def load_raw_jsons(raw_dir: Path) -> list[dict]:
    """Load all JSON files from raw directory, sorted by run_index."""
    if not raw_dir.is_dir():
        raise FileNotFoundError(f"Raw directory not found: {raw_dir}")

    rows = []
    for path in sorted(raw_dir.glob("*.json"), key=lambda p: int(p.stem) if p.stem.isdigit() else 0):
        with open(path, encoding="utf-8") as f:
            rows.append(json.load(f))
    return rows


def transform_to_table(rows: list[dict]) -> pd.DataFrame:
    """
    Build a single table from raw JSON rows.

    Each row corresponds to one simulation point with columns:
    timestamp, run_id, run_index, seed, snr_db, channel_type, modulation,
    bler, ber, effective_throughput, num_frames, total_bits, total_bit_errors,
    total_blocks, total_block_errors.
    """
    return pd.DataFrame(rows)


def main(run_id: str, project_root: Path | None = None) -> Path:
    """
    Read raw JSONs, build dataset table, write CSV.

    Args:
        run_id: Run identifier (e.g. from run_plan).
        project_root: Project root path; if None, auto-detect.

    Returns:
        Path to written dataset.csv.
    """
    if project_root is None:
        project_root = get_project_root()

    raw_dir = project_root / "artifacts" / run_id / "raw"
    rows = load_raw_jsons(raw_dir)
    df = transform_to_table(rows)

    output_path = project_root / "artifacts" / run_id / "dataset.csv"
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
