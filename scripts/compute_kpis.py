import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

def get_project_root() -> Path:
    """Resolve project root (parent of scripts/)."""
    return Path(__file__).resolve().parent.parent


def load_dataset(csv_path: Path) -> pd.DataFrame:
    """Load consolidated dataset from CSV."""
    if not csv_path.is_file():
        raise FileNotFoundError(f"Dataset not found: {csv_path}")
    return pd.read_csv(csv_path, encoding="utf-8")


def compute_overall_kpis(df: pd.DataFrame) -> dict:
    """Compute overall average KPIs across all simulations."""
    if df.empty:
        return {
            "mean_ber": None,
            "mean_bler": None,
            "mean_effective_throughput": None,
            "num_rows": 0,
        }

    return {
        "mean_ber": float(df["ber"].mean()),
        "mean_bler": float(df["bler"].mean()),
        "mean_effective_throughput": float(df["effective_throughput"].mean()),
        "num_rows": int(len(df)),
    }


def compute_by_channel_modulation(df: pd.DataFrame) -> list[dict]:
    """Compute average KPIs grouped by channel_type, modulation."""
    if df.empty:
        return []

    grouped = (
        df.groupby(["channel_type", "modulation"], dropna=False)[
            ["ber", "bler", "effective_throughput"]
        ]
        .mean()
        .reset_index()
    )

    records: list[dict] = []
    for _, row in grouped.iterrows():
        records.append(
            {
                "channel_type": row["channel_type"],
                "modulation": row["modulation"],
                "mean_ber": float(row["ber"]),
                "mean_bler": float(row["bler"]),
                "mean_effective_throughput": float(row["effective_throughput"]),
            }
        )
    return records


def compute_bler_vs_snr(df: pd.DataFrame) -> list[dict]:
    if df.empty:
        return []

    grouped = (
        df.groupby(["channel_type", "modulation", "snr_db"], dropna=False)[
            ["ber", "bler", "effective_throughput"]
        ]
        .mean()
        .reset_index()
        .sort_values(["channel_type", "modulation", "snr_db"])
    )

    records: list[dict] = []
    for _, row in grouped.iterrows():
        records.append(
            {
                "channel_type": row["channel_type"],
                "modulation": row["modulation"],
                "snr_db": float(row["snr_db"]),
                "mean_ber": float(row["ber"]),
                "mean_bler": float(row["bler"]),
                "mean_effective_throughput": float(row["effective_throughput"]),
            }
        )
    return records


def run_kpi_computation(run_id: str, project_root: Path | None = None) -> dict:
    if project_root is None:
        project_root = get_project_root()

    csv_path = project_root / "artifacts" / run_id / "dataset.csv"
    df = load_dataset(csv_path)

    overall = compute_overall_kpis(df)
    by_channel_mod = compute_by_channel_modulation(df)
    bler_vs_snr = compute_bler_vs_snr(df)

    return {
        "run_id": run_id,
        "generated_at": datetime.now().isoformat(),
        "overall": overall,
        "by_channel_modulation": by_channel_mod,
        "bler_vs_snr": bler_vs_snr,
    }


def main(run_id: str, project_root: Path | None = None) -> Path:
    result = run_kpi_computation(run_id, project_root)

    if project_root is None:
        project_root = get_project_root()

    output_path = project_root / "artifacts" / run_id / "kpis.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    return output_path


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python compute_kpis.py <run_id> [project_root]")
        sys.exit(2)

    run_id_arg = sys.argv[1]
    root_arg = Path(sys.argv[2]) if len(sys.argv) > 2 else None

    out_path = main(run_id_arg, root_arg)
    print(out_path)

