import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from _path_utils import safe_run_id

def _to_native(obj):
    if isinstance(obj, dict):
        return {k: _to_native(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_native(v) for v in obj]
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

REQUIRED_COLUMNS = [
    "snr_db",
    "signal_to_noise_ratio_db",
    "channel_type",
    "modulation",
    "ber",
    "bler",
    "effective_throughput",
]

def get_project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def load_dataset(csv_path: Path) -> pd.DataFrame:
    if not csv_path.is_file():
        raise FileNotFoundError(f"Dataset not found: {csv_path}")
    return pd.read_csv(csv_path, encoding="utf-8")


def load_run_plan(run_plan_path: Path) -> dict:
    if not run_plan_path.is_file():
        raise FileNotFoundError(f"Run plan not found: {run_plan_path}")
    with open(run_plan_path, encoding="utf-8") as f:
        return json.load(f)


def check_no_missing(df: pd.DataFrame) -> dict:

    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        return {
            "check": "no_missing_values",
            "passed": False,
            "required_columns": REQUIRED_COLUMNS,
            "columns_with_missing": missing_cols,
            "message": f"Missing columns in dataset: {missing_cols}",
        }
    missing = df[REQUIRED_COLUMNS].isna().any()
    failed_cols = [c for c in REQUIRED_COLUMNS if missing[c]]
    passed = len(failed_cols) == 0
    return {
        "check": "no_missing_values",
        "passed": passed,
        "required_columns": REQUIRED_COLUMNS,
        "columns_with_missing": failed_cols,
        "message": "No missing values in required columns" if passed else f"Missing values in: {failed_cols}",
    }


def check_row_count(df: pd.DataFrame, expected_size: int) -> dict:
    actual = len(df)
    passed = actual == expected_size
    return {
        "check": "row_count_matches_plan",
        "passed": passed,
        "expected_rows": expected_size,
        "actual_rows": actual,
        "message": f"Row count matches plan ({actual})" if passed else f"Expected {expected_size} rows, got {actual}",
    }


def check_metric_ranges(df: pd.DataFrame) -> dict:
    for col in ["ber", "bler", "effective_throughput"]:
        if col not in df.columns:
            return {
                "check": "metric_ranges_valid",
                "passed": False,
                "invalid_ber_count": 0,
                "invalid_bler_count": 0,
                "invalid_throughput_count": 0,
                "message": f"Missing column for metric check: {col}",
            }
    ber_ok = (df["ber"] >= 0) & (df["ber"] <= 1)
    bler_ok = (df["bler"] >= 0) & (df["bler"] <= 1)
    thr_ok = df["effective_throughput"] >= 0
    invalid_ber = (~ber_ok).sum()
    invalid_bler = (~bler_ok).sum()
    invalid_thr = (~thr_ok).sum()
    passed = invalid_ber == 0 and invalid_bler == 0 and invalid_thr == 0
    return {
        "check": "metric_ranges_valid",
        "passed": passed,
        "invalid_ber_count": int(invalid_ber),
        "invalid_bler_count": int(invalid_bler),
        "invalid_throughput_count": int(invalid_thr),
        "message": "All metrics in valid range" if passed else f"Invalid: ber={invalid_ber}, bler={invalid_bler}, throughput={invalid_thr}",
    }


def check_snr_matches_plan(df: pd.DataFrame, expected_snr: list) -> dict:
    if "snr_db" not in df.columns:
        return {
            "check": "snr_values_match_plan",
            "passed": False,
            "expected_snr": sorted(int(x) for x in expected_snr),
            "actual_snr": [],
            "message": "Missing column: snr_db",
        }
    actual_snr = set(df["snr_db"].dropna().astype(int).tolist())
    expected_set = set(int(x) for x in expected_snr)
    passed = actual_snr == expected_set
    return {
        "check": "snr_values_match_plan",
        "passed": passed,
        "expected_snr": sorted(expected_set),
        "actual_snr": sorted(actual_snr),
        "message": "SNR values match plan" if passed else f"SNR mismatch: expected {expected_set}, got {actual_snr}",
    }


def check_parameter_combinations(df: pd.DataFrame, run_plan_data: dict) -> dict:
    plan = run_plan_data.get("run_plan", [])
    expected_combos = set()
    for row in plan:
        combo = (row["channel_type"], row["modulation"], float(row["snr_db"]))
        expected_combos.add(combo)

    actual_combos = set()
    for _, row in df.iterrows():
        combo = (row["channel_type"], row["modulation"], float(row["snr_db"]))
        actual_combos.add(combo)

    missing = expected_combos - actual_combos
    extra = actual_combos - expected_combos
    passed = len(missing) == 0 and len(extra) == 0

    return {
        "check": "parameter_combinations_match_plan",
        "passed": passed,
        "missing_combos": list(missing),
        "extra_combos": list(extra),
        "message": "All parameter combinations match plan" if passed else f"Combo mismatch: missing {len(missing)}, extra {len(extra)}",
    }


def run_checks(run_id: str, project_root: Path | None = None) -> dict:
   
    if project_root is None:
        project_root = get_project_root()

    base = project_root / "artifacts" / safe_run_id(run_id)
    csv_path = base / "dataset.csv"
    run_plan_path = base / "run_plan.json"

    df = load_dataset(csv_path)
    run_plan_data = load_run_plan(run_plan_path)
    expected_size = run_plan_data.get("plan_size", len(run_plan_data.get("run_plan", [])))
    config = run_plan_data.get("config", {})
    expected_snr = config.get("snr_db", [])
    if not expected_snr and run_plan_data.get("run_plan"):
        expected_snr = sorted(set(row["snr_db"] for row in run_plan_data["run_plan"]))

    checks = [
        check_no_missing(df),
        check_row_count(df, expected_size),
        check_metric_ranges(df),
        check_snr_matches_plan(df, expected_snr),
        check_parameter_combinations(df, run_plan_data),
    ]

    all_passed = all(c["passed"] for c in checks)
    return {
        "run_id": run_id,
        "all_passed": all_passed,
        "checks": checks,
    }


def main(run_id: str, project_root: Path | None = None) -> Path:
   
    result = run_checks(run_id, project_root)

    output_path = (project_root or get_project_root()) / "artifacts" / safe_run_id(run_id) / "dq_checks.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out = _to_native(result)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    if not result["all_passed"]:
        print("Data quality checks failed:", json.dumps(out, indent=2, ensure_ascii=False))
        sys.exit(1)
    return output_path


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python data_quality_checks.py <run_id> [project_root]")
        sys.exit(2)

    run_id = sys.argv[1]
    project_root = Path(sys.argv[2]) if len(sys.argv) > 2 else None

    main(run_id, project_root)
