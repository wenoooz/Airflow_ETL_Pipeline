"""
Data Quality Checks for Sionna ETL Pipeline

Runs at least two data-quality checks on the consolidated dataset:
1. No missing values in required columns
2. Number of rows equals expected plan size
3. Metric ranges valid (BER/BLER in [0,1], effective_throughput >= 0)
4. SNR values match plan

Writes results to artifacts/<run_id>/dq_checks.json and exits with non-zero if any check fails.
"""

import json
import sys
from pathlib import Path

import pandas as pd

REQUIRED_COLUMNS = [
    "snr_db",
    "channel_type",
    "modulation",
    "ber",
    "bler",
    "effective_throughput",
]


def get_project_root() -> Path:
    """Resolve project root (parent of scripts/)."""
    return Path(__file__).resolve().parent.parent


def load_dataset(csv_path: Path) -> pd.DataFrame:
    """Load consolidated dataset from CSV."""
    if not csv_path.is_file():
        raise FileNotFoundError(f"Dataset not found: {csv_path}")
    return pd.read_csv(csv_path, encoding="utf-8")


def load_run_plan(run_plan_path: Path) -> dict:
    """Load run plan JSON."""
    if not run_plan_path.is_file():
        raise FileNotFoundError(f"Run plan not found: {run_plan_path}")
    with open(run_plan_path, encoding="utf-8") as f:
        return json.load(f)


def check_no_missing(df: pd.DataFrame) -> dict:
    """Check that required columns have no missing values."""
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
    """Check that number of rows equals expected plan size."""
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
    """Check that BER/BLER are in [0,1] and effective_throughput >= 0."""
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
    """Check that SNR values in dataset match the planned SNR set."""
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


def run_checks(run_id: str, project_root: Path | None = None) -> dict:
    """
    Run all data quality checks and return results.

    Args:
        run_id: Run identifier.
        project_root: Project root path; if None, auto-detect.

    Returns:
        Dict with keys: run_id, all_passed, checks (list of check results).
    """
    if project_root is None:
        project_root = get_project_root()

    base = project_root / "artifacts" / run_id
    csv_path = base / "dataset.csv"
    run_plan_path = base / "run_plan.json"

    df = load_dataset(csv_path)
    run_plan_data = load_run_plan(run_plan_path)
    expected_size = run_plan_data.get("plan_size", len(run_plan_data.get("run_plan", [])))
    config = run_plan_data.get("config", {})
    expected_snr = config.get("snr_db", [])

    checks = [
        check_no_missing(df),
        check_row_count(df, expected_size),
        check_metric_ranges(df),
        check_snr_matches_plan(df, expected_snr),
    ]

    all_passed = all(c["passed"] for c in checks)
    return {
        "run_id": run_id,
        "all_passed": all_passed,
        "checks": checks,
    }


def main(run_id: str, project_root: Path | None = None) -> Path:
    """
    Run checks, write dq_checks.json, exit 0 if all passed else 1.

    Returns:
        Path to written dq_checks.json.
    """
    result = run_checks(run_id, project_root)

    output_path = (project_root or get_project_root()) / "artifacts" / run_id / "dq_checks.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    if not result["all_passed"]:
        sys.exit(1)
    return output_path


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python data_quality_checks.py <run_id> [project_root]")
        sys.exit(2)

    run_id = sys.argv[1]
    project_root = Path(sys.argv[2]) if len(sys.argv) > 2 else None

    main(run_id, project_root)
