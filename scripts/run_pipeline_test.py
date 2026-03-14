from __future__ import print_function

import json
import sys
from datetime import datetime
from pathlib import Path

# If run.
print("run_pipeline_test.py loaded.", flush=True)

# Give sth immediately
def log(msg: str) -> None:
    print(msg, flush=True)

# Run from project root; add scripts dir so we can import the other modules
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))


def main() -> None:
    log("Starting pipeline test (no Sionna)...")

    run_id = "pipeline_test_run"
    root = PROJECT_ROOT
    log(f"Project root: {root}")

    # 1. Generate run plan
    log("Step 1: Generating run plan...")
    from run_plan_generator import main as gen_main
    gen_main(run_id)
    log("Run plan generated.")

    # 2. Load run plan and create mock raw JSONs
    run_plan_path = root / "artifacts" / run_id / "run_plan.json"
    with open(run_plan_path, encoding="utf-8") as f:
        data = json.load(f)
    run_plan = data["run_plan"]
    raw_dir = root / "artifacts" / run_id / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    # Use run_plan timestamp so mock data matches the run time; fallback to now()
    run_ts = data.get("timestamp") or datetime.now().isoformat()

    for row in run_plan:
        # Mock metrics: BLER/BER decrease with SNR, Rayleigh worse than AWGN, 16QAM worse than QPSK
        snr = row["snr_db"]
        ch = row["channel_type"]
        mod = row["modulation"]
        bler = max(1e-6, 0.5 - snr * 0.02 + (0.1 if ch == "Rayleigh" else 0) + (0.05 if mod == "16QAM" else 0))
        ber = bler * 0.1
        thr = (1 - bler) * (4 if mod == "16QAM" else 2)
        out = {
            "timestamp": run_ts,
            "run_id": run_id,
            "run_index": row["run_index"],
            "seed": row["seed"],
            "snr_db": snr,
            "channel_type": ch,
            "modulation": mod,
            "bler": min(1.0, bler),
            "ber": min(1.0, ber),
            "effective_throughput": max(0, thr),
            "num_frames": row["num_frames_per_run"],
            "total_bits": 100000,
            "total_bit_errors": int(ber * 100000),
            "total_blocks": 1000,
            "total_block_errors": int(bler * 1000),
        }
        with open(raw_dir / f"{row['run_index']}.json", "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
    log("Mock raw JSONs written.")

    # 3. Transform
    log("Step 2: Transform raw -> dataset.csv...")
    from transform_raw_to_table import main as transform_main
    transform_main(run_id, root)
    log("Dataset CSV written.")

    # 4. Data quality checks
    log("Step 3: Data quality checks...")
    from data_quality_checks import main as dq_main
    dq_main(run_id, root)
    log("Data quality checks passed.")

    # 5. KPIs
    log("Step 4: Compute KPIs...")
    from compute_kpis import main as kpis_main
    kpis_main(run_id, root)
    log("KPIs written.")

    # 6. Report
    log("Step 5: Generate report...")
    from report_generator import main as report_main
    report_main(run_id, root)
    log("Report generated.")

    log("All steps OK. Check artifacts/" + run_id + "/")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}", flush=True)
        import traceback
        traceback.print_exc()
        sys.exit(1)
