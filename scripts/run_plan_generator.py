"""
Run Plan Generator for Sionna ETL Pipeline

Reads param_grid.yaml, generates the Cartesian product of all parameter combinations,
assigns deterministic seeds per row, and writes run_plan.json to artifacts/<run_id>/.
"""

import hashlib
import itertools
import json
import sys
from datetime import datetime
from pathlib import Path

import yaml

from _path_utils import safe_run_id


def get_project_root() -> Path:
    """Resolve project root (parent of scripts/)."""
    return Path(__file__).resolve().parent.parent


def load_config(config_path: Path) -> dict:
    """Load parameter grid from YAML."""
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)

def load_existing_run_plan(project_root: Path, source_run_id: str) -> dict:
    source_path = project_root / "artifacts" / safe_run_id(source_run_id) / "run_plan.json"
    if not source_path.is_file():
        raise FileNotFoundError(
            f"Reproduce source run_plan not found: {source_path}. "
            f"Please make sure reproduce_run_id points to an existing run."
        )
    with open(source_path, encoding="utf-8") as f:
        return json.load(f)


def generate_run_plan(config: dict, run_id: str, seed_run_id: str | None = None) -> list[dict]:
    channel_types = config["channel_types"]
    channel_types_ordered = (
        ["Rayleigh", "AWGN"]
        if "Rayleigh" in channel_types and "AWGN" in channel_types
        else channel_types
    )
    modulations = config["modulations"]
    snr_db = config["snr_db"]
    repeats = config["repeats"]
    num_frames = config["num_frames_per_run"]
    base_seed = config.get("base_seed", 42)

    # Cartesian product: (channel, modulation, snr, repeat_idx)
    repeat_indices = list(range(repeats))
    combinations = list(
        itertools.product(channel_types_ordered, modulations, snr_db, repeat_indices)
    )

    # Ensure at least 120 simulation points as per requirements
    if len(combinations) < 120:
        raise ValueError(f"Current configuration only produces {len(combinations)} simulation points. "
                         f"Requirements specify at least 120 (e.g., 2 channels * 2 modulations * 10 SNRs * 3 repeats). "
                         f"Please update config/param_grid.yaml.")

    run_plan = []
    hash_id = safe_run_id((seed_run_id or run_id).strip() or run_id)
    run_id_hash = int(hashlib.md5(hash_id.encode()).hexdigest()[:8], 16)
    for idx, (channel_type, modulation, snr, repeat_idx) in enumerate(combinations):
        # Deterministic seed: base_seed + hash(run_id) + row_index (reproducible)
        seed = base_seed + run_id_hash + idx
        run_plan.append({
            "run_index": idx,
            "channel_type": channel_type,
            "modulation": modulation,
            "snr_db": snr,
            "repeat_index": repeat_idx,
            "seed": seed,
            "num_frames_per_run": num_frames,
        })

    return run_plan


def main(run_id: str | None = None, seed_run_id: str | None = None) -> str:

    project_root = get_project_root()
    config_path = project_root / "config" / "param_grid.yaml"

    if run_id is None:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    if seed_run_id:
        source_plan = load_existing_run_plan(project_root, seed_run_id)
        config = source_plan.get("config", {})
        run_plan = source_plan.get("run_plan", [])
        if not run_plan:
            raise ValueError(
                f"Source run_plan is empty for reproduce_run_id={seed_run_id}"
            )
    else:
        config = load_config(config_path)
        run_plan = generate_run_plan(config, run_id, seed_run_id=seed_run_id)

    output_dir = project_root / "artifacts" / safe_run_id(run_id)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "run_plan.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({
            "run_id": run_id,
            "timestamp": datetime.now().isoformat(),
            "seed_source_run_id": seed_run_id or run_id,
            "config": config,
            "plan_size": len(run_plan),
            "run_plan": run_plan,
        }, f, indent=2, ensure_ascii=False)

    return run_id


if __name__ == "__main__":
    run_id = sys.argv[1] if len(sys.argv) > 1 else None
    seed_run_id = sys.argv[2].strip() if len(sys.argv) > 2 and sys.argv[2].strip() else None
    result = main(run_id, seed_run_id=seed_run_id)
    print(result)
