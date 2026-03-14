import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

from _path_utils import safe_run_id
import tensorflow as tf

# Sionna imports
try:
    import sionna
    from sionna.channel import AWGN, RayleighBlockFading
    from sionna.mapping import Mapper, Demapper
    from sionna.utils import BinarySource, compute_ber, compute_bler
except ImportError as e:
    print(f"Error importing Sionna: {e}")
    print("Please install Sionna: pip install sionna")
    sys.exit(1)

def get_project_root() -> Path:
    """Resolve project root (parent of scripts/)."""
    return Path(__file__).resolve().parent.parent

def create_channel(channel_type: str, **kwargs):
    """Create channel model (AWGN or Rayleigh)."""
    if channel_type.upper() == "AWGN":
        return AWGN()
    elif channel_type.upper() == "RAYLEIGH":
        # Current Sionna RayleighBlockFading expects num_rx, num_tx, num_rx_ant, num_tx_ant
        return RayleighBlockFading(num_rx=1, num_tx=1, num_rx_ant=1, num_tx_ant=1)
    else:
        raise ValueError(f"Unknown channel type: {channel_type}")

def create_modulation(modulation: str):
    """Create mapper and demapper for modulation scheme."""
    if modulation.upper() == "QPSK":
        num_bits_per_symbol = 2
    elif modulation.upper() == "16QAM":
        num_bits_per_symbol = 4
    else:
        raise ValueError(f"Unknown modulation: {modulation}")
    mapper = Mapper(constellation_type="qam", num_bits_per_symbol=num_bits_per_symbol)
    demapper = Demapper("app", constellation_type="qam", num_bits_per_symbol=num_bits_per_symbol)
    return mapper, demapper, num_bits_per_symbol

def simulate_siso_link(
    channel_type: str,
    modulation: str,
    snr_db: float,
    num_frames: int,
    batch_size: int,
    seed: int,
) -> dict:   
    # Set random seeds for reproducibility
    import random
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    # For full reproducibility on GPU, though might slow down simulation
    # os.environ['TF_DETERMINISTIC_OPS'] = '1'

    # Create components
    mapper, demapper, num_bits_per_symbol = create_modulation(modulation)
    channel = create_channel(channel_type)

    # Binary source
    binary_source = BinarySource(seed=seed)

    # Accumulators for errors
    total_bit_errors = 0
    total_block_errors = 0
    total_bits = 0
    total_blocks = 0

    # Simulate in batches
    num_batches = (num_frames + batch_size - 1) // batch_size
    num_symbols_per_frame = 100  # Fixed number of symbols per frame

    for batch_idx in range(num_batches):
        current_batch_size = min(batch_size, num_frames - batch_idx * batch_size)
        if current_batch_size <= 0:
            break

        # Generate random bits: [batch_size, num_symbols, num_bits_per_symbol]
        bits_shape = [current_batch_size, num_symbols_per_frame, num_bits_per_symbol]
        bits = binary_source(bits_shape)

        # Map bits to symbols
        symbols = mapper(bits)

        # Channel processing
        if channel_type.upper() == "AWGN":
            # AWGN channel: normalize power and add noise
            symbol_power = tf.reduce_mean(tf.abs(symbols) ** 2)
            norm = tf.cast(tf.sqrt(symbol_power + 1e-10), symbols.dtype)
            symbols_normalized = symbols / norm
            out = channel([symbols_normalized, tf.cast(snr_db, tf.float32)])
            received_symbols = out[0] if isinstance(out, (list, tuple)) else out
        else:
            # Rayleigh: use 2D throughout so AWGN gets same shape as in AWGN branch (avoids shape mismatches)
            symbol_power = tf.reduce_mean(tf.abs(symbols) ** 2)
            norm = tf.cast(tf.sqrt(symbol_power + 1e-10), symbols.dtype)
            symbols_normalized = symbols / norm
            # Get channel gains: RayleighBlockFading(batch_size, num_time_steps) -> tuple, first element is h
            out = channel(current_batch_size, num_symbols_per_frame)
            h = out[0] if isinstance(out, (list, tuple)) else out
            # Force h to 2D [batch, num_symbols]; mapper output may be [batch, num_symbols, 1]
            h_flat = tf.reshape(h, [-1])
            h_2d = tf.reshape(h_flat, [current_batch_size, num_symbols_per_frame])
            h_2d = tf.cast(h_2d, symbols_normalized.dtype)
            symbols_2d = tf.squeeze(symbols_normalized, axis=-1)  # [batch, num_symbols]
            faded_2d = h_2d * symbols_2d
            faded_symbols = tf.expand_dims(faded_2d, axis=-1)  # [batch, num_symbols, 1] for AWGN
            awgn = AWGN()
            snr_tensor_ray = tf.cast(snr_db, tf.float32)
            out_awgn = awgn([faded_symbols, snr_tensor_ray])
            received_symbols = out_awgn[0] if isinstance(out_awgn, (list, tuple)) else out_awgn
            if len(received_symbols.shape) == 3 and received_symbols.shape[-1] == 1:
                received_symbols = tf.squeeze(received_symbols, axis=-1)

        # Demap symbols to LLRs (demapper expects [batch, num_symbols] and SNR as float32)
        snr_tensor = tf.cast(snr_db, tf.float32)
        llrs = demapper([received_symbols, snr_tensor])

        # Hard decision: convert LLRs to bits
        bits_hat = tf.cast(llrs > 0, tf.float32)

        # Compute errors using Sionna utilities
        # Reshape for error computation: flatten last two dimensions
        bits_flat = tf.reshape(bits, [-1])
        bits_hat_flat = tf.reshape(bits_hat, [-1])
        
        bit_errors = tf.reduce_sum(tf.cast(bits_flat != bits_hat_flat, tf.int64))
        
        # Block errors: at least one error per block (symbol)
        bits_per_block = tf.reshape(bits, [current_batch_size * num_symbols_per_frame, num_bits_per_symbol])
        bits_hat_per_block = tf.reshape(bits_hat, [current_batch_size * num_symbols_per_frame, num_bits_per_symbol])
        block_errors = tf.reduce_sum(
            tf.cast(tf.reduce_any(bits_per_block != bits_hat_per_block, axis=1), tf.int64)
        )

        total_bit_errors += int(bit_errors.numpy())
        total_block_errors += int(block_errors.numpy())
        total_bits += int(tf.size(bits).numpy())
        total_blocks += current_batch_size * num_symbols_per_frame

    # Compute metrics
    ber = float(total_bit_errors) / total_bits if total_bits > 0 else 0.0
    bler = float(total_block_errors) / total_blocks if total_blocks > 0 else 0.0

    # Effective throughput: (1 - BLER) * bits_per_symbol
    # Assumes normalized symbol rate = 1
    effective_throughput = (1.0 - bler) * num_bits_per_symbol

    return {
        "ber": ber,
        "bler": bler,
        "effective_throughput": effective_throughput,
        "total_bits": int(total_bits),
        "total_bit_errors": int(total_bit_errors),
        "total_blocks": int(total_blocks),
        "total_block_errors": int(total_block_errors),
    }


def main(run_id: str, run_index: int, run_plan_path: Path | None = None) -> None:
  
    project_root = get_project_root()

    if run_plan_path is None:
        run_plan_path = project_root / "artifacts" / safe_run_id(run_id) / "run_plan.json"

    # Load run plan
    with open(run_plan_path, encoding="utf-8") as f:
        run_plan_data = json.load(f)

    run_plan = run_plan_data["run_plan"]
    if run_index >= len(run_plan):
        raise ValueError(f"run_index {run_index} >= plan size {len(run_plan)}")

    plan_row = run_plan[run_index]

    # Extract parameters
    channel_type = plan_row["channel_type"]
    modulation = plan_row["modulation"]
    snr_db = plan_row["snr_db"]
    seed = plan_row["seed"]
    num_frames = plan_row["num_frames_per_run"]

    # Run simulation
    batch_size = 32  # Process frames in batches
    metrics = simulate_siso_link(
        channel_type=channel_type,
        modulation=modulation,
        snr_db=snr_db,
        num_frames=num_frames,
        batch_size=batch_size,
        seed=seed,
    )

    # Prepare output
    output = {
        "timestamp": datetime.now().isoformat(),
        "run_id": run_id,
        "run_index": run_index,
        "seed": seed,
        "snr_db": snr_db,
        "channel_type": channel_type,
        "modulation": modulation,
        "bler": metrics["bler"],
        "ber": metrics["ber"],
        "effective_throughput": metrics["effective_throughput"],
        "num_frames": num_frames,
        "total_bits": metrics["total_bits"],
        "total_bit_errors": metrics["total_bit_errors"],
        "total_blocks": metrics["total_blocks"],
        "total_block_errors": metrics["total_block_errors"],
    }

    # Write output JSON
    output_dir = project_root / "artifacts" / safe_run_id(run_id) / "raw"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{run_index}.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"Simulation completed: {output_path}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python sionna_simulator.py <run_id> <run_index> [run_plan_path]")
        sys.exit(1)

    run_id = sys.argv[1]
    run_index = int(sys.argv[2])
    run_plan_path = Path(sys.argv[3]) if len(sys.argv) > 3 else None

    main(run_id, run_index, run_plan_path)
