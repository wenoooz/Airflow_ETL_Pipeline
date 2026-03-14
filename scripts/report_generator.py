import json
import sys
from pathlib import Path

import matplotlib

from _path_utils import safe_run_id
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

def get_project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def load_kpis(kpis_path: Path) -> dict:
    """Load kpis.json."""
    if not kpis_path.is_file():
        raise FileNotFoundError(f"KPIs not found: {kpis_path}")
    with open(kpis_path, encoding="utf-8") as f:
        return json.load(f)


def plot_bler_vs_snr(bler_vs_snr: list[dict], output_path: Path) -> None:
    """Plot BLER vs SNR with one curve per (channel_type, modulation)."""
    df = pd.DataFrame(bler_vs_snr)
    if df.empty:
        plt.figure(figsize=(8, 5))
        plt.title("BLER vs SNR (no data)")
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    for (ch, mod), group in df.groupby(["channel_type", "modulation"]):
        label = f"{ch} / {mod}"
        ax.semilogy(
            group["snr_db"],
            group["mean_bler"].clip(lower=1e-6),
            marker="o",
            markersize=4,
            label=label,
        )
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("BLER")
    ax.set_title("Block Error Rate vs SNR by Channel and Modulation")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    ax.set_ylim(bottom=1e-6)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def write_interpretation(kpis: dict) -> str:
    """Produce a short written interpretation from KPIs and curves."""
    run_id = kpis.get("run_id", "?")
    overall = kpis.get("overall", {})
    bler_vs_snr = kpis.get("bler_vs_snr", [])
    by_cm = kpis.get("by_channel_modulation", [])

    lines = [
        f"<h2>Interpretation (Run: {run_id})</h2>",
        "<p>",
        "BLER generally decreases as SNR increases, as expected: higher signal-to-noise ratio improves reliability. ",
        "Rayleigh fading typically requires higher SNR than AWGN to achieve the same BLER, because fading introduces additional variability. ",
        "16QAM is less robust than QPSK at low SNR, since higher-order modulation carries more bits per symbol and is more sensitive to noise. ",
        "The curves in the plot above summarize these effects across channel types and modulations.",
        "</p>",
    ]

    if overall and overall.get("num_rows"):
        lines.extend([
            "<p>",
            f"Overall, this run aggregated <b>{overall['num_rows']}</b> simulation points, "
            f"with mean BLER <b>{overall.get('mean_bler', 0):.4f}</b>, "
            f"mean BER <b>{overall.get('mean_ber', 0):.6f}</b>, "
            f"and mean effective throughput <b>{overall.get('mean_effective_throughput', 0):.4f}</b> bits/symbol.",
            "</p>",
        ])

    if by_cm:
        lines.append("<p>Per channel/modulation: ")
        parts = []
        for r in by_cm:
            bler = r.get("mean_bler")
            thr = r.get("mean_effective_throughput")
            bler_s = f"{bler:.4f}" if bler is not None else "N/A"
            thr_s = f"{thr:.4f}" if thr is not None else "N/A"
            parts.append(f"{r.get('channel_type', '?')}-{r.get('modulation', '?')} (BLER={bler_s}, throughput={thr_s})")
        lines.append(", ".join(parts) + ".</p>")

    return "\n".join(lines)


def generate_html_report(
    run_id: str,
    kpis: dict,
    plot_filename: str,
    project_root: Path,
) -> str:
    """Build HTML report body with plot reference and interpretation."""
    plot_src = f"plots/{plot_filename}"
    interpretation = write_interpretation(kpis)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Sionna ETL Report — {run_id}</title>
  <style>
    body {{ font-family: sans-serif; margin: 1.5rem; max-width: 900px; }}
    h1 {{ color: #333; }}
    img {{ max-width: 100%; height: auto; }}
  </style>
</head>
<body>
  <h1>Sionna ETL Pipeline Report</h1>
  <p><b>Run ID:</b> {run_id}</p>
  <h2>BLER vs SNR</h2>
  <img src="{plot_src}" alt="BLER vs SNR" />
  {interpretation}
</body>
</html>
"""


def main(run_id: str, project_root: Path | None = None) -> Path:
    """
    Generate report and plot; write artifacts/<run_id>/report.html and plots/bler_vs_snr.png.
    Returns:Path to report.html.
    """
    if project_root is None:
        project_root = get_project_root()

    base = project_root / "artifacts" / safe_run_id(run_id)
    kpis_path = base / "kpis.json"
    kpis = load_kpis(kpis_path)
    bler_vs_snr = kpis.get("bler_vs_snr", [])

    plots_dir = base / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    plot_path = plots_dir / "bler_vs_snr.png"
    plot_bler_vs_snr(bler_vs_snr, plot_path)

    report_path = base / "report.html"
    html = generate_html_report(run_id, kpis, "bler_vs_snr.png", project_root)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html)

    return report_path


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python report_generator.py <run_id> [project_root]")
        sys.exit(2)

    run_id = sys.argv[1]
    root = Path(sys.argv[2]) if len(sys.argv) > 2 else None
    out = main(run_id, root)
    print(out)
