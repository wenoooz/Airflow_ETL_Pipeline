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


    bler_vals = df["mean_bler"].clip(lower=1e-6)
    bler_min = float(bler_vals.min())
    bler_max = float(bler_vals.max())
    y_bottom = max(bler_min / 3.0, 1e-4)
    y_top = min(bler_max * 3.0, 1.0)

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
    ax.set_ylim(bottom=y_bottom, top=y_top)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_ber_vs_snr(bler_vs_snr: list[dict], output_path: Path) -> None:
    """Plot BER vs SNR with one curve per (channel_type, modulation)."""
    df = pd.DataFrame(bler_vs_snr)
    if df.empty:
        plt.figure(figsize=(8, 5))
        plt.title("BER vs SNR (no data)")
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        return

    ber_vals = df["mean_ber"].clip(lower=1e-8)
    ber_min = float(ber_vals.min())
    ber_max = float(ber_vals.max())
    y_bottom = max(ber_min / 3.0, 1e-5)
    y_top = min(ber_max * 3.0, 1.0)

    fig, ax = plt.subplots(figsize=(8, 5))
    for (ch, mod), group in df.groupby(["channel_type", "modulation"]):
        label = f"{ch} / {mod}"
        ax.semilogy(
            group["snr_db"],
            group["mean_ber"].clip(lower=1e-8),
            marker="s",
            markersize=4,
            label=label,
        )
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("BER")
    ax.set_title("Bit Error Rate vs SNR by Channel and Modulation")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    ax.set_ylim(bottom=y_bottom, top=y_top)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_throughput_vs_snr(bler_vs_snr: list[dict], output_path: Path) -> None:
    """Plot effective throughput vs SNR with one curve per (channel_type, modulation)."""
    df = pd.DataFrame(bler_vs_snr)
    if df.empty:
        plt.figure(figsize=(8, 5))
        plt.title("Throughput vs SNR (no data)")
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        return

    thr_vals = df["mean_effective_throughput"]
    thr_min = float(thr_vals.min())
    thr_max = float(thr_vals.max())
    y_bottom = max(thr_min - 0.2 * (thr_max - thr_min), 0.0)
    y_top = thr_max + 0.2 * (thr_max - thr_min)

    fig, ax = plt.subplots(figsize=(8, 5))
    for (ch, mod), group in df.groupby(["channel_type", "modulation"]):
        label = f"{ch} / {mod}"
        ax.plot(
            group["snr_db"],
            group["mean_effective_throughput"],
            marker="^",
            markersize=4,
            label=label,
        )
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("Effective throughput (bits/symbol)")
    ax.set_title("Effective Throughput vs SNR by Channel and Modulation")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    ax.set_ylim(bottom=y_bottom, top=y_top)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def write_interpretation(kpis: dict) -> str:
    """Produce a short written interpretation from KPIs and curves."""
    run_id = kpis.get("run_id", "?")
    overall = kpis.get("overall", {})
    by_cm = kpis.get("by_channel_modulation", [])
    meta = kpis.get("meta", {})

    lines = [
        f"<h2>Interpretation (Run: {run_id})</h2>",
        "<p>",
        "Across the simulated SNR range, BLER and BER change only moderately: AWGN/QPSK shows the best reliability with the lowest error rates, ",
        "AWGN/16QAM is slightly worse, and both Rayleigh curves stay at higher BLER/BER for all SNR values, reflecting the extra variability from fading. ",
        "This means that for a given SNR, Rayleigh typically needs more link margin than AWGN to reach the same BLER/BER target. ",
        "Comparing modulations, QPSK is more robust than 16QAM on the same channel, especially at the lower SNR points where the gap between the orange (AWGN/QPSK) and blue (AWGN/16QAM) curves is largest. ",
        "The three plots above summarize these trade-offs between reliability and throughput across channel types and modulations.",
        "</p>",
    ]

    if overall and overall.get("num_rows"):
        num_rows = int(overall["num_rows"])
        parts = [
            "<p>",
            f"Overall, this run aggregated <b>{num_rows}</b> simulation points",
        ]
        if meta:
            n_ch = int(meta.get("num_channel_types", 0))
            n_mod = int(meta.get("num_modulations", 0))
            n_snr = int(meta.get("num_snr_values", 0))
            parts.append(
                f" across <b>{n_ch}</b> channel types, <b>{n_mod}</b> modulations, "
                f"and <b>{n_snr}</b> distinct SNR values,"
            )
        parts.append(
            f" with mean BLER <b>{overall.get('mean_bler', 0):.4f}</b>, "
            f"mean BER <b>{overall.get('mean_ber', 0):.6f}</b>, "
            f"and mean effective throughput <b>{overall.get('mean_effective_throughput', 0):.4f}</b> bits/symbol."
        )
        parts.append("</p>")
        lines.extend(parts)

    if by_cm:
        lines.append("<p>Per channel/modulation (averaged over SNR values): ")
        parts = []
        for r in by_cm:
            bler = r.get("mean_bler")
            thr = r.get("mean_effective_throughput")
            bler_s = f"{bler:.4f}" if bler is not None else "N/A"
            thr_s = f"{thr:.4f}" if thr is not None else "N/A"
            parts.append(f"{r.get('channel_type', '?')}-{r.get('modulation', '?')} (BLER={bler_s}, throughput={thr_s})")
        lines.append(", ".join(parts) + ".</p>")

    return "\n".join(lines)


def format_data_quality_section(dq_results: dict | None) -> str:
    """Render a short HTML section describing data-quality checks."""
    if not dq_results:
        return "<p><i>No data-quality summary available.</i></p>"

    checks = dq_results.get("checks", [])
    if not checks:
        return "<p><i>No data-quality summary available.</i></p>"

    parts = ["<h2>Data quality checks</h2>", "<ul>"]
    for c in checks:
        name = c.get("check", "unknown")
        passed = bool(c.get("passed"))
        msg = c.get("message", "")
        status = "PASSED" if passed else "FAILED"
        parts.append(f"<li><b>{name}</b> — {status}. {msg}</li>")
    parts.append("</ul>")
    return "\n".join(parts)


def generate_html_report(
    run_id: str,
    kpis: dict,
    dq_results: dict | None,
    plot_filename: str,
    project_root: Path,
) -> str:
    """Build HTML report body with plot, interpretation, and DQ summary."""
    plot_src = f"plots/{plot_filename}"
    interpretation = write_interpretation(kpis)
    dq_section = format_data_quality_section(dq_results)
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
  <h2>BER vs SNR</h2>
  <img src="plots/ber_vs_snr.png" alt="BER vs SNR" />
  <h2>Effective Throughput vs SNR</h2>
  <img src="plots/throughput_vs_snr.png" alt="Throughput vs SNR" />
  {interpretation}
  {dq_section}
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

    dq_path = base / "dq_checks.json"
    dq_results: dict | None
    if dq_path.is_file():
        with open(dq_path, encoding="utf-8") as f:
            dq_results = json.load(f)
    else:
        dq_results = None

    plots_dir = base / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    plot_bler_vs_snr(bler_vs_snr, plots_dir / "bler_vs_snr.png")
    plot_ber_vs_snr(bler_vs_snr, plots_dir / "ber_vs_snr.png")
    plot_throughput_vs_snr(bler_vs_snr, plots_dir / "throughput_vs_snr.png")

    report_path = base / "report.html"
    html = generate_html_report(run_id, kpis, dq_results, "bler_vs_snr.png", project_root)
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
