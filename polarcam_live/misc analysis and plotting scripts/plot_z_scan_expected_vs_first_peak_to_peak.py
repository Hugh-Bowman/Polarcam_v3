from __future__ import annotations

import csv
import json
import argparse
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
ANALYSIS_DIR = (
    ROOT
    / "background characterisation"
    / "z scan of a rod"
    / "pending"
    / "z_scan_interference_analysis"
)


def moving_average_reflect(x: np.ndarray, window: int) -> np.ndarray:
    window = max(3, int(window))
    if window % 2 == 0:
        window += 1
    pad = window // 2
    xp = np.pad(np.asarray(x, dtype=np.float64), pad, mode="reflect")
    return np.convolve(xp, np.ones(window) / window, mode="valid")


def first_peak_to_peak(time_s: np.ndarray, trace: np.ndarray) -> dict[str, float]:
    # Smooth enough to find the first z-envelope extrema, not frame noise.
    smooth = moving_average_reflect(trace, 81)
    dy = np.diff(smooth)
    sign = np.sign(dy)
    sign[sign == 0] = 1
    changes = np.diff(sign)
    maxima = np.where(changes < 0)[0] + 1
    minima = np.where(changes > 0)[0] + 1
    extrema = sorted([(int(i), "max") for i in maxima] + [(int(i), "min") for i in minima])

    # Ignore edge artefacts from smoothing and very small wiggles.
    edge = 100
    extrema = [(i, kind) for i, kind in extrema if edge <= i < len(trace) - edge]
    if len(extrema) < 2:
        raise RuntimeError("Could not find two extrema in the z-scan trace")

    global_range = float(np.max(smooth) - np.min(smooth))
    min_delta = max(10.0, 0.15 * global_range)
    for (i0, k0), (i1, k1) in zip(extrema, extrema[1:]):
        if k0 == k1:
            continue
        delta = abs(float(smooth[i1] - smooth[i0]))
        if delta >= min_delta:
            lo_i, hi_i = (i0, i1) if smooth[i0] <= smooth[i1] else (i1, i0)
            return {
                "idx0": float(i0),
                "idx1": float(i1),
                "time0_s": float(time_s[i0]),
                "time1_s": float(time_s[i1]),
                "kind0": k0,
                "kind1": k1,
                "value0": float(smooth[i0]),
                "value1": float(smooth[i1]),
                "low_idx": float(lo_i),
                "high_idx": float(hi_i),
                "low_value": float(smooth[lo_i]),
                "high_value": float(smooth[hi_i]),
                "peak_to_peak_intensity": delta,
                "half_peak_to_peak_intensity": 0.5 * delta,
            }
    raise RuntimeError("Could not find a sufficiently large first peak-to-peak excursion")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis-dir", type=Path, default=ANALYSIS_DIR)
    args = parser.parse_args()
    analysis_dir = args.analysis_dir

    summary = json.loads((analysis_dir / "z_scan_summary.json").read_text())
    rod_signal = float(summary["estimated_rod_signal_r_squared"])
    traces = np.genfromtxt(analysis_dir / "intensity_traces.csv", delimiter=",", names=True)
    time_s = np.asarray(traces["time_s"], dtype=np.float64)
    rod_trace = np.asarray(traces["rod_11x11_central_mean"], dtype=np.float64)
    smooth = moving_average_reflect(rod_trace, 81)
    p2p = first_peak_to_peak(time_s, rod_trace)
    actual_p2p_percent = 100.0 * p2p["peak_to_peak_intensity"] / rod_signal
    actual_half_percent = 100.0 * p2p["half_peak_to_peak_intensity"] / rod_signal

    expected_rows = list(csv.DictReader((analysis_dir / "expected_interference_amplitudes.csv").open()))
    plot_rows = []
    for row in expected_rows:
        amp = float(row["amplitude_intensity_units"])
        amp_pct = float(row["amplitude_percent_of_rod_signal"])
        plot_rows.append(
            {
                "case": row["case"],
                "half_amplitude_intensity_units": amp,
                "half_amplitude_percent_of_rod_signal": amp_pct,
                "peak_to_peak_intensity_units": 2.0 * amp,
                "peak_to_peak_percent_of_rod_signal": 2.0 * amp_pct,
            }
        )
    plot_rows.append(
        {
            "case": "actual first peak-to-peak in z scan",
            "half_amplitude_intensity_units": p2p["half_peak_to_peak_intensity"],
            "half_amplitude_percent_of_rod_signal": actual_half_percent,
            "peak_to_peak_intensity_units": p2p["peak_to_peak_intensity"],
            "peak_to_peak_percent_of_rod_signal": actual_p2p_percent,
        }
    )

    out_csv = analysis_dir / "expected_vs_actual_first_peak_to_peak_percent.csv"
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(plot_rows[0].keys()))
        writer.writeheader()
        writer.writerows(plot_rows)

    labels = [r["case"] for r in plot_rows]
    values = [float(r["peak_to_peak_percent_of_rod_signal"]) for r in plot_rows]
    colors = ["tab:blue"] * (len(values) - 1) + ["tab:red"]
    fig, ax = plt.subplots(figsize=(9, 5.2))
    ax.barh(np.arange(len(values)), values, color=colors)
    ax.set_yticks(np.arange(len(values)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Peak-to-peak intensity variation (% of rod signal)")
    ax.set_title("Expected interference variation vs first measured z-scan peak-to-peak")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(analysis_dir / "expected_vs_actual_first_peak_to_peak_percent.png", dpi=240)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(time_s, rod_trace, lw=0.6, alpha=0.45, label="rod 11x11 mean")
    ax.plot(time_s, smooth, lw=1.8, label="smoothed trace used for first peak-to-peak")
    i0 = int(p2p["idx0"])
    i1 = int(p2p["idx1"])
    ax.scatter([time_s[i0], time_s[i1]], [smooth[i0], smooth[i1]], color="tab:red", zorder=5, label="first peak-to-peak extrema")
    ax.plot([time_s[i0], time_s[i1]], [smooth[i0], smooth[i1]], color="tab:red", lw=2)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("central 11x11 mean intensity")
    ax.set_title("First peak-to-peak selected from z-scan trace")
    ax.legend()
    fig.tight_layout()
    fig.savefig(analysis_dir / "first_peak_to_peak_selected_on_z_scan.png", dpi=240)
    plt.close(fig)

    with (analysis_dir / "first_peak_to_peak_summary.json").open("w") as f:
        json.dump(
            {
                **p2p,
                "rod_signal_reference": rod_signal,
                "actual_peak_to_peak_percent_of_rod_signal": actual_p2p_percent,
                "actual_half_peak_to_peak_percent_of_rod_signal": actual_half_percent,
                "smoothing_window_frames": 81,
            },
            f,
            indent=2,
        )

    print("Wrote", out_csv)
    print("actual_peak_to_peak_percent", actual_p2p_percent)
    print("actual_peak_to_peak_intensity", p2p["peak_to_peak_intensity"])


if __name__ == "__main__":
    main()
