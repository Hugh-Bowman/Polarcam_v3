from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
import numpy as np
from scipy.signal import find_peaks

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_DIR = (
    Path(__file__).resolve().parent
    / "background characterisation"
    / "z scan of a rod"
    / "pending"
    / "z_scan_interference_analysis_20260720-140342"
)


def moving_average_reflect(x: np.ndarray, window: int) -> np.ndarray:
    if window % 2 == 0:
        window += 1
    pad = window // 2
    xp = np.pad(np.asarray(x, dtype=np.float64), pad, mode="reflect")
    return np.convolve(xp, np.ones(window) / window, mode="valid")


def line_from_points(t: np.ndarray, y: np.ndarray, idxs: list[int]) -> tuple[float, float]:
    return tuple(np.polyfit(t[idxs], y[idxs], 1))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis-dir", type=Path, default=DEFAULT_DIR)
    parser.add_argument("--smooth-window", type=int, default=81)
    parser.add_argument("--peak-distance", type=int, default=400)
    parser.add_argument("--peak-prominence", type=float, default=20.0)
    parser.add_argument("--max-time-s", type=float, default=None)
    parser.add_argument("--upper-times-s", type=float, nargs=2, default=None)
    parser.add_argument("--lower-times-s", type=float, nargs=2, default=None)
    args = parser.parse_args()
    analysis_dir = args.analysis_dir

    summary = json.loads((analysis_dir / "z_scan_summary.json").read_text())
    rod_signal = float(summary["estimated_rod_signal_r_squared"])
    traces = np.genfromtxt(analysis_dir / "intensity_traces.csv", delimiter=",", names=True)
    t = np.asarray(traces["time_s"], dtype=np.float64)
    y = np.asarray(traces["rod_11x11_central_mean"], dtype=np.float64)
    if args.max_time_s is not None:
        keep = t <= float(args.max_time_s)
        if int(np.sum(keep)) < max(20, int(args.smooth_window)):
            raise RuntimeError("Too few frames remain after applying --max-time-s")
        t = t[keep]
        y = y[keep]
    ys = moving_average_reflect(y, int(args.smooth_window))

    peaks, _ = find_peaks(ys, distance=int(args.peak_distance), prominence=float(args.peak_prominence))
    troughs, _ = find_peaks(-ys, distance=int(args.peak_distance), prominence=float(args.peak_prominence))
    edge = 10
    early_min_idx = int(edge + np.argmin(ys[edge : max(edge + 1, len(ys) // 4)]))
    if args.upper_times_s is not None:
        top_two = sorted([int(np.argmin(np.abs(t - tt))) for tt in args.upper_times_s])
    else:
        top_two = [int(i) for i in peaks[:2]]
        if len(top_two) < 2:
            raise RuntimeError("Could not find two upper-envelope peaks")
        top_two = sorted([int(i) for i in top_two])
    trough_list = [int(i) for i in troughs]
    if args.lower_times_s is not None:
        lower_two = sorted([int(np.argmin(np.abs(t - tt))) for tt in args.lower_times_s])
    elif len(trough_list) >= 2:
        lower_two = sorted(trough_list[:2])
    else:
        inner_troughs = [int(i) for i in troughs if i > top_two[0] and i < top_two[1]]
        if not inner_troughs:
            raise RuntimeError("Could not find central lower-envelope trough")
        central_min_idx = int(min(inner_troughs, key=lambda i: ys[i]))
        lower_two = [early_min_idx, central_min_idx]

    upper_anchor = y if args.upper_times_s is not None else ys
    lower_anchor = y if args.lower_times_s is not None else ys
    upper_m, upper_c = line_from_points(t, upper_anchor, top_two)
    lower_m, lower_c = line_from_points(t, lower_anchor, lower_two)
    upper = upper_m * t + upper_c
    lower = lower_m * t + lower_c
    center = 0.5 * (upper + lower)
    separation = upper - lower
    half_amp = 0.5 * separation

    lo = max(min(top_two), min(lower_two))
    hi = min(max(top_two), max(lower_two))
    overlap = (np.arange(len(t)) >= lo) & (np.arange(len(t)) <= hi)
    # If the fitted envelope overlap is short, use all frames between first lower point and second upper point.
    if int(np.sum(overlap)) < 50:
        overlap = (np.arange(len(t)) >= min(lower_two + top_two)) & (np.arange(len(t)) <= max(lower_two + top_two))

    half_amp_mean = float(np.mean(half_amp[overlap]))
    slope_corrected_peak_to_min = float(2.0 * half_amp_mean)
    half_amp_percent = float(100.0 * half_amp_mean / rod_signal)
    slope_corrected_peak_to_min_percent = float(100.0 * slope_corrected_peak_to_min / rod_signal)
    residual = ys - center
    residual_amp = float(0.5 * (np.percentile(residual[overlap], 95) - np.percentile(residual[overlap], 5)))

    # Sign-aware check using the first upper maximum followed by the central lower minimum.
    peak_idx = int(top_two[0])
    min_idx = int(lower_two[1])
    observed_peak_to_min = float(ys[peak_idx] - ys[min_idx])
    mean_slope = float(0.5 * (upper_m + lower_m))
    expected_trend_change = float(mean_slope * (t[min_idx] - t[peak_idx]))
    slope_corrected_from_points = float(observed_peak_to_min + expected_trend_change)
    half_amp_from_points = 0.5 * slope_corrected_from_points

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
                "slope_corrected_peak_to_min_intensity_units": 2.0 * amp,
                "slope_corrected_peak_to_min_percent_of_rod_signal": 2.0 * amp_pct,
            }
        )
    plot_rows.append(
        {
            "case": "actual slope-corrected peak-to-minimum oscillation",
            "half_amplitude_intensity_units": half_amp_mean,
            "half_amplitude_percent_of_rod_signal": half_amp_percent,
            "slope_corrected_peak_to_min_intensity_units": slope_corrected_peak_to_min,
            "slope_corrected_peak_to_min_percent_of_rod_signal": slope_corrected_peak_to_min_percent,
        }
    )

    with (analysis_dir / "expected_vs_actual_slope_corrected_peak_to_min_percent.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(plot_rows[0].keys()))
        writer.writeheader()
        writer.writerows(plot_rows)

    env_summary = {
        "smoothing_window_frames": int(args.smooth_window),
        "peak_distance_frames": int(args.peak_distance),
        "peak_prominence": float(args.peak_prominence),
        "max_time_s": None if args.max_time_s is None else float(args.max_time_s),
        "manual_upper_times_s": None if args.upper_times_s is None else [float(v) for v in args.upper_times_s],
        "manual_lower_times_s": None if args.lower_times_s is None else [float(v) for v in args.lower_times_s],
        "upper_peak_indices": top_two,
        "upper_peak_times_s": [float(t[i]) for i in top_two],
        "upper_peak_values": [float(ys[i]) for i in top_two],
        "upper_anchor_values": [float(upper_anchor[i]) for i in top_two],
        "lower_min_indices": lower_two,
        "lower_min_times_s": [float(t[i]) for i in lower_two],
        "lower_min_values": [float(ys[i]) for i in lower_two],
        "lower_anchor_values": [float(lower_anchor[i]) for i in lower_two],
        "upper_line_slope_intensity_per_s": float(upper_m),
        "lower_line_slope_intensity_per_s": float(lower_m),
        "peak_idx_used_for_slope_corrected_peak_to_min": peak_idx,
        "min_idx_used_for_slope_corrected_peak_to_min": min_idx,
        "observed_peak_to_min_intensity": observed_peak_to_min,
        "expected_trend_change_peak_to_min_intensity": expected_trend_change,
        "slope_corrected_peak_to_min_from_points": slope_corrected_from_points,
        "half_amplitude_from_slope_corrected_points": half_amp_from_points,
        "mean_half_amplitude_intensity": half_amp_mean,
        "slope_corrected_peak_to_min_intensity": slope_corrected_peak_to_min,
        "half_amplitude_percent_of_rod_signal": half_amp_percent,
        "slope_corrected_peak_to_min_percent_of_rod_signal": slope_corrected_peak_to_min_percent,
        "residual_half_p05_p95_about_center_line": residual_amp,
        "rod_signal_reference": rod_signal,
    }
    (analysis_dir / "slope_corrected_peak_to_min_amplitude_summary.json").write_text(json.dumps(env_summary, indent=2))

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(t, y, lw=0.55, alpha=0.35, label="raw 11x11 mean")
    ax.plot(t, ys, lw=1.5, label="smoothed z trace")
    ax.plot(t, upper, color="tab:red", lw=2.0, label="upper envelope line")
    ax.plot(t, lower, color="tab:blue", lw=2.0, label="lower envelope line")
    ax.plot(t, center, color="black", lw=1.5, ls="--", label="midline trend")
    ax.scatter(t[top_two], upper_anchor[top_two], color="tab:red", s=45, zorder=5)
    ax.scatter(t[lower_two], lower_anchor[lower_two], color="tab:blue", s=45, zorder=5)
    marker_idx = int(np.where(overlap)[0][len(np.where(overlap)[0]) // 2])
    ax.annotate(
        "",
        xy=(t[marker_idx], upper[marker_idx]),
        xytext=(t[marker_idx], center[marker_idx]),
        arrowprops=dict(arrowstyle="<->", color="tab:green", lw=2.0),
    )
    ax.text(
        t[marker_idx],
        0.5 * (upper[marker_idx] + center[marker_idx]),
        f"half amplitude = {half_amp_mean:.2f}\n({half_amp_percent:.2f}% of rod)",
        color="tab:green",
        fontsize=8,
        ha="left",
        va="center",
        bbox=dict(facecolor="white", edgecolor="tab:green", alpha=0.8),
    )
    ax.set_xlabel("time (s)")
    ax.set_ylabel("central 11x11 mean intensity")
    ax.set_title("Z-scan slope-corrected peak-to-minimum amplitude estimate")
    ax.legend(fontsize=8)
    fig.tight_layout()
    plot_path = analysis_dir / "z_scan_slope_corrected_peak_to_min_amplitude.png"
    fig.savefig(plot_path, dpi=240)
    sim_dir = analysis_dir / "pixel_resolved_interference_simulation"
    if sim_dir.exists():
        fig.savefig(sim_dir / "z_scan_intensity_fit_halfway_amplitude.png", dpi=240)
    plt.close(fig)

    labels = [r["case"] for r in plot_rows]
    values = [float(r["slope_corrected_peak_to_min_percent_of_rod_signal"]) for r in plot_rows]
    colors = ["tab:blue"] * (len(values) - 1) + ["tab:red"]
    fig, ax = plt.subplots(figsize=(9, 5.2))
    ax.barh(np.arange(len(values)), values, color=colors)
    ax.set_yticks(np.arange(len(values)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Slope-corrected peak-to-minimum variation (% of rod signal)")
    ax.set_title("Expected interference variation vs slope-corrected measured variation")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(analysis_dir / "expected_vs_actual_slope_corrected_peak_to_min_percent.png", dpi=240)
    plt.close(fig)

    print(json.dumps(env_summary, indent=2))


if __name__ == "__main__":
    main()
