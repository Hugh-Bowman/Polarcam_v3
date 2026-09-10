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
DATA_DIR = ROOT / "datasets" / "background characterisation" / "z scan of a rod" / "pending"
OUT_DIR = DATA_DIR / "z_scan_interference_analysis"

ROD_DIR = DATA_DIR / "rod_x1587_y1619_20260720-130213_1784548933621193100"
BG_DIR = DATA_DIR / "rod_x1587_y1619_20260720-130226_1784548946034183400"

RAW_CROP = 14
CENTRAL_WINDOW = 11
FPS = 1640.0709219858156


def strip_marker(arr: np.ndarray) -> np.ndarray:
    a = np.asarray(arr)
    if a.ndim < 3 or a.shape[0] < 2:
        return a
    marker = np.asarray(a[-1])
    if marker.ndim != 2:
        return a
    nz = np.argwhere(marker != 0)
    if nz.shape[0] != 1:
        return a
    my, mx = int(nz[0][0]), int(nz[0][1])
    if float(marker[my, mx]) == 1.0 and float(np.sum(marker, dtype=np.float64)) == 1.0:
        return np.asarray(a[:-1])
    return a


def first_crop(stack: np.ndarray, side: int = RAW_CROP) -> np.ndarray:
    return np.asarray(stack[:, :side, :side], dtype=np.float64)


def central_crop(crop: np.ndarray, side: int = CENTRAL_WINDOW) -> np.ndarray:
    h, w = crop.shape[1:]
    y0 = max(0, (h - side) // 2)
    x0 = max(0, (w - side) // 2)
    return crop[:, y0 : y0 + side, x0 : x0 + side]


def moving_average_reflect(x: np.ndarray, window: int) -> np.ndarray:
    window = max(3, int(window))
    if window % 2 == 0:
        window += 1
    pad = window // 2
    xp = np.pad(np.asarray(x, dtype=np.float64), pad, mode="reflect")
    kernel = np.ones(window, dtype=np.float64) / window
    return np.convolve(xp, kernel, mode="valid")


def robust_amp(x: np.ndarray) -> dict[str, float]:
    x = np.asarray(x, dtype=np.float64)
    p01, p05, p50, p95, p99 = np.percentile(x, [1, 5, 50, 95, 99])
    return {
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
        "p01": float(p01),
        "p05": float(p05),
        "p50": float(p50),
        "p95": float(p95),
        "p99": float(p99),
        "half_p05_p95": float(0.5 * (p95 - p05)),
        "half_p01_p99": float(0.5 * (p99 - p01)),
    }


def split_channel_means(crop: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "I0": crop[:, 0::2, 0::2].mean(axis=(1, 2)),
        "I45": crop[:, 0::2, 1::2].mean(axis=(1, 2)),
        "I135": crop[:, 1::2, 0::2].mean(axis=(1, 2)),
        "I90": crop[:, 1::2, 1::2].mean(axis=(1, 2)),
    }


def xy_from_channels(ch: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    eps = 1e-9
    x = (ch["I0"] - ch["I90"]) / (ch["I0"] + ch["I90"] + eps)
    y = (ch["I45"] - ch["I135"]) / (ch["I45"] + ch["I135"] + eps)
    r = np.sqrt(x * x + y * y)
    return x, y, r


def fft_peak(trace: np.ndarray, fps: float) -> dict[str, float]:
    x = np.asarray(trace, dtype=np.float64)
    x = x - np.mean(x)
    x *= np.hanning(x.size)
    spec = np.abs(np.fft.rfft(x)) ** 2
    freqs = np.fft.rfftfreq(x.size, d=1.0 / fps)
    valid = freqs > 1.0
    if not np.any(valid):
        return {"peak_hz": float("nan"), "peak_power_fraction": float("nan")}
    idxs = np.where(valid)[0]
    peak = idxs[int(np.argmax(spec[valid]))]
    return {
        "peak_hz": float(freqs[peak]),
        "peak_power_fraction": float(spec[peak] / max(np.sum(spec[valid]), 1e-30)),
    }


def expected_amplitudes(rod_i: float, bg_i: float, window_px: int) -> list[dict[str, float | str]]:
    coherent = 2.0 * np.sqrt(max(rod_i, 0.0) * max(bg_i, 0.0))
    cases = []
    for label, corr_px in [
        ("random phase per raw pixel across 11px window", 1.0),
        ("intermediate phase scale 3px", 3.0),
        ("intermediate phase scale 5px", 5.0),
        ("intermediate phase scale 7px", 7.0),
        ("constant phase across 11px window", float(window_px)),
    ]:
        n_eff = max(1.0, (float(window_px) / corr_px) ** 2)
        amp = coherent / np.sqrt(n_eff)
        cases.append(
            {
                "case": label,
                "phase_correlation_length_px": corr_px,
                "n_eff": n_eff,
                "amplitude_intensity_units": float(amp),
                "amplitude_percent_of_rod_signal": float(100.0 * amp / max(rod_i, 1e-12)),
                "amplitude_percent_of_total_mean": float(100.0 * amp / max(rod_i + bg_i, 1e-12)),
            }
        )
    return cases


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rod-dir", type=Path, default=ROD_DIR)
    parser.add_argument("--bg-dir", type=Path, default=BG_DIR)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()
    out_dir = args.out_dir
    rod_dir = args.rod_dir
    bg_dir = args.bg_dir

    out_dir.mkdir(parents=True, exist_ok=True)
    rod = strip_marker(np.load(rod_dir / "capture_maxfps_15x15.npy", mmap_mode="r"))
    bg = strip_marker(np.load(bg_dir / "capture_maxfps_15x15.npy", mmap_mode="r"))
    rod14 = first_crop(rod, RAW_CROP)
    bg14 = first_crop(bg, RAW_CROP)
    rod11 = central_crop(rod14, CENTRAL_WINDOW)
    bg11 = central_crop(bg14, CENTRAL_WINDOW)

    traces = {
        "rod_14x14_mean": rod14.mean(axis=(1, 2)),
        "background_14x14_mean": bg14.mean(axis=(1, 2)),
        "rod_11x11_central_mean": rod11.mean(axis=(1, 2)),
        "background_11x11_central_mean": bg11.mean(axis=(1, 2)),
    }
    rod_trace = traces["rod_11x11_central_mean"]
    bg_trace = traces["background_11x11_central_mean"]
    bg_mean = float(np.mean(bg_trace))
    total_mean = float(np.mean(rod_trace))
    rod_signal_mean = float(max(total_mean - bg_mean, 1e-12))

    smooth_window = int(round(0.12 * FPS))
    trend = moving_average_reflect(rod_trace, smooth_window)
    residual = rod_trace - trend
    bg_demean = bg_trace - np.mean(bg_trace)

    summary = {
        "rod_recording": str(rod_dir),
        "background_recording": str(bg_dir),
        "fps": FPS,
        "intensity_measure_used": "central 11x11 raw-pixel mean inside the first 14x14 inspection crop",
        "rod_total_mean": total_mean,
        "background_mean_b_squared": bg_mean,
        "estimated_rod_signal_r_squared": rod_signal_mean,
        "background_percent_of_rod_signal": 100.0 * bg_mean / rod_signal_mean,
        "field_ratio_b_over_r": float(np.sqrt(bg_mean / rod_signal_mean)),
        "rod_total_trace_stats": robust_amp(rod_trace),
        "rod_slow_envelope_stats": robust_amp(trend),
        "rod_residual_after_0p12s_moving_average_stats": robust_amp(residual),
        "background_trace_stats": robust_amp(bg_trace),
        "background_demeaned_stats": robust_amp(bg_demean),
        "rod_residual_fft": fft_peak(residual, FPS),
    }

    expected = expected_amplitudes(rod_signal_mean, bg_mean, CENTRAL_WINDOW)
    with (out_dir / "expected_interference_amplitudes.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(expected[0].keys()))
        writer.writeheader()
        writer.writerows(expected)
    with (out_dir / "z_scan_summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    with (out_dir / "intensity_traces.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "time_s", *traces.keys(), "rod_11x11_trend", "rod_11x11_residual", "background_11x11_demeaned"])
        n = min(len(rod_trace), len(bg_trace))
        for i in range(n):
            writer.writerow(
                [
                    i,
                    i / FPS,
                    traces["rod_14x14_mean"][i],
                    traces["background_14x14_mean"][i],
                    traces["rod_11x11_central_mean"][i],
                    traces["background_11x11_central_mean"][i],
                    trend[i],
                    residual[i],
                    bg_demean[i],
                ]
            )

    # Channel traces and anisotropy.
    ch = split_channel_means(rod11)
    x, y, rr = xy_from_channels(ch)
    with (out_dir / "rod_channel_xy_traces.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "time_s", "I0", "I45", "I135", "I90", "X", "Y", "r"])
        for i in range(len(x)):
            writer.writerow([i, i / FPS, ch["I0"][i], ch["I45"][i], ch["I135"][i], ch["I90"][i], x[i], y[i], rr[i]])

    t = np.arange(len(rod_trace)) / FPS
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(t, rod_trace, lw=1.0, label="rod total 11x11 mean")
    ax.plot(t, trend, lw=2.0, label="0.12 s moving-average trend")
    ax.axhline(bg_mean, color="tab:red", ls="--", label="background mean")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("mean intensity")
    ax.set_title("Rod z-scan intensity trace")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "rod_z_scan_intensity_trace.png", dpi=240)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(t, residual, lw=0.8, label="rod residual")
    ax.plot(np.arange(len(bg_demean)) / FPS, bg_demean, lw=0.8, alpha=0.7, label="background demeaned")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("mean intensity")
    ax.set_title("Fast residual after removing slow z-scan envelope")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "rod_z_scan_fast_residual_vs_background.png", dpi=240)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    cases = [e["case"] for e in expected]
    amps = [float(e["amplitude_percent_of_rod_signal"]) for e in expected]
    measured = 100.0 * summary["rod_residual_after_0p12s_moving_average_stats"]["half_p05_p95"] / rod_signal_mean
    ax.barh(range(len(cases)), amps, label="expected")
    ax.axvline(measured, color="tab:red", lw=2, label="measured residual half p05-p95")
    ax.set_yticks(range(len(cases)))
    ax.set_yticklabels(cases, fontsize=8)
    ax.set_xlabel("amplitude (% of rod signal)")
    ax.set_title("Expected interference amplitude vs measured fast residual")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "expected_vs_measured_interference_amplitude.png", dpi=240)
    plt.close(fig)

    readme = [
        "Z-scan rod/background interference analysis",
        "",
        f"Rod recording: {rod_dir.name}",
        f"Background recording: {bg_dir.name}",
        "",
        "Intensity measure used:",
        "- central 11x11 raw-pixel mean inside the first 14x14 inspection crop.",
        "- This is an intensity average, not an X/Y anisotropy.",
        "",
        f"Rod total mean = {total_mean:.6g}",
        f"Background mean b^2 = {bg_mean:.6g}",
        f"Estimated rod signal r^2 = total-background = {rod_signal_mean:.6g}",
        f"Background / rod = {100.0 * bg_mean / rod_signal_mean:.3g}%",
        "",
        "Interpretation:",
        "- The total z-scan trace has a large slow envelope, mainly focus/collection change with z.",
        "- The interference comparison uses the faster residual after subtracting a 0.12 s moving average.",
        "- If the phase is constant across the integration window, the possible 2rb modulation is very large.",
        "- The measured fast residual is much smaller than the coherent limit and closest to a random/intermediate phase case.",
    ]
    (out_dir / "README_z_scan_interference_analysis.txt").write_text("\n".join(readme) + "\n")
    print(f"Wrote outputs to {out_dir}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
