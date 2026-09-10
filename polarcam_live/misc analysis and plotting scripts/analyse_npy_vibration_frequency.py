from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
import numpy as np
from scipy.signal import butter, find_peaks, sosfiltfilt, spectrogram, welch

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_FPS = 1640.0709219858156
CENTRAL_WINDOW = 11


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


def central_crop(stack: np.ndarray, window: int = CENTRAL_WINDOW) -> np.ndarray:
    roi = np.asarray(stack[:, :14, :14], dtype=np.float64)
    h, w = roi.shape[1:]
    y0 = (h - window) // 2
    x0 = (w - window) // 2
    return roi[:, y0 : y0 + window, x0 : x0 + window]


def top_peaks(trace: np.ndarray, fps: float, fmin: float = 1.0, fmax: float | None = None) -> list[dict[str, float]]:
    x = np.asarray(trace, dtype=np.float64)
    x = x - np.mean(x)
    fmax = min(float(fmax if fmax is not None else fps / 2.0), fps / 2.0 * 0.98)
    nperseg = min(len(x), 4096)
    f, pxx = welch(x, fs=fps, nperseg=nperseg, noverlap=nperseg // 2, scaling="density")
    mask = (f >= fmin) & (f <= fmax)
    ff = f[mask]
    pp = pxx[mask]
    peaks, _ = find_peaks(pp, distance=3, prominence=np.max(pp) * 0.02)
    if len(peaks) == 0:
        peaks = np.argsort(pp)[-10:]
    order = peaks[np.argsort(pp[peaks])[-12:]][::-1]
    return [
        {
            "frequency_hz": float(ff[i]),
            "psd": float(pp[i]),
            "relative_to_max": float(pp[i] / np.max(pp)),
        }
        for i in order
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("npy", type=Path)
    parser.add_argument("--fps", type=float, default=DEFAULT_FPS)
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()

    npy_path = args.npy
    fps = float(args.fps)
    out_dir = args.out_dir or (npy_path.parent / "npy_frequency_analysis")
    out_dir.mkdir(parents=True, exist_ok=True)

    stack = strip_marker(np.load(npy_path, mmap_mode="r"))
    crop = central_crop(stack)
    trace_mean = np.mean(crop, axis=(1, 2))
    trace_sum = np.sum(crop, axis=(1, 2))
    t = np.arange(trace_mean.size, dtype=np.float64) / fps

    high = min(0.98 * fps / 2.0, 800.0)
    sos = butter(4, [2.0, high], btype="bandpass", fs=fps, output="sos")
    bp = sosfiltfilt(sos, trace_mean - np.mean(trace_mean))

    env_win = max(3, int(round(0.020 * fps)))
    n_blocks = len(bp) // env_win
    blocks = bp[: n_blocks * env_win].reshape(n_blocks, env_win)
    env_t = (np.arange(n_blocks) + 0.5) * env_win / fps
    env_rms = np.sqrt(np.mean(blocks * blocks, axis=1))
    event_peaks, _ = find_peaks(env_rms, distance=max(1, int(round(0.15 / (env_win / fps)))), prominence=np.percentile(env_rms, 90) * 0.4)
    events = [
        {
            "time_s": float(env_t[i]),
            "rms_bandpassed_mean_intensity": float(env_rms[i]),
            "relative_to_median": float(env_rms[i] / np.median(env_rms)),
        }
        for i in event_peaks
    ]

    peak_rows = []
    for name, tr in [("central_11x11_mean", trace_mean), ("central_11x11_sum", trace_sum)]:
        for rank, row in enumerate(top_peaks(tr, fps, 1.0, high), start=1):
            peak_rows.append({"trace": name, "rank": rank, **row})

    with (out_dir / "frequency_peaks.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["trace", "rank", "frequency_hz", "psd", "relative_to_max"])
        writer.writeheader()
        writer.writerows(peak_rows)

    with (out_dir / "detected_events.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["time_s", "rms_bandpassed_mean_intensity", "relative_to_median"])
        writer.writeheader()
        writer.writerows(events)

    f, tt, sxx = spectrogram(bp, fs=fps, window="hann", nperseg=min(512, len(bp)), noverlap=min(448, max(0, len(bp) // 2)), scaling="density", mode="psd")
    mask = (f >= 1.0) & (f <= high)

    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=False)
    axes[0].plot(t, trace_mean, lw=0.7)
    axes[0].set_title("Central 11x11 mean intensity")
    axes[0].set_xlabel("time (s)")
    axes[0].set_ylabel("mean intensity")
    axes[1].plot(t, bp, lw=0.7)
    axes[1].set_title("Bandpassed mean-intensity fluctuation")
    axes[1].set_xlabel("time (s)")
    axes[1].set_ylabel("intensity")
    axes[2].plot(env_t, env_rms, lw=0.9)
    if events:
        axes[2].scatter([e["time_s"] for e in events], [e["rms_bandpassed_mean_intensity"] for e in events], color="tab:red", s=25)
    axes[2].set_title("20 ms RMS envelope with detected events")
    axes[2].set_xlabel("time (s)")
    axes[2].set_ylabel("RMS")
    fig.tight_layout()
    fig.savefig(out_dir / "intensity_trace_and_detected_events.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 5.5))
    pcm = ax.pcolormesh(tt, f[mask], 10.0 * np.log10(sxx[mask] + np.finfo(float).tiny), shading="auto", cmap="magma")
    ax.set_ylim(0, high)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("frequency (Hz)")
    ax.set_title("Spectrogram of central 11x11 mean-intensity fluctuation")
    fig.colorbar(pcm, ax=ax, label="PSD (dB)")
    fig.tight_layout()
    fig.savefig(out_dir / "npy_intensity_spectrogram.png", dpi=240)
    plt.close(fig)

    fwelch, pwelch = welch(trace_mean - np.mean(trace_mean), fs=fps, nperseg=min(len(trace_mean), 4096), noverlap=None, scaling="density")
    maskp = (fwelch >= 1.0) & (fwelch <= high)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.semilogy(fwelch[maskp], pwelch[maskp], lw=1.2)
    for row in peak_rows[:8]:
        if row["trace"] == "central_11x11_mean":
            ax.axvline(row["frequency_hz"], color="tab:red", alpha=0.25, lw=0.8)
    ax.set_xlabel("frequency (Hz)")
    ax.set_ylabel("PSD")
    ax.set_title("Central 11x11 mean-intensity frequency spectrum")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "npy_intensity_frequency_spectrum.png", dpi=240)
    plt.close(fig)

    summary = {
        "npy_path": str(npy_path),
        "shape": list(stack.shape),
        "fps_assumed": fps,
        "duration_s": float(len(trace_mean) / fps),
        "nyquist_hz": float(fps / 2.0),
        "top_mean_intensity_frequency_peaks_hz": [r["frequency_hz"] for r in peak_rows if r["trace"] == "central_11x11_mean"][:8],
        "detected_event_times_s": [e["time_s"] for e in events],
        "out_dir": str(out_dir),
    }
    (out_dir / "analysis_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
