from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
import numpy as np
from scipy.signal import find_peaks, spectrogram, welch

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_FPS = 1640.0709219858156


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


def load_fps(recording_dir: Path) -> float:
    meta = recording_dir / "capture_maxfps_15x15_meta.json"
    if not meta.exists():
        return DEFAULT_FPS
    data = json.loads(meta.read_text())
    return float(data.get("actual", {}).get("fps") or DEFAULT_FPS)


def central_11x11_trace(stack: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    roi = np.asarray(stack[:, :14, :14], dtype=np.float64)
    central = roi[:, 1:12, 1:12]
    return np.mean(central, axis=(1, 2)), np.sum(central, axis=(1, 2))


def write_peak_table(path: Path, trace: np.ndarray, fps: float) -> list[dict[str, float]]:
    x = np.asarray(trace, dtype=np.float64) - float(np.mean(trace))
    nperseg = min(len(x), 1024)
    f, pxx = welch(x, fs=fps, nperseg=nperseg, noverlap=nperseg // 2, scaling="density")
    mask = (f >= 1.0) & (f <= fps / 2.0 * 0.98)
    ff = f[mask]
    pp = pxx[mask]
    peaks, _ = find_peaks(pp, distance=2, prominence=np.max(pp) * 0.02)
    if len(peaks) == 0:
        peaks = np.argsort(pp)[-12:]
    order = peaks[np.argsort(pp[peaks])[-12:]][::-1]
    rows = [
        {
            "rank": int(rank),
            "frequency_hz": float(ff[i]),
            "psd": float(pp[i]),
            "relative_to_max": float(pp[i] / np.max(pp)),
        }
        for rank, i in enumerate(order, start=1)
    ]
    with path.open("w", newline="") as fcsv:
        writer = csv.DictWriter(fcsv, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("recording_dir", type=Path)
    args = parser.parse_args()
    rec = args.recording_dir
    out = rec / "sound_frequency_response_analysis"
    out.mkdir(parents=True, exist_ok=True)

    fps = load_fps(rec)
    stack = strip_marker(np.load(rec / "capture_maxfps_15x15.npy", mmap_mode="r"))
    mean_trace, sum_trace = central_11x11_trace(stack)
    t = np.arange(mean_trace.size, dtype=np.float64) / fps
    x = mean_trace - float(np.mean(mean_trace))

    nperseg = min(512, max(64, int(2 ** np.floor(np.log2(max(64, len(x) // 2))))))
    noverlap = int(0.85 * nperseg)
    f, tt, sxx = spectrogram(
        x,
        fs=fps,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        scaling="density",
        mode="psd",
    )
    mask = (f >= 1.0) & (f <= fps / 2.0 * 0.98)

    rows = write_peak_table(out / "frequency_peaks_central_11x11_mean.csv", mean_trace, fps)
    write_peak_table(out / "frequency_peaks_central_11x11_sum.csv", sum_trace, fps)

    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=False)
    axes[0].plot(t, mean_trace, lw=0.8)
    axes[0].set_xlabel("time (s)")
    axes[0].set_ylabel("central 11x11 mean intensity")
    axes[0].set_title("NPY intensity trace")
    pcm = axes[1].pcolormesh(
        tt,
        f[mask],
        10.0 * np.log10(sxx[mask] + np.finfo(float).tiny),
        shading="auto",
        cmap="magma",
    )
    axes[1].set_ylim(0, fps / 2.0 * 0.98)
    axes[1].set_xlabel("time (s)")
    axes[1].set_ylabel("frequency (Hz)")
    axes[1].set_title("Spectral heatmap of intensity fluctuations")
    fig.colorbar(pcm, ax=axes[1], label="PSD (dB)")
    fig.tight_layout()
    fig.savefig(out / "intensity_trace_and_spectral_heatmap.png", dpi=240)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    pcm = ax.pcolormesh(
        tt,
        f[mask],
        10.0 * np.log10(sxx[mask] + np.finfo(float).tiny),
        shading="auto",
        cmap="magma",
    )
    ax.set_ylim(0, fps / 2.0 * 0.98)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("frequency (Hz)")
    ax.set_title("Spectral heatmap, central 11x11 mean intensity")
    fig.colorbar(pcm, ax=ax, label="PSD (dB)")
    fig.tight_layout()
    fig.savefig(out / "spectral_heatmap_0_to_nyquist.png", dpi=240)
    plt.close(fig)

    fwelch, pwelch = welch(x, fs=fps, nperseg=min(len(x), 1024), noverlap=min(len(x) // 2, 512), scaling="density")
    maskp = (fwelch >= 1.0) & (fwelch <= fps / 2.0 * 0.98)
    fig, ax = plt.subplots(figsize=(9, 5.2))
    ax.semilogy(fwelch[maskp], pwelch[maskp], lw=1.2)
    for row in rows[:8]:
        ax.axvline(row["frequency_hz"], color="tab:red", alpha=0.25, lw=0.8)
    ax.set_xlabel("frequency (Hz)")
    ax.set_ylabel("PSD")
    ax.set_title("Mean intensity spectrum")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out / "mean_intensity_spectrum.png", dpi=240)
    plt.close(fig)

    summary = {
        "recording_dir": str(rec),
        "shape": [int(v) for v in stack.shape],
        "fps": fps,
        "duration_s": float(len(mean_trace) / fps),
        "nyquist_hz": float(fps / 2.0),
        "spectrogram_nperseg": int(nperseg),
        "spectrogram_frequency_resolution_hz": float(fps / nperseg),
        "top_frequency_peaks_hz": [r["frequency_hz"] for r in rows[:8]],
        "output_dir": str(out),
    }
    (out / "analysis_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
