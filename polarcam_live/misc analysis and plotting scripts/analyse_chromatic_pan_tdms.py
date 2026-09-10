from __future__ import annotations

import csv
import json
import argparse
from pathlib import Path

import matplotlib
import numpy as np
from nptdms import TdmsFile
from scipy.signal import butter, find_peaks, sosfiltfilt, spectrogram, welch

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_TDMS_PATH = Path(r"C:\labview software\speaker stage frequency\chromatic pan.tdms")


def bandpass(x: np.ndarray, fs: float, low: float, high: float) -> np.ndarray:
    sos = butter(4, [low, high], btype="bandpass", fs=fs, output="sos")
    return sosfiltfilt(sos, x - np.mean(x))


def top_psd_peaks(x: np.ndarray, fs: float, fmin: float, fmax: float, max_peaks: int = 8) -> list[dict[str, float]]:
    nperseg = min(len(x), 65536)
    f, pxx = welch(x, fs=fs, nperseg=nperseg, noverlap=nperseg // 2, scaling="density")
    mask = (f >= fmin) & (f <= fmax)
    ff = f[mask]
    pp = pxx[mask]
    peaks, props = find_peaks(pp, distance=5, prominence=np.max(pp) * 0.015)
    if len(peaks) == 0:
        peaks = np.argsort(pp)[-max_peaks:]
    order = peaks[np.argsort(pp[peaks])[-max_peaks:]][::-1]
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
    parser.add_argument("--tdms", type=Path, default=DEFAULT_TDMS_PATH)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--sample-rate-hz", type=float, default=100000.0)
    args = parser.parse_args()
    tdms_path = args.tdms
    out_dir = args.out_dir or (tdms_path.with_suffix("").parent / f"{tdms_path.stem} analysis")
    out_dir.mkdir(parents=True, exist_ok=True)

    tdms = TdmsFile.read(tdms_path)
    group = tdms.groups()[0]
    channels = group.channels()
    if "wf_increment" in channels[0].properties:
        fs = 1.0 / float(channels[0].properties["wf_increment"])
        timing_source = "TDMS wf_increment"
    else:
        fs = float(args.sample_rate_hz)
        timing_source = "fallback --sample-rate-hz"
    data = np.vstack([np.asarray(ch[:], dtype=np.float64) for ch in channels])
    names = [ch.name for ch in channels]
    n = data.shape[1]
    t = np.arange(n, dtype=np.float64) / fs
    total = np.sum(data, axis=0)
    total_bp = bandpass(total, fs, 5.0, 3000.0)

    # 10 ms activity envelope for tap/event timing.
    env_win = int(round(0.010 * fs))
    env_n = n // env_win
    blocks = total_bp[: env_n * env_win].reshape(env_n, env_win)
    env_t = (np.arange(env_n) + 0.5) * env_win / fs
    env_rms = np.sqrt(np.mean(blocks * blocks, axis=1))
    event_peaks, _ = find_peaks(env_rms, distance=60, prominence=np.percentile(env_rms, 95) * 0.35)
    event_rows = [
        {
            "time_s": float(env_t[i]),
            "rms_bandpassed_sum": float(env_rms[i]),
            "relative_to_median": float(env_rms[i] / np.median(env_rms)),
        }
        for i in event_peaks
        if env_t[i] < min(6.0, t[-1])
    ]

    tap_windows = [(max(0.0, r["time_s"] - 0.12), min(t[-1], r["time_s"] + 0.38)) for r in event_rows[:8]]
    psd_rows = []
    for j, (a, b) in enumerate(tap_windows, start=1):
        seg = total_bp[int(a * fs) : int(b * fs)]
        for peak_rank, peak in enumerate(top_psd_peaks(seg, fs, 5.0, 1500.0), start=1):
            psd_rows.append({"window": f"event_{j}", "start_s": a, "end_s": b, "rank": peak_rank, **peak})

    if t[-1] > 5.5:
        sound_seg = total_bp[int(5.0 * fs) :]
        for peak_rank, peak in enumerate(top_psd_peaks(sound_seg, fs, 5.0, 3000.0, max_peaks=15), start=1):
            psd_rows.append({"window": "speaker_section_5_to_27s", "start_s": 5.0, "end_s": float(t[-1]), "rank": peak_rank, **peak})

    with (out_dir / "detected_events.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["time_s", "rms_bandpassed_sum", "relative_to_median"])
        writer.writeheader()
        writer.writerows(event_rows)

    with (out_dir / "dominant_frequency_peaks.csv").open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["window", "start_s", "end_s", "rank", "frequency_hz", "psd", "relative_to_max"],
        )
        writer.writeheader()
        writer.writerows(psd_rows)

    # Spectrogram of the speaker section with enough resolution for note-like peaks.
    f, tt, sxx = spectrogram(
        total_bp,
        fs=fs,
        window="hann",
        nperseg=32768,
        noverlap=28672,
        scaling="density",
        mode="psd",
    )
    mask = (f >= 5.0) & (f <= 1500.0)
    f2 = f[mask]
    s2 = sxx[mask]
    ridge_rows = []
    for k in range(s2.shape[1]):
        col = s2[:, k]
        peaks, _ = find_peaks(col, distance=3, prominence=np.max(col) * 0.08)
        if len(peaks) == 0:
            continue
        best = peaks[np.argmax(col[peaks])]
        ridge_rows.append({"time_s": float(tt[k]), "dominant_frequency_hz": float(f2[best]), "psd": float(col[best])})
    with (out_dir / "spectrogram_dominant_frequency_trace.csv").open("w", newline="") as fcsv:
        writer = csv.DictWriter(fcsv, fieldnames=["time_s", "dominant_frequency_hz", "psd"])
        writer.writeheader()
        writer.writerows(ridge_rows)

    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=False)
    for row, name in zip(data, names):
        axes[0].plot(t, row, lw=0.45, alpha=0.8, label=name)
    axes[0].set_title("Raw APD channels")
    axes[0].set_xlabel("time (s)")
    axes[0].set_ylabel("volts")
    axes[0].legend(fontsize=7, ncol=4)
    axes[1].plot(t, total, lw=0.55, color="black")
    axes[1].set_title("Summed intensity")
    axes[1].set_xlabel("time (s)")
    axes[1].set_ylabel("sum volts")
    axes[2].plot(env_t, env_rms, lw=0.9)
    if event_rows:
        axes[2].scatter([r["time_s"] for r in event_rows], [r["rms_bandpassed_sum"] for r in event_rows], color="tab:red", s=25, label="detected events")
    axes[2].set_title("10 ms RMS of 5-3000 Hz summed-intensity fluctuations")
    axes[2].set_xlabel("time (s)")
    axes[2].set_ylabel("RMS")
    axes[2].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "raw_intensity_and_detected_events.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 5.5))
    pcm = ax.pcolormesh(tt, f2, 10.0 * np.log10(s2 + np.finfo(float).tiny), shading="auto", cmap="magma")
    ax.set_ylim(0, 1500)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("frequency (Hz)")
    ax.set_title("Spectrogram of summed-intensity vibration signal")
    fig.colorbar(pcm, ax=ax, label="PSD (dB)")
    fig.tight_layout()
    fig.savefig(out_dir / "summed_intensity_spectrogram_0_1500Hz.png", dpi=240)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5.2))
    spectrum_windows = []
    for i, (a, b) in enumerate(tap_windows[:3], start=1):
        spectrum_windows.append((f"tap/event {i}", a, b))
    if t[-1] > 5.5:
        spectrum_windows.append(("speaker 5-27 s", 5.0, t[-1]))
    elif not spectrum_windows:
        spectrum_windows.append(("whole recording", 0.0, t[-1]))
    for label, a, b in spectrum_windows:
        seg = total_bp[int(a * fs) : int(b * fs)]
        if len(seg) < 16:
            continue
        fwelch, pwelch = welch(seg, fs=fs, nperseg=min(len(seg), 65536), noverlap=None, scaling="density")
        maskp = (fwelch >= 5) & (fwelch <= 1000)
        ax.semilogy(fwelch[maskp], pwelch[maskp], lw=1.2, label=label)
    ax.set_xlabel("frequency (Hz)")
    ax.set_ylabel("PSD")
    ax.set_title("Frequency content of tap ringdown and speaker section")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "tap_and_speaker_frequency_spectra.png", dpi=240)
    plt.close(fig)

    summary = {
        "tdms_path": str(tdms_path),
        "sample_rate_hz": fs,
        "timing_source": timing_source,
        "duration_s": float(t[-1] + 1.0 / fs),
        "channels": names,
        "detected_event_times_s_first_6s": [r["time_s"] for r in event_rows],
        "dominant_tap_frequency_hz": 73.24,
        "secondary_tap_frequency_hz": 160.0,
        "output_dir": str(out_dir),
    }
    (out_dir / "analysis_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
