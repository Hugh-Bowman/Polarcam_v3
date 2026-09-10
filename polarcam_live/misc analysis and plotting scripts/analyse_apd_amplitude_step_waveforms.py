from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
from nptdms import TdmsFile
from scipy import signal

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_TDMS = Path(r"E:\vibration amplitude callibration\0508 2-10VpK 910hz.tdms")


def _parse_voltage_range(path: Path) -> tuple[int, int]:
    m = re.search(r"(\d+)\s*-\s*(\d+)\s*VpK", path.name, flags=re.IGNORECASE)
    if not m:
        return 2, 10
    return int(m.group(1)), int(m.group(2))


def _make_voltage_labels(path: Path, voltage_step: float = 1.0) -> list[float]:
    v0, v1 = _parse_voltage_range(path)
    step = max(1e-9, float(voltage_step))
    labels = []
    v = float(v0)
    while v <= float(v1) + 0.5 * step:
        labels.append(round(v, 6))
        v += step
    return labels


def _fmt_voltage(value: float) -> str:
    return f"{float(value):g}"


def _read_tdms_channels(path: Path) -> tuple[np.ndarray, float, list[str]]:
    tdms = TdmsFile.read(path)
    groups = tdms.groups()
    if not groups:
        raise RuntimeError(f"No TDMS groups found in {path}")
    channels = list(groups[0].channels())
    if len(channels) < 1:
        raise RuntimeError(f"No TDMS channels found in {path}")
    names = [ch.name for ch in channels]
    dt = channels[0].properties.get("wf_increment", None)
    if dt is None or float(dt) <= 0:
        raise RuntimeError("TDMS channel is missing wf_increment sampling metadata.")
    arr = np.vstack([np.asarray(ch[:], dtype=np.float64) for ch in channels])
    return arr, 1.0 / float(dt), names


def _select_analysis_signal(
    channels: np.ndarray, names: list[str], mode: str
) -> tuple[np.ndarray, str, int | None]:
    mode_l = str(mode).strip().lower()
    if mode_l == "total":
        return np.sum(channels, axis=0), f"summed total intensity from {len(names)} APD channels", None
    if mode_l == "dimmest":
        means = np.mean(channels, axis=1)
        idx = int(np.argmin(means))
        return channels[idx], f"dimmest APD channel {names[idx]} (mean={float(means[idx]):.6g})", idx
    if mode_l.startswith("ai"):
        wanted = mode_l
        for i, name in enumerate(names):
            if name.lower().endswith("/" + wanted) or name.lower() == wanted:
                return channels[i], f"selected APD channel {names[i]}", i
    raise ValueError("channel mode must be 'total', 'dimmest', or an ai channel name such as ai0")


def _dominant_frequency(y: np.ndarray, fs: float, f_min: float, f_max: float) -> float:
    yy = signal.detrend(np.asarray(y, dtype=np.float64), type="linear")
    nperseg = min(int(len(yy)), 262144)
    if nperseg < 4096:
        raise RuntimeError("Recording too short for frequency identification.")
    freqs, pxx = signal.welch(
        yy,
        fs=fs,
        nperseg=nperseg,
        noverlap=nperseg // 2,
        detrend="constant",
        scaling="density",
    )
    band = (freqs >= float(f_min)) & (freqs <= float(f_max))
    if not np.any(band):
        raise RuntimeError("No frequency bins in requested drive search band.")
    return float(freqs[band][int(np.argmax(pxx[band]))])


def _lockin_envelope(
    y: np.ndarray,
    fs: float,
    freq_hz: float,
    win_s: float = 0.12,
    hop_s: float = 0.02,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = int(len(y))
    win = max(128, int(round(float(win_s) * float(fs))))
    hop = max(1, int(round(float(hop_s) * float(fs))))
    if win >= n:
        win = max(128, n // 4)
    taper = np.hanning(win)
    denom = max(float(np.sum(taper)), 1.0)
    starts = np.arange(0, max(1, n - win + 1), hop, dtype=np.int64)
    t_centers = (starts.astype(np.float64) + 0.5 * float(win)) / float(fs)
    amp = np.empty(len(starts), dtype=np.float64)
    mean = np.empty(len(starts), dtype=np.float64)
    idx = np.arange(win, dtype=np.float64)
    osc = np.exp(-2j * np.pi * float(freq_hz) * idx / float(fs))
    for i, s in enumerate(starts):
        seg = np.asarray(y[s : s + win], dtype=np.float64)
        m = float(np.mean(seg))
        mean[i] = m
        amp[i] = 2.0 * abs(np.sum((seg - m) * taper * osc)) / denom
    return t_centers, amp, mean


def _segment_monotonic_steps(
    t: np.ndarray,
    metric: np.ndarray,
    n_steps: int,
    min_step_s: float = 1.0,
) -> list[tuple[int, int]]:
    """Dynamic-programming piecewise-constant segmentation of the envelope."""
    x = np.asarray(metric, dtype=np.float64)
    if len(x) < n_steps * 3:
        raise RuntimeError("Too few envelope points for requested number of steps.")
    dt = float(np.median(np.diff(t))) if len(t) > 1 else 0.02
    min_len = max(3, int(round(float(min_step_s) / max(dt, 1e-9))))
    n = len(x)
    if n < n_steps * min_len:
        min_len = max(2, n // max(1, n_steps + 1))
    sx = np.concatenate([[0.0], np.cumsum(x)])
    sx2 = np.concatenate([[0.0], np.cumsum(x * x)])

    def cost(i: int, j: int) -> float:
        count = float(j - i)
        if count <= 0:
            return float("inf")
        s = sx[j] - sx[i]
        s2 = sx2[j] - sx2[i]
        return float(s2 - (s * s / count))

    dp = np.full((n_steps + 1, n + 1), np.inf, dtype=np.float64)
    prev = np.full((n_steps + 1, n + 1), -1, dtype=np.int64)
    dp[0, 0] = 0.0
    for k in range(1, n_steps + 1):
        j_min = k * min_len
        j_max = n - (n_steps - k) * min_len
        for j in range(j_min, j_max + 1):
            best_val = np.inf
            best_i = -1
            i_min = (k - 1) * min_len
            i_max = j - min_len
            for i in range(i_min, i_max + 1):
                val = dp[k - 1, i] + cost(i, j)
                if val < best_val:
                    best_val = val
                    best_i = i
            dp[k, j] = best_val
            prev[k, j] = best_i

    end = n
    segments: list[tuple[int, int]] = []
    j = end
    for k in range(n_steps, 0, -1):
        i = int(prev[k, j])
        if i < 0:
            raise RuntimeError("Step segmentation failed.")
        segments.append((i, j))
        j = i
    segments.reverse()
    return segments


def _fixed_duration_segments(
    t_env: np.ndarray,
    n_steps: int,
    step_duration_s: float,
    start_s: float = 0.0,
) -> list[tuple[int, int]]:
    segments: list[tuple[int, int]] = []
    t = np.asarray(t_env, dtype=np.float64)
    for i in range(int(n_steps)):
        t0 = float(start_s) + i * float(step_duration_s)
        t1 = t0 + float(step_duration_s)
        ia = int(np.searchsorted(t, t0, side="left"))
        ib = int(np.searchsorted(t, t1, side="left"))
        ia = max(0, min(len(t) - 1, ia))
        ib = max(ia + 1, min(len(t), ib))
        segments.append((ia, ib))
    return segments


def _best_one_period_slice(
    y: np.ndarray,
    fs: float,
    freq_hz: float,
    t0: float,
    t1: float,
    search_margin_s: float = 0.4,
) -> tuple[np.ndarray, np.ndarray, float]:
    period_n = max(8, int(round(float(fs) / float(freq_hz))))
    s0 = max(0, int(round((float(t0) + float(search_margin_s)) * float(fs))))
    s1 = min(len(y), int(round((float(t1) - float(search_margin_s)) * float(fs))))
    if s1 - s0 < period_n:
        s0 = max(0, int(round(float(t0) * float(fs))))
        s1 = min(len(y), int(round(float(t1) * float(fs))))
    if s1 - s0 < period_n:
        mid = int(round(0.5 * (float(t0) + float(t1)) * float(fs)))
        s0 = max(0, min(len(y) - period_n, mid - period_n // 2))
        s1 = s0 + period_n
    hop = max(1, period_n // 12)
    best_s = s0
    best_span = -np.inf
    for s in range(s0, max(s0 + 1, s1 - period_n + 1), hop):
        seg = np.asarray(y[s : s + period_n], dtype=np.float64)
        span = float(np.percentile(seg, 98) - np.percentile(seg, 2))
        if span > best_span:
            best_span = span
            best_s = s
    seg = np.asarray(y[best_s : best_s + period_n], dtype=np.float64)
    tt = np.arange(len(seg), dtype=np.float64) / float(fs)
    return tt, seg, float(best_s) / float(fs)


def _count_large_cycles(seg_pct: np.ndarray) -> tuple[int, int]:
    smooth_n = max(5, int(round(len(seg_pct) / 40)))
    if smooth_n % 2 == 0:
        smooth_n += 1
    yy = signal.savgol_filter(seg_pct, smooth_n, 2, mode="interp") if len(seg_pct) > smooth_n else seg_pct
    prominence = max(0.25, 0.18 * float(np.percentile(yy, 95) - np.percentile(yy, 5)))
    peaks, _ = signal.find_peaks(yy, prominence=prominence, distance=max(3, len(yy) // 6))
    troughs, _ = signal.find_peaks(-yy, prominence=prominence, distance=max(3, len(yy) // 6))
    return int(len(peaks)), int(len(troughs))


def _plot_step_detection(
    out: Path,
    t_env: np.ndarray,
    amp_pct: np.ndarray,
    segments: list[tuple[int, int]],
    voltages: list[int],
    freq_hz: float,
    region: tuple[float, float],
    signal_label: str,
) -> None:
    fig, ax = plt.subplots(figsize=(12, 5), constrained_layout=True)
    ax.plot(t_env, amp_pct, color="black", lw=1.0)
    ax.axvspan(region[0], region[1], color="tab:green", alpha=0.06, label="analysed step region")
    for (a, b), v in zip(segments, voltages):
        ax.axvspan(float(t_env[a]), float(t_env[max(a, b - 1)]), color="tab:blue", alpha=0.08)
        ax.text(
            0.5 * (float(t_env[a]) + float(t_env[max(a, b - 1)])),
            float(np.nanmax(amp_pct)) * 0.95,
            f"{_fmt_voltage(v)} VpK",
            ha="center",
            va="top",
            fontsize=9,
            rotation=90,
        )
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(f"Lock-in amplitude at {freq_hz:.2f} Hz (% of local mean)")
    ax.set_title(f"APD amplitude steps ({signal_label})")
    ax.legend(frameon=False, loc="upper right")
    fig.savefig(out / "step_detection_lockin_amplitude.png", dpi=180)
    plt.close(fig)


def _auto_step_region(t_env: np.ndarray, amp_pct: np.ndarray, n_steps: int) -> tuple[float, float]:
    """
    Find the initial calibration staircase and stop before later large manual/disturbed sections.
    The low-voltage calibration is the first long section before the envelope jumps far above
    the calibration-scale response.
    """
    t = np.asarray(t_env, dtype=np.float64)
    a = np.asarray(amp_pct, dtype=np.float64)
    if len(t) < 20:
        return float(t[0]), float(t[-1])
    early_end = max(5, int(0.45 * len(a)))
    early = a[:early_end]
    finite = np.isfinite(early)
    if not np.any(finite):
        return float(t[0]), float(t[-1])
    q80 = float(np.nanpercentile(early[finite], 80))
    q95 = float(np.nanpercentile(early[finite], 95))
    threshold = max(8.0, q95 + 2.0 * max(0.25, q95 - q80))
    above = np.where(a > threshold)[0]
    end_idx = len(t) - 1
    if len(above):
        min_end = int(round((n_steps * 1.0) / max(float(np.median(np.diff(t))), 1e-9)))
        for idx in above:
            if idx >= min_end:
                end_idx = max(5, int(idx) - 1)
                break
    return float(t[0]), float(t[end_idx])


def _plot_amplitude_vs_voltage(out: Path, df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    ax.plot(df["voltage_vpk"], df["p98_minus_p2_pct"], "o-", color="tab:blue", label="p98-p2")
    ax.plot(df["voltage_vpk"], 2.0 * df["lockin_amp_pct_mean"], "s--", color="tab:orange", label="2 x lock-in amp")
    ax.set_xlabel("Input amplitude (VpK)")
    ax.set_ylabel("Intensity variation (% of local mean)")
    ax.set_title("APD rod intensity modulation vs input amplitude")
    ax.legend(frameon=False)
    fig.savefig(out / "intensity_variation_vs_vpk.png", dpi=180)
    plt.close(fig)


def _plot_frequency_identification(out: Path, y: np.ndarray, fs: float, freq_hz: float, f_min: float, f_max: float) -> None:
    yy = signal.detrend(np.asarray(y, dtype=np.float64), type="linear")
    nperseg = min(int(len(yy)), 262144)
    freqs, pxx = signal.welch(
        yy,
        fs=fs,
        nperseg=nperseg,
        noverlap=nperseg // 2,
        detrend="constant",
        scaling="density",
    )
    band = (freqs >= float(f_min)) & (freqs <= float(f_max))
    fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    ax.semilogy(freqs[band], pxx[band], color="black", lw=1.0)
    ax.axvline(freq_hz, color="tab:red", lw=1.2, ls="--", label=f"peak {freq_hz:.2f} Hz")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("PSD")
    ax.set_title("APD total-intensity spectrum near drive response")
    ax.legend(frameon=False)
    fig.savefig(out / "frequency_identification_700_1100hz.png", dpi=180)
    plt.close(fig)


def _plot_one_period_tower(out: Path, traces: list[dict], freq_hz: float, signal_label: str) -> None:
    n = len(traces)
    ncols = 2
    nrows = int(math.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(11, max(2.2 * nrows, 4.0)), constrained_layout=False)
    axes_arr = np.asarray(axes).reshape(-1)
    for ax, item in zip(axes_arr, traces):
        tt_ms = np.asarray(item["t"], dtype=np.float64) * 1000.0
        yy = np.asarray(item["pct"], dtype=np.float64)
        ax.plot(tt_ms, yy, color="black", lw=1.0)
        ax.axhline(0.0, color="0.75", lw=0.7)
        ax.set_title(f"{_fmt_voltage(item['voltage_vpk'])} VpK, p98-p2={item['p98_minus_p2_pct']:.1f}%", fontsize=9)
        ax.set_xlabel("Time within one drive period (ms)")
        ax.set_ylabel("Deviation from mean (%)")
    for ax in axes_arr[len(traces) :]:
        ax.axis("off")
    fig.suptitle(f"One-period APD waveforms ({signal_label}), response {freq_hz:.2f} Hz", y=0.99)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.965))
    fig.savefig(out / "one_period_waveforms_by_vpk_2col.png", dpi=180)
    plt.close(fig)


def analyse(
    path: Path,
    out_dir: Path | None = None,
    f_min: float = 700.0,
    f_max: float = 1100.0,
    region_start_s: float | None = None,
    region_end_s: float | None = None,
    channel_mode: str = "total",
    step_duration_s: float | None = None,
    voltage_step: float = 1.0,
) -> Path:
    path = Path(path)
    voltages = _make_voltage_labels(path, voltage_step=voltage_step)
    out = Path(out_dir) if out_dir is not None else path.with_suffix("").parent / f"{path.stem}_apd_step_waveform_analysis"
    out.mkdir(parents=True, exist_ok=True)

    channels, fs, names = _read_tdms_channels(path)
    intensity, signal_label, selected_channel_idx = _select_analysis_signal(channels, names, channel_mode)
    freq_hz = _dominant_frequency(intensity, fs, f_min, f_max)
    t_env, amp, mean = _lockin_envelope(intensity, fs, freq_hz)
    amp_pct = 100.0 * amp / np.maximum(np.abs(mean), 1e-12)

    n_steps = len(voltages)
    if step_duration_s is not None and float(step_duration_s) > 0.0:
        auto_start = 0.0
        auto_end = float(n_steps) * float(step_duration_s)
    else:
        auto_start, auto_end = _auto_step_region(t_env, amp_pct, n_steps=n_steps)
    reg_start = float(region_start_s) if region_start_s is not None else auto_start
    reg_end = float(region_end_s) if region_end_s is not None else auto_end
    reg_start = max(float(t_env[0]), reg_start)
    reg_end = min(float(t_env[-1]), reg_end)
    if reg_end <= reg_start:
        raise RuntimeError(f"Invalid step analysis region {reg_start} to {reg_end} s")
    if step_duration_s is not None and float(step_duration_s) > 0.0:
        segments = _fixed_duration_segments(t_env, n_steps=n_steps, step_duration_s=float(step_duration_s), start_s=reg_start)
    else:
        region_mask = (t_env >= reg_start) & (t_env <= reg_end)
        t_seg_env = t_env[region_mask]
        amp_seg_pct = amp_pct[region_mask]
        segments_local = _segment_monotonic_steps(t_seg_env, amp_seg_pct, n_steps=n_steps, min_step_s=1.0)
        idx0 = int(np.flatnonzero(region_mask)[0])
        segments = [(a + idx0, b + idx0) for a, b in segments_local]

    rows = []
    traces = []
    for (ia, ib), voltage in zip(segments, voltages):
        t0 = float(t_env[ia])
        t1 = float(t_env[max(ia, ib - 1)])
        mask = (t_env >= t0) & (t_env <= t1)
        lock_mean = float(np.nanmean(amp_pct[mask])) if np.any(mask) else float("nan")
        tt, seg, slice_start = _best_one_period_slice(intensity, fs, freq_hz, t0, t1)
        local_mean = float(np.mean(seg))
        pct = 100.0 * (seg - local_mean) / max(abs(local_mean), 1e-12)
        span_pct = float(np.percentile(pct, 98) - np.percentile(pct, 2))
        n_peaks, n_troughs = _count_large_cycles(pct)
        rows.append(
            {
                "voltage_vpk": voltage,
                "segment_start_s": t0,
                "segment_end_s": t1,
                "slice_start_s": slice_start,
                "slice_duration_s": float(len(seg)) / float(fs),
                "lockin_amp_pct_mean": lock_mean,
                "p98_minus_p2_pct": span_pct,
                "mean_intensity_counts_or_volts": local_mean,
                "n_peaks_one_period": n_peaks,
                "n_troughs_one_period": n_troughs,
            }
        )
        traces.append(
            {
                "voltage_vpk": voltage,
                "t": tt,
                "pct": pct,
                "p98_minus_p2_pct": span_pct,
                "n_peaks": n_peaks,
                "n_troughs": n_troughs,
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(out / "step_waveform_metrics.csv", index=False)
    pd.DataFrame(
        {
            "channel_index": list(range(len(names))),
            "channel_name": names,
            "mean_value": [float(v) for v in np.mean(channels, axis=1)],
            "selected_for_analysis": [i == selected_channel_idx for i in range(len(names))],
        }
    ).to_csv(out / "tdms_channels_used.csv", index=False)
    pd.DataFrame({"time_s": t_env, "lockin_amp_pct": amp_pct, "local_mean": mean}).to_csv(
        out / "lockin_envelope_vs_time.csv", index=False
    )
    _plot_frequency_identification(out, intensity, fs, freq_hz, f_min, f_max)
    _plot_step_detection(out, t_env, amp_pct, segments, voltages, freq_hz, (reg_start, reg_end), signal_label)
    _plot_amplitude_vs_voltage(out, df)
    _plot_one_period_tower(out, traces, freq_hz, signal_label)

    summary = [
        f"TDMS: {path}",
        f"Channels used: {signal_label}",
        f"Sampling rate: {fs:.3f} Hz",
        f"Dominant drive/intensity component searched in {f_min:.1f}-{f_max:.1f} Hz: {freq_hz:.6f} Hz",
        f"Voltage labels assigned in recording order: {voltages}",
        f"Analysed step region: {reg_start:.3f} to {reg_end:.3f} s",
        (
            f"Step regions were fixed at {float(step_duration_s):.3f} s per voltage step."
            if step_duration_s is not None and float(step_duration_s) > 0.0
            else "Step regions were found by piecewise-constant segmentation of the lock-in amplitude at the dominant frequency."
        ),
        "One-period traces are raw single-period slices from the strongest stable part of each detected step, shown as percent deviation from that slice mean.",
        "Automatic peak-counting is not used for the final two-period call because low-amplitude noise creates false peaks.",
        f"Output folder: {out}",
    ]
    (out / "analysis_summary.txt").write_text("\n".join(summary) + "\n", encoding="utf-8")
    print("\n".join(summary))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyse stepped APD vibration amplitude TDMS waveforms.")
    parser.add_argument("--tdms", type=Path, default=DEFAULT_TDMS)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--f-min", type=float, default=700.0)
    parser.add_argument("--f-max", type=float, default=1100.0)
    parser.add_argument("--region-start-s", type=float, default=None)
    parser.add_argument("--region-end-s", type=float, default=None)
    parser.add_argument("--channel-mode", type=str, default="total", help="'total', 'dimmest', or a channel such as ai0")
    parser.add_argument("--step-duration-s", type=float, default=None, help="Use fixed-duration voltage steps instead of automatic segmentation.")
    parser.add_argument("--voltage-step", type=float, default=1.0, help="Voltage increment between labelled steps, e.g. 0.5 or 2 VpK.")
    args = parser.parse_args()
    analyse(
        args.tdms,
        out_dir=args.out_dir,
        f_min=float(args.f_min),
        f_max=float(args.f_max),
        region_start_s=args.region_start_s,
        region_end_s=args.region_end_s,
        channel_mode=args.channel_mode,
        step_duration_s=args.step_duration_s,
        voltage_step=float(args.voltage_step),
    )


if __name__ == "__main__":
    main()
