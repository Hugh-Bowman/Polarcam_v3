from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import matplotlib
import numpy as np
from scipy.signal import butter, sosfiltfilt

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_RECORDING_DIR = Path(
    r"C:\Polarcam Software\Polarcam_v3\polarcam_live\datasets\background characterisation\z scan of a rod\pending\pending\rod_x1617_y470_20260720-182320_1784568200062348400"
)
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


def load_meta(recording_dir: Path) -> dict:
    for name in ("capture_maxfps_15x15_meta.json", "meta.json"):
        p = recording_dir / name
        if p.exists():
            return json.loads(p.read_text())
    m = re.search(r"x(\d+)_y(\d+)", recording_dir.name)
    x = int(m.group(1)) if m else 0
    y = int(m.group(2)) if m else 0
    return {
        "actual": {
            "fps": DEFAULT_FPS,
            "roi": {
                "x": x - 7,
                "y": y - 7,
                "cx": x,
                "cy": y,
                "win_raw": 14,
                "phase_x": x % 2,
                "phase_y": y % 2,
            },
        }
    }


def roi_from_meta(meta: dict) -> tuple[float, dict]:
    if "actual" in meta:
        actual = meta.get("actual", {})
        fps = float(actual.get("fps") or DEFAULT_FPS)
        roi = actual.get("roi", {})
        return fps, roi
    modes = meta.get("modes", {})
    summary = modes.get("capture_maxfps_15x15_summary", {})
    fps = float(summary.get("actual_fps") or DEFAULT_FPS)
    center = meta.get("center_px", {})
    cx = float(center.get("x", 0.0))
    cy = float(center.get("y", 0.0))
    return fps, {
        "x": int(round(cx)) - 7,
        "y": int(round(cy)) - 7,
        "cx": cx,
        "cy": cy,
        "win_raw": 14,
        "phase_x": int(round(cx)) % 2,
        "phase_y": int(round(cy)) % 2,
    }


def xy_trace(stack: np.ndarray, roi: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    px = int(roi.get("phase_x", 0)) % 2
    py = int(roi.get("phase_y", 0)) % 2
    win_raw = int(roi.get("win_raw", 14))
    win = max(1, int(round(win_raw / 2.0)))
    if win % 2 == 0:
        win += 1
    half = win // 2
    cx_i = int(round((float(roi.get("cx", 0.0)) - float(roi.get("x", 0.0))) / 2.0))
    cy_i = int(round((float(roi.get("cy", 0.0)) - float(roi.get("y", 0.0))) / 2.0))

    xs = np.empty(stack.shape[0], dtype=np.float64)
    ys = np.empty(stack.shape[0], dtype=np.float64)
    intensity = np.empty(stack.shape[0], dtype=np.float64)
    for i, g in enumerate(stack):
        frame = np.asarray(g[:, :14], dtype=np.float64)
        I0 = frame[py::2, px::2]
        I45 = frame[py::2, (1 - px) :: 2]
        I135 = frame[(1 - py) :: 2, px::2]
        I90 = frame[(1 - py) :: 2, (1 - px) :: 2]

        ih, iw = I0.shape
        x0 = max(0, cx_i - half)
        x1 = min(iw, cx_i + half + 1)
        y0 = max(0, cy_i - half)
        y1 = min(ih, cy_i + half + 1)
        a0 = I0[y0:y1, x0:x1]
        a45 = I45[y0:y1, x0:x1]
        a135 = I135[y0:y1, x0:x1]
        a90 = I90[y0:y1, x0:x1]

        s0 = float(np.sum(a0))
        s45 = float(np.sum(a45))
        s135 = float(np.sum(a135))
        s90 = float(np.sum(a90))
        xs[i] = (s0 - s90) / (s0 + s90 + 1e-12)
        ys[i] = (s45 - s135) / (s45 + s135 + 1e-12)
        intensity[i] = s0 + s45 + s135 + s90
    return xs, ys, intensity


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--recording-dir", type=Path, default=DEFAULT_RECORDING_DIR)
    parser.add_argument("--half-window-s", type=float, default=0.25)
    args = parser.parse_args()

    recording_dir = args.recording_dir
    out_dir = recording_dir / "npy_frequency_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = load_meta(recording_dir)
    fps, roi = roi_from_meta(meta)
    stack = strip_marker(np.load(recording_dir / "capture_maxfps_15x15.npy", mmap_mode="r"))
    x, y, intensity = xy_trace(stack, roi)
    t = np.arange(x.size, dtype=np.float64) / fps

    high = min(0.98 * fps / 2.0, 800.0)
    sos = butter(4, [2.0, high], btype="bandpass", fs=fps, output="sos")
    bp = sosfiltfilt(sos, intensity - np.mean(intensity))
    env_win = max(3, int(round(0.020 * fps)))
    n_blocks = len(bp) // env_win
    env = np.sqrt(np.mean(bp[: n_blocks * env_win].reshape(n_blocks, env_win) ** 2, axis=1))
    env_t = (np.arange(n_blocks) + 0.5) * env_win / fps
    event_i = int(np.argmax(env))
    event_t = float(env_t[event_i])
    start_t = max(0.0, event_t - float(args.half_window_s))
    end_t = min(float(t[-1]), event_t + float(args.half_window_s))
    keep = (t >= start_t) & (t <= end_t)

    with (out_dir / "xy_around_largest_vibration.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "time_s", "X", "Y", "r", "summed_intensity", "selected_for_tap_window"])
        r = np.sqrt(x * x + y * y)
        for i in range(x.size):
            writer.writerow([i, t[i], x[i], y[i], r[i], intensity[i], bool(keep[i])])

    fig, ax = plt.subplots(figsize=(6.4, 6.0))
    sc = ax.scatter(x[keep], y[keep], c=t[keep] - event_t, s=10, cmap="viridis", alpha=0.85, linewidths=0)
    ax.scatter([x[keep][0]], [y[keep][0]], marker="o", s=60, facecolors="none", edgecolors="white", linewidths=1.2, label="window start")
    ax.scatter([x[keep][-1]], [y[keep][-1]], marker="s", s=55, facecolors="none", edgecolors="red", linewidths=1.2, label="window end")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("X = (I0 - I90) / (I0 + I90)")
    ax.set_ylabel("Y = (I45 - I135) / (I45 + I135)")
    ax.set_title(f"XY around largest vibration tap, {start_t:.3f}-{end_t:.3f} s")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    cb = fig.colorbar(sc, ax=ax)
    cb.set_label("time relative to tap centre (s)")
    fig.tight_layout()
    fig.savefig(out_dir / "xy_around_largest_vibration_tap.png", dpi=240)
    plt.close(fig)

    fig, axes = plt.subplots(3, 1, figsize=(11, 7.5), sharex=True)
    axes[0].plot(t, intensity, lw=0.6)
    axes[0].axvspan(start_t, end_t, color="tab:red", alpha=0.15, label="XY window")
    axes[0].set_ylabel("summed intensity")
    axes[0].legend(fontsize=8)
    axes[1].plot(t, x, lw=0.7, label="X")
    axes[1].plot(t, y, lw=0.7, label="Y")
    axes[1].axvspan(start_t, end_t, color="tab:red", alpha=0.15)
    axes[1].set_ylabel("XY")
    axes[1].legend(fontsize=8)
    axes[2].plot(env_t, env, lw=0.9, color="black")
    axes[2].scatter([event_t], [env[event_i]], color="tab:red", zorder=5, label="largest RMS burst")
    axes[2].axvspan(start_t, end_t, color="tab:red", alpha=0.15)
    axes[2].set_xlabel("time (s)")
    axes[2].set_ylabel("20 ms RMS")
    axes[2].legend(fontsize=8)
    fig.suptitle("Largest vibration event used for XY zoom")
    fig.tight_layout()
    fig.savefig(out_dir / "largest_vibration_window_timeseries.png", dpi=240)
    plt.close(fig)

    summary = {
        "recording_dir": str(recording_dir),
        "fps": fps,
        "event_time_s": event_t,
        "window_start_s": start_t,
        "window_end_s": end_t,
        "n_points_in_window": int(np.sum(keep)),
        "phase_x": int(roi.get("phase_x", 0)) % 2,
        "phase_y": int(roi.get("phase_y", 0)) % 2,
        "output_dir": str(out_dir),
    }
    (out_dir / "xy_around_largest_vibration_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
