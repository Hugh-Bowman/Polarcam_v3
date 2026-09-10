from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
import numpy as np

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


def load_meta(recording_dir: Path) -> tuple[float, dict]:
    meta_path = recording_dir / "capture_maxfps_15x15_meta.json"
    if not meta_path.exists():
        return DEFAULT_FPS, {"phase_x": 0, "phase_y": 0, "win_raw": 14, "x": 0, "y": 0, "cx": 7, "cy": 7}
    meta = json.loads(meta_path.read_text())
    fps = float(meta.get("actual", {}).get("fps") or DEFAULT_FPS)
    roi = dict(meta.get("actual", {}).get("roi", {}))
    return fps, roi


def xy_brightness_trace(stack: np.ndarray, roi: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    px = int(roi.get("phase_x", 0)) % 2
    py = int(roi.get("phase_y", 0)) % 2
    win_raw = int(roi.get("win_raw", 14))
    win = max(1, int(round(win_raw / 2.0)))
    if win % 2 == 0:
        win += 1
    half = win // 2
    cx_i = int(round((float(roi.get("cx", 7.0)) - float(roi.get("x", 0.0))) / 2.0))
    cy_i = int(round((float(roi.get("cy", 7.0)) - float(roi.get("y", 0.0))) / 2.0))

    n = int(stack.shape[0])
    x = np.empty(n, dtype=np.float64)
    y = np.empty(n, dtype=np.float64)
    r = np.empty(n, dtype=np.float64)
    brightness = np.empty(n, dtype=np.float64)
    for i, frame0 in enumerate(stack):
        frame = np.asarray(frame0[:, :14], dtype=np.float64)
        I0 = frame[py::2, px::2]
        I45 = frame[py::2, (1 - px) :: 2]
        I135 = frame[(1 - py) :: 2, px::2]
        I90 = frame[(1 - py) :: 2, (1 - px) :: 2]
        ih, iw = I0.shape
        x0 = max(0, cx_i - half)
        x1 = min(iw, cx_i + half + 1)
        y0 = max(0, cy_i - half)
        y1 = min(ih, cy_i + half + 1)
        s0 = float(np.sum(I0[y0:y1, x0:x1]))
        s45 = float(np.sum(I45[y0:y1, x0:x1]))
        s135 = float(np.sum(I135[y0:y1, x0:x1]))
        s90 = float(np.sum(I90[y0:y1, x0:x1]))
        xv = (s0 - s90) / (s0 + s90 + 1e-12)
        yv = (s45 - s135) / (s45 + s135 + 1e-12)
        x[i] = xv
        y[i] = yv
        r[i] = float(np.hypot(xv, yv))
        brightness[i] = s0 + s45 + s135 + s90
    return x, y, r, brightness


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("recording_dir", type=Path)
    args = parser.parse_args()
    rec = args.recording_dir
    out = rec / "sound_frequency_response_analysis"
    out.mkdir(parents=True, exist_ok=True)
    fps, roi = load_meta(rec)
    stack = strip_marker(np.load(rec / "capture_maxfps_15x15.npy", mmap_mode="r"))
    x, y, r, brightness = xy_brightness_trace(stack, roi)
    t = np.arange(x.size, dtype=np.float64) / fps

    with (out / "xy_brightness_trace.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "time_s", "X", "Y", "r", "brightness_sum_11x11_channel_window"])
        for i in range(x.size):
            writer.writerow([i, t[i], x[i], y[i], r[i], brightness[i]])

    fig, ax = plt.subplots(figsize=(6.5, 6.0))
    sc = ax.scatter(x, y, c=t, s=5, cmap="viridis", alpha=0.75, linewidths=0)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("X = (I0 - I90) / (I0 + I90)")
    ax.set_ylabel("Y = (I45 - I135) / (I45 + I135)")
    ax.set_title("XY trace over recording")
    ax.grid(alpha=0.25)
    cb = fig.colorbar(sc, ax=ax)
    cb.set_label("time (s)")
    fig.tight_layout()
    fig.savefig(out / "xy_scatter_coloured_by_time.png", dpi=240)
    plt.close(fig)

    fig, axes = plt.subplots(4, 1, figsize=(12, 9), sharex=True)
    axes[0].plot(t, x, lw=0.7)
    axes[0].set_ylabel("X")
    axes[1].plot(t, y, lw=0.7, color="tab:orange")
    axes[1].set_ylabel("Y")
    axes[2].plot(t, r, lw=0.7, color="tab:green")
    axes[2].set_ylabel("r")
    axes[3].plot(t, brightness, lw=0.7, color="black")
    axes[3].set_ylabel("brightness")
    axes[3].set_xlabel("time (s)")
    fig.suptitle("XY and central channel-window brightness over time")
    for ax in axes:
        ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out / "xy_and_brightness_vs_time.png", dpi=240)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(t, brightness, lw=0.75, color="black")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("summed brightness")
    ax.set_title("Brightness of central channel-window region over time")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out / "brightness_11x11_region_vs_time.png", dpi=240)
    plt.close(fig)

    summary = {
        "recording_dir": str(rec),
        "fps": fps,
        "n_frames": int(x.size),
        "duration_s": float(x.size / fps),
        "x_range": float(np.max(x) - np.min(x)),
        "y_range": float(np.max(y) - np.min(y)),
        "r_mean": float(np.mean(r)),
        "r_std": float(np.std(r)),
        "brightness_mean": float(np.mean(brightness)),
        "brightness_std": float(np.std(brightness)),
        "brightness_range": float(np.max(brightness) - np.min(brightness)),
        "output_dir": str(out),
    }
    (out / "xy_brightness_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
