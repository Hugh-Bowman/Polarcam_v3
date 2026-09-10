from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
BACKGROUND_DIR = ROOT / "datasets" / "background characterisation" / "16072026 backgrounds"
OUT_DIR = BACKGROUND_DIR / "average frame percentage distributions no extra subtraction"
SUMMARY_JSON = ROOT / "outputs" / "40nm_vs_25nm_std_comparison_refit_new_curves" / "summary.json"
REFIT_DIR = ROOT / "outputs" / "40nm_vs_25nm_std_comparison_refit_new_curves"

ANGLE_WINDOW_RAW = 14
AVERAGE_FILES = [
    BACKGROUND_DIR / "frame_stack_20260716-140000 average frame.npy",
    BACKGROUND_DIR / "frame_stack_20260716-140047 average frame.npy",
    BACKGROUND_DIR / "frame_stack_20260716-140137 average frame.npy",
]
MAX_SAMPLES_PER_IMAGE = 500_000
R_STEP = 0.01
PERCENT_BIN_WIDTH = 0.1


def box_mean_14x14(image: np.ndarray) -> np.ndarray:
    return cv2.blur(
        np.asarray(image, dtype=np.float32),
        (ANGLE_WINDOW_RAW, ANGLE_WINDOW_RAW),
        borderType=cv2.BORDER_REFLECT,
    )


def sample_values(values: np.ndarray, max_count: int, rng: np.random.Generator) -> np.ndarray:
    flat = np.asarray(values).ravel()
    if flat.size <= max_count:
        return flat.astype(np.float32, copy=True)
    idx = rng.choice(flat.size, size=max_count, replace=False)
    return flat[idx].astype(np.float32, copy=False)


def describe(values: np.ndarray) -> dict[str, float]:
    v = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(v)),
        "std": float(np.std(v)),
        "p01": float(np.percentile(v, 1)),
        "p05": float(np.percentile(v, 5)),
        "p50": float(np.percentile(v, 50)),
        "p95": float(np.percentile(v, 95)),
        "p99": float(np.percentile(v, 99)),
        "zero_fraction": float(np.mean(v <= 0)),
    }


def strip_phase_marker(arr: np.ndarray, roi_meta: dict[str, Any] | None = None) -> np.ndarray:
    a = np.asarray(arr)
    if a.ndim < 3 or int(a.shape[0]) < 2:
        return a
    marker = np.asarray(a[-1])
    if marker.ndim != 2:
        return a
    nz = np.argwhere(marker != 0)
    if nz.shape[0] != 1:
        return a
    my, mx = int(nz[0][0]), int(nz[0][1])
    if float(marker[my, mx]) != 1.0:
        return a
    if float(np.sum(marker, dtype=np.float64)) != 1.0:
        return a
    if roi_meta is not None:
        roi_meta["phase_x"] = int(mx) % 2
        roi_meta["phase_y"] = int(my) % 2
    return np.asarray(a[:-1])


def angle_window_slices(frame_shape: tuple[int, int], roi_meta: dict[str, Any]) -> tuple[slice, slice]:
    gh, gw = int(frame_shape[0]), int(frame_shape[1])
    win_raw = int(roi_meta.get("win_raw", int(roi_meta.get("w", ANGLE_WINDOW_RAW))))
    win_raw = max(2, win_raw)
    if win_raw % 2:
        win_raw -= 1
    try:
        cx = float(roi_meta["cx"]) - float(roi_meta["x"])
        cy = float(roi_meta["cy"]) - float(roi_meta["y"])
    except Exception:
        cx = (gw - 1) / 2.0
        cy = (gh - 1) / 2.0
    x0 = max(0, int(round(cx)) - (win_raw // 2))
    y0 = max(0, int(round(cy)) - (win_raw // 2))
    x1 = min(gw, x0 + win_raw)
    y1 = min(gh, y0 + win_raw)
    return slice(y0, y1), slice(x0, x1)


def rod_angle_window_mean(rod_dir: Path) -> float:
    meta_path = rod_dir / "capture_maxfps_15x15_meta.json"
    npy_path = rod_dir / "capture_maxfps_15x15.npy"
    if not meta_path.exists() or not npy_path.exists():
        return float("nan")
    meta = json.loads(meta_path.read_text())
    roi_meta = dict(meta.get("actual", {}).get("roi", {}))
    arr = np.load(npy_path, mmap_mode="r")
    arr_use = strip_phase_marker(arr, roi_meta=roi_meta)
    ys, xs = angle_window_slices(tuple(arr_use.shape[1:]), roi_meta)
    win = np.asarray(arr_use[:, ys, xs], dtype=np.float64)
    return float(np.mean(win)) if win.size else float("nan")


def load_stationary_intensity_models() -> dict[str, dict[str, Any]]:
    summary = json.loads(SUMMARY_JSON.read_text())
    out: dict[str, dict[str, Any]] = {}
    for size, info in summary["datasets"].items():
        good_dir = Path(info["good_dir"])
        rows = list(csv.DictReader((REFIT_DIR / f"{size}_refitted_points.csv").open("r", encoding="utf-8", newline="")))
        records = []
        for row in rows:
            intensity = rod_angle_window_mean(good_dir / row["rod"])
            r_mean = float(row["r_mean"])
            theta_deg = float(row["theta_deg"])
            if np.isfinite(r_mean) and np.isfinite(theta_deg) and np.isfinite(intensity) and intensity > 0:
                records.append((r_mean, theta_deg, intensity, row["rod"]))
        records.sort(key=lambda x: x[0])
        r = np.asarray([x[0] for x in records], dtype=np.float64)
        theta = np.asarray([x[1] for x in records], dtype=np.float64)
        intensity = np.asarray([x[2] for x in records], dtype=np.float64)
        design = np.column_stack([np.sin(np.radians(theta)) ** 2, np.cos(np.radians(theta)) ** 2])
        coeffs, *_ = np.linalg.lstsq(design, intensity, rcond=None)
        out[size] = {
            "r_measured": r,
            "theta_measured_deg": theta,
            "intensity_measured": intensity,
            "mean_intensity": float(np.mean(intensity)),
            "a_sin2": float(coeffs[0]),
            "b_cos2": float(coeffs[1]),
            "curve_csv": str(info["curve_csv"]),
            "stretch_scale": float(info.get("stretch_scale", 1.0)),
        }
    return out


def fitted_intensity_grid(data: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    curve = np.genfromtxt(str(data["curve_csv"]), delimiter=",", names=True)
    r_curve = np.asarray(curve["r"], dtype=np.float64) * float(data["stretch_scale"])
    theta_curve = np.asarray(curve["theta_deg_center"], dtype=np.float64)
    order = np.argsort(r_curve)
    r_curve = r_curve[order]
    theta_curve = theta_curve[order]
    r_grid = np.round(
        np.arange(
            float(np.ceil(np.min(r_curve) / R_STEP) * R_STEP),
            float(np.floor(np.max(r_curve) / R_STEP) * R_STEP) + 0.5 * R_STEP,
            R_STEP,
        ),
        4,
    )
    theta_grid = np.interp(r_grid, r_curve, theta_curve)
    theta_rad = np.radians(theta_grid)
    intensity = float(data["a_sin2"]) * np.sin(theta_rad) ** 2 + float(data["b_cos2"]) * np.cos(theta_rad) ** 2
    return r_grid, theta_grid, intensity


def load_average_frame_samples(rng: np.random.Generator) -> list[dict[str, Any]]:
    records = []
    for path in AVERAGE_FILES:
        image = np.asarray(np.load(path, mmap_mode="r"), dtype=np.float32)
        pixel_samples = sample_values(image, MAX_SAMPLES_PER_IMAGE, rng)
        window_samples = sample_values(box_mean_14x14(image), MAX_SAMPLES_PER_IMAGE, rng)
        records.append(
            {
                "recording": path.name,
                "shape": tuple(int(x) for x in image.shape),
                "pixel_samples": pixel_samples,
                "window_samples": window_samples,
                "pixel_stats": describe(pixel_samples),
                "window_stats": describe(window_samples),
            }
        )
    return records


def write_basic_summary(records: list[dict[str, Any]], rod_data: dict[str, dict[str, Any]]) -> None:
    with (OUT_DIR / "average_frame_background_summary.csv").open("w", newline="") as f:
        fields = [
            "recording",
            "pixel_mean",
            "pixel_std",
            "pixel_p50",
            "pixel_p99",
            "window14_mean",
            "window14_std",
            "window14_p50",
            "window14_p99",
            "mean_percent_40x65nm",
            "mean_percent_25x65nm",
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for record in records:
            row = {
                "recording": record["recording"],
                "pixel_mean": record["pixel_stats"]["mean"],
                "pixel_std": record["pixel_stats"]["std"],
                "pixel_p50": record["pixel_stats"]["p50"],
                "pixel_p99": record["pixel_stats"]["p99"],
                "window14_mean": record["window_stats"]["mean"],
                "window14_std": record["window_stats"]["std"],
                "window14_p50": record["window_stats"]["p50"],
                "window14_p99": record["window_stats"]["p99"],
                "mean_percent_40x65nm": 100.0 * record["window_stats"]["mean"] / float(rod_data["40x65nm"]["mean_intensity"]),
                "mean_percent_25x65nm": 100.0 * record["window_stats"]["mean"] / float(rod_data["25x65nm"]["mean_intensity"]),
            }
            writer.writerow(row)


def write_mean_percentage_plots(records: list[dict[str, Any]], rod_data: dict[str, dict[str, Any]]) -> None:
    for size, data in rod_data.items():
        denom = float(data["mean_intensity"])
        all_pct = [100.0 * record["window_samples"] / denom for record in records]
        upper = max(float(np.percentile(vals, 99.5)) for vals in all_pct)
        bins = np.arange(0.0, max(upper + PERCENT_BIN_WIDTH, 1.0), PERCENT_BIN_WIDTH)
        plt.figure(figsize=(9, 5.5))
        for record, vals in zip(records, all_pct):
            label = record["recording"].replace("frame_stack_", "").replace(" average frame.npy", "")
            plt.hist(vals, bins=bins, density=True, histtype="step", linewidth=1.7, label=label)
        plt.title(f"Stored average background as percentage of mean {size} rod signal")
        plt.xlabel("Background / mean rod signal (%)")
        plt.ylabel("Probability density")
        plt.yscale("log")
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(OUT_DIR / f"average_frame_percentage_distribution_mean_intensity_{size}.png", dpi=240)
        plt.close()


def write_density_vs_r_tables(records: list[dict[str, Any]], rod_data: dict[str, dict[str, Any]]) -> None:
    aggregate = np.concatenate([record["window_samples"] for record in records])
    for size, data in rod_data.items():
        r_grid, theta_grid, intensity_grid = fitted_intensity_grid(data)
        max_pct = float(np.percentile(100.0 * aggregate / np.min(intensity_grid), 99.9))
        bins = np.arange(0.0, max(max_pct * 1.05, 1.0) + PERCENT_BIN_WIDTH, PERCENT_BIN_WIDTH)
        centers = 0.5 * (bins[:-1] + bins[1:])
        density_cols = []
        for intensity in intensity_grid:
            pct = 100.0 * aggregate / intensity
            hist, _ = np.histogram(pct, bins=bins, density=True)
            density_cols.append(hist)
        density = np.asarray(density_cols).T
        with (OUT_DIR / f"average_frame_percentage_density_vs_r_{size}.csv").open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["percentage_bin_center"] + [f"r={r:.4f}" for r in r_grid])
            for center, row in zip(centers, density):
                writer.writerow([center] + list(row))
        with (OUT_DIR / f"average_frame_intensity_fitted_vs_r_{size}.csv").open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["r", "theta_deg_from_curve", "fitted_angle_window_intensity", "a_sin2", "b_cos2"])
            for r, theta, intensity in zip(r_grid, theta_grid, intensity_grid):
                writer.writerow([r, theta, intensity, data["a_sin2"], data["b_cos2"]])
        fig, ax = plt.subplots(figsize=(9, 5.5))
        mesh = ax.pcolormesh(r_grid, centers, density, shading="auto", cmap="viridis")
        ax.set_title(f"Stored average-frame background percentage density vs r ({size})")
        ax.set_xlabel("Measured r")
        ax.set_ylabel("Background / rod signal (%)")
        fig.colorbar(mesh, ax=ax, label="Probability density")
        fig.tight_layout()
        fig.savefig(OUT_DIR / f"average_frame_percentage_density_vs_r_{size}.png", dpi=240)
        plt.close(fig)


def write_readme(records: list[dict[str, Any]], rod_data: dict[str, dict[str, Any]]) -> None:
    lines = [
        "Stored average-frame background statistics, no extra subtraction",
        "",
        f"Input folder: {BACKGROUND_DIR}",
        f"Output folder: {OUT_DIR}",
        "",
        "Method:",
        "- Used only the three stored average-frame npy files, excluding the first/blue recording.",
        "- No additional subtraction was applied.",
        "- Background statistics are computed on the images exactly as stored.",
        "- Percentage distributions use 14x14 window means to match the angle-calculation window.",
        "- r-dependent percentage tables use the fitted I(theta)=a sin^2(theta)+b cos^2(theta) stationary-rod intensity model.",
        "",
        "Mean rod intensities:",
    ]
    for size, data in rod_data.items():
        lines.append(
            f"- {size}: mean={float(data['mean_intensity']):.6g}; "
            f"a_sin2={float(data['a_sin2']):.6g}; b_cos2={float(data['b_cos2']):.6g}"
        )
    lines.extend(["", "Average-frame summaries:"])
    for record in records:
        lines.append(
            f"- {record['recording']}: pixel mean {record['pixel_stats']['mean']:.4g}, "
            f"14x14 mean {record['window_stats']['mean']:.4g}; "
            f"40x65nm {100.0 * record['window_stats']['mean'] / float(rod_data['40x65nm']['mean_intensity']):.3g}%; "
            f"25x65nm {100.0 * record['window_stats']['mean'] / float(rod_data['25x65nm']['mean_intensity']):.3g}%."
        )
    lines.extend(
        [
            "",
            "Main outputs:",
            "- average_frame_background_summary.csv",
            "- average_frame_percentage_distribution_mean_intensity_40x65nm.png",
            "- average_frame_percentage_distribution_mean_intensity_25x65nm.png",
            "- average_frame_percentage_density_vs_r_40x65nm.csv/png",
            "- average_frame_percentage_density_vs_r_25x65nm.csv/png",
        ]
    )
    (OUT_DIR / "README_average_frame_no_extra_subtraction.txt").write_text("\n".join(lines) + "\n")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(1607202604)
    records = load_average_frame_samples(rng)
    rod_data = load_stationary_intensity_models()
    write_basic_summary(records, rod_data)
    write_mean_percentage_plots(records, rod_data)
    write_density_vs_r_tables(records, rod_data)
    write_readme(records, rod_data)
    print(f"Wrote outputs to {OUT_DIR}")


if __name__ == "__main__":
    main()
