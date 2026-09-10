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
OUT_DIR = BACKGROUND_DIR / "labelled percentage heatmaps no extra subtraction"
SUMMARY_JSON = ROOT / "outputs" / "40nm_vs_25nm_std_comparison_refit_new_curves" / "summary.json"
REFIT_DIR = ROOT / "outputs" / "40nm_vs_25nm_std_comparison_refit_new_curves"

ANGLE_WINDOW_RAW = 14
MAX_SAMPLES_PER_IMAGE = 800_000
R_STEP = 0.01
PERCENT_BIN_WIDTH = 0.1

FILES = [
    {
        "path": BACKGROUND_DIR / "frame_stack_20260716-140000 average frame.npy",
        "condition": "uncorrected_background",
        "label": "Uncorrected background",
    },
    {
        "path": BACKGROUND_DIR / "frame_stack_20260716-140137 average frame.npy",
        "condition": "different_coverslip_background_subtracted",
        "label": "Background subtracted from different coverslip",
    },
    {
        "path": BACKGROUND_DIR / "frame_stack_20260716-140047 average frame.npy",
        "condition": "same_point_same_coverslip_best_case_background_subtracted",
        "label": "Best case: same point/same coverslip background subtracted",
    },
]


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
    meta = json.loads(meta_path.read_text())
    roi_meta = dict(meta.get("actual", {}).get("roi", {}))
    arr = np.load(npy_path, mmap_mode="r")
    arr_use = strip_phase_marker(arr, roi_meta=roi_meta)
    ys, xs = angle_window_slices(tuple(arr_use.shape[1:]), roi_meta)
    return float(np.mean(np.asarray(arr_use[:, ys, xs], dtype=np.float64)))


def load_intensity_models() -> dict[str, dict[str, Any]]:
    summary = json.loads(SUMMARY_JSON.read_text())
    out: dict[str, dict[str, Any]] = {}
    for size, info in summary["datasets"].items():
        good_dir = Path(info["good_dir"])
        rows = list(csv.DictReader((REFIT_DIR / f"{size}_refitted_points.csv").open("r", encoding="utf-8", newline="")))
        theta_vals = []
        intensity_vals = []
        for row in rows:
            inten = rod_angle_window_mean(good_dir / row["rod"])
            theta = float(row["theta_deg"])
            if np.isfinite(inten) and inten > 0 and np.isfinite(theta):
                theta_vals.append(theta)
                intensity_vals.append(inten)
        theta_rad = np.radians(np.asarray(theta_vals, dtype=np.float64))
        intensity = np.asarray(intensity_vals, dtype=np.float64)
        design = np.column_stack([np.sin(theta_rad) ** 2, np.cos(theta_rad) ** 2])
        coeffs, *_ = np.linalg.lstsq(design, intensity, rcond=None)
        out[size] = {
            "a_sin2": float(coeffs[0]),
            "b_cos2": float(coeffs[1]),
            "curve_csv": str(info["curve_csv"]),
            "stretch_scale": float(info.get("stretch_scale", 1.0)),
        }
    return out


def fitted_intensity_grid(model: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    curve = np.genfromtxt(str(model["curve_csv"]), delimiter=",", names=True)
    r_curve = np.asarray(curve["r"], dtype=np.float64) * float(model["stretch_scale"])
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
    intensity_grid = (
        float(model["a_sin2"]) * np.sin(theta_rad) ** 2
        + float(model["b_cos2"]) * np.cos(theta_rad) ** 2
    )
    return r_grid, theta_grid, intensity_grid


def make_density_table(window_samples: np.ndarray, intensity_grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    max_pct = float(np.percentile(100.0 * window_samples / np.min(intensity_grid), 99.9))
    bins = np.arange(0.0, max(max_pct * 1.05, 1.0) + PERCENT_BIN_WIDTH, PERCENT_BIN_WIDTH)
    centers = 0.5 * (bins[:-1] + bins[1:])
    cols = []
    for intensity in intensity_grid:
        pct = 100.0 * window_samples / intensity
        hist, _ = np.histogram(pct, bins=bins, density=True)
        cols.append(hist)
    return centers, np.asarray(cols).T


def write_heatmap_csv(path: Path, percentage_centers: np.ndarray, r_grid: np.ndarray, density: np.ndarray) -> None:
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["percentage_bin_center"] + [f"r={r:.4f}" for r in r_grid])
        for center, row in zip(percentage_centers, density):
            writer.writerow([center] + list(row))


def plot_heatmap(path: Path, title: str, r_grid: np.ndarray, pct_centers: np.ndarray, density: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(9, 5.5))
    mesh = ax.pcolormesh(r_grid, pct_centers, density, shading="auto", cmap="viridis")
    ax.set_title(title)
    ax.set_xlabel("Measured r")
    ax.set_ylabel("Background / rod signal (%)")
    fig.colorbar(mesh, ax=ax, label="Probability density")
    fig.tight_layout()
    fig.savefig(path, dpi=240)
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(1607202605)
    models = load_intensity_models()
    summary_rows = []
    for item in FILES:
        image = np.asarray(np.load(item["path"], mmap_mode="r"), dtype=np.float32)
        window_samples = sample_values(box_mean_14x14(image), MAX_SAMPLES_PER_IMAGE, rng)
        window_mean = float(np.mean(window_samples))
        window_std = float(np.std(window_samples))
        for size, model in models.items():
            r_grid, theta_grid, intensity_grid = fitted_intensity_grid(model)
            pct_centers, density = make_density_table(window_samples, intensity_grid)
            stem = f"{item['condition']}_{size}_percentage_heatmap"
            write_heatmap_csv(OUT_DIR / f"{stem}.csv", pct_centers, r_grid, density)
            plot_heatmap(
                OUT_DIR / f"{stem}.png",
                f"{item['label']} ({size})",
                r_grid,
                pct_centers,
                density,
            )
            summary_rows.append(
                {
                    "condition": item["condition"],
                    "label": item["label"],
                    "source_file": item["path"].name,
                    "rod_size": size,
                    "window14_mean": window_mean,
                    "window14_std": window_std,
                    "a_sin2": model["a_sin2"],
                    "b_cos2": model["b_cos2"],
                    "csv": f"{stem}.csv",
                    "png": f"{stem}.png",
                }
            )
    with (OUT_DIR / "heatmap_file_summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)
    readme = [
        "Labelled average-frame percentage heatmaps, no extra subtraction",
        "",
        f"Input folder: {BACKGROUND_DIR}",
        f"Output folder: {OUT_DIR}",
        "",
        "Conditions are ordered by largest to smallest mean background:",
        "- uncorrected_background: frame_stack_20260716-140000 average frame.npy",
        "- different_coverslip_background_subtracted: frame_stack_20260716-140137 average frame.npy",
        "- same_point_same_coverslip_best_case_background_subtracted: frame_stack_20260716-140047 average frame.npy",
        "",
        "Each rod size has one heatmap CSV and PNG per condition, giving 6 CSVs and 6 PNGs.",
        "CSV format: first column is percentage_bin_center; each following column is an r value; cells are probability density.",
        "No additional subtraction is applied to the stored average-frame npy files.",
    ]
    (OUT_DIR / "README_labelled_heatmaps.txt").write_text("\n".join(readme) + "\n")
    print(f"Wrote outputs to {OUT_DIR}")


if __name__ == "__main__":
    main()
