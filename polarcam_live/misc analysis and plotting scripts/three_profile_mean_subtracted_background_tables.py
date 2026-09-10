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
OUT_DIR = BACKGROUND_DIR / "mean subtraction 3 profiles percentage distributions"
SUMMARY_JSON = ROOT / "outputs" / "40nm_vs_25nm_std_comparison_refit_new_curves" / "summary.json"
REFIT_DIR = ROOT / "outputs" / "40nm_vs_25nm_std_comparison_refit_new_curves"

ANGLE_WINDOW_RAW = 14
STACKS = [
    BACKGROUND_DIR / "frame_stack_20260716-140000.npy",
    BACKGROUND_DIR / "frame_stack_20260716-140047.npy",
    BACKGROUND_DIR / "frame_stack_20260716-140137.npy",
]
MAX_FRAMES_PER_STACK = 25
MAX_SAMPLES_PER_STACK = 400_000
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


def selected_indices(n_frames: int) -> np.ndarray:
    return np.unique(np.linspace(0, n_frames - 1, min(MAX_FRAMES_PER_STACK, n_frames), dtype=int))


def temporal_mean_profile(stack: np.ndarray) -> np.ndarray:
    acc = np.zeros(stack.shape[1:], dtype=np.float64)
    for i in range(int(stack.shape[0])):
        acc += np.asarray(stack[i], dtype=np.float64)
    return (acc / int(stack.shape[0])).astype(np.float32)


def mean_subtracted_window_samples(path: Path, rng: np.random.Generator) -> dict[str, Any]:
    stack = np.load(path, mmap_mode="r")
    if stack.ndim != 3:
        raise ValueError(f"{path} is not a 3D stack: shape={stack.shape}")
    profile = temporal_mean_profile(stack)
    idx = selected_indices(int(stack.shape[0]))
    per_frame = max(1, MAX_SAMPLES_PER_STACK // len(idx))
    samples = []
    raw_samples = []
    for i in idx:
        frame = np.asarray(stack[i], dtype=np.float32)
        raw_samples.append(sample_values(box_mean_14x14(frame), per_frame, rng))
        residual = np.clip(frame - profile, 0.0, None)
        samples.append(sample_values(box_mean_14x14(residual), per_frame, rng))
    residual_windows = np.concatenate(samples)
    raw_windows = np.concatenate(raw_samples)
    return {
        "recording": path.name,
        "shape": tuple(int(x) for x in stack.shape),
        "raw_window_samples": raw_windows,
        "mean_subtracted_window_samples": residual_windows,
        "raw_window_stats": describe(raw_windows),
        "mean_subtracted_window_stats": describe(residual_windows),
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
    x0 = int(round(cx)) - (win_raw // 2)
    y0 = int(round(cy)) - (win_raw // 2)
    x1 = min(gw, x0 + win_raw)
    y1 = min(gh, y0 + win_raw)
    x0 = max(0, x0)
    y0 = max(0, y0)
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
    if arr_use.ndim != 3:
        return float("nan")
    ys, xs = angle_window_slices(tuple(arr_use.shape[1:]), roi_meta)
    win = np.asarray(arr_use[:, ys, xs], dtype=np.float64)
    if win.size == 0:
        return float("nan")
    return float(np.mean(win))


def load_stationary_intensity_models() -> dict[str, dict[str, np.ndarray | float | str]]:
    summary = json.loads(SUMMARY_JSON.read_text())
    out: dict[str, dict[str, np.ndarray | float | str]] = {}
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
        theta_deg = np.asarray([x[1] for x in records], dtype=np.float64)
        intensity = np.asarray([x[2] for x in records], dtype=np.float64)
        theta_rad = np.radians(theta_deg)
        design = np.column_stack([np.sin(theta_rad) ** 2, np.cos(theta_rad) ** 2])
        coeffs, *_ = np.linalg.lstsq(design, intensity, rcond=None)
        mean_intensity = float(np.mean(intensity))
        out[size] = {
            "r_measured": r,
            "theta_measured_deg": theta_deg,
            "intensity_measured": intensity,
            "mean_intensity": mean_intensity,
            "a_sin2": float(coeffs[0]),
            "b_cos2": float(coeffs[1]),
            "curve_csv": str(info["curve_csv"]),
            "stretch_scale": float(info.get("stretch_scale", 1.0)),
        }
        with (OUT_DIR / f"measured_intensity_vs_r_{size}.csv").open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["r_mean", "theta_deg", "angle_window_intensity_mean", "rod"])
            for r_val, theta_val, inten, rod in records:
                writer.writerow([r_val, theta_val, inten, rod])
        with (OUT_DIR / f"fitted_intensity_model_{size}.csv").open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["rod_size", "a_sin2", "b_cos2", "model"])
            writer.writerow([size, float(coeffs[0]), float(coeffs[1]), "I(theta)=a_sin2*sin(theta)^2+b_cos2*cos(theta)^2"])
    return out


def fitted_intensity_grid(data: dict[str, np.ndarray | float | str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    curve = np.genfromtxt(str(data["curve_csv"]), delimiter=",", names=True)
    curve_r = np.asarray(curve["r"], dtype=np.float64) * float(data["stretch_scale"])
    theta_curve = np.asarray(curve["theta_deg_center"], dtype=np.float64)
    order = np.argsort(curve_r)
    curve_r = curve_r[order]
    theta_curve = theta_curve[order]
    r_min = float(np.ceil(np.min(curve_r) / R_STEP) * R_STEP)
    r_max = float(np.floor(np.max(curve_r) / R_STEP) * R_STEP)
    r_grid = np.round(np.arange(r_min, r_max + 0.5 * R_STEP, R_STEP), 4)
    theta_grid = np.interp(r_grid, curve_r, theta_curve)
    theta_rad = np.radians(theta_grid)
    intensity_grid = (
        float(data["a_sin2"]) * np.sin(theta_rad) ** 2
        + float(data["b_cos2"]) * np.cos(theta_rad) ** 2
    )
    return r_grid, theta_grid, intensity_grid


def write_mean_intensity_percentage_outputs(samples_by_recording: list[dict[str, Any]], rod_data: dict[str, dict[str, np.ndarray | float | str]]) -> None:
    aggregate = np.concatenate([r["mean_subtracted_window_samples"] for r in samples_by_recording])
    with (OUT_DIR / "mean_intensity_percentage_summary.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["rod_size", "mean_rod_intensity_14x14", "background_mean", "background_std", "mean_percent", "std_percent", "p50_percent", "p95_percent", "p99_percent"])
        for size, data in rod_data.items():
            denom = float(data["mean_intensity"])
            pct = 100.0 * aggregate / denom
            d = describe(pct)
            writer.writerow([size, denom, float(np.mean(aggregate)), float(np.std(aggregate)), d["mean"], d["std"], d["p50"], d["p95"], d["p99"]])

    for size, data in rod_data.items():
        denom = float(data["mean_intensity"])
        all_pct = [100.0 * r["mean_subtracted_window_samples"] / denom for r in samples_by_recording]
        upper = max(float(np.percentile(v, 99.5)) for v in all_pct)
        bins = np.arange(0.0, max(upper + PERCENT_BIN_WIDTH, 1.0), PERCENT_BIN_WIDTH)
        plt.figure(figsize=(9, 5.5))
        for record, pct in zip(samples_by_recording, all_pct):
            label = record["recording"].replace("frame_stack_", "").replace(".npy", "")
            plt.hist(pct, bins=bins, density=True, histtype="step", linewidth=1.7, label=label)
        plt.title(f"Mean-subtracted background percentage distributions ({size})")
        plt.xlabel("Background / mean rod signal (%)")
        plt.ylabel("Probability density")
        plt.yscale("log")
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(OUT_DIR / f"mean_intensity_percentage_distributions_{size}.png", dpi=240)
        plt.close()


def write_density_vs_r_tables(samples_by_recording: list[dict[str, Any]], rod_data: dict[str, dict[str, np.ndarray | float | str]]) -> None:
    aggregate = np.concatenate([r["mean_subtracted_window_samples"] for r in samples_by_recording])
    for size, data in rod_data.items():
        r_measured = np.asarray(data["r_measured"], dtype=np.float64)
        theta_measured = np.asarray(data["theta_measured_deg"], dtype=np.float64)
        intensity_measured = np.asarray(data["intensity_measured"], dtype=np.float64)
        r_grid, theta_grid, intensity_grid = fitted_intensity_grid(data)
        max_pct = float(np.percentile(100.0 * aggregate / np.min(intensity_grid), 99.9))
        bins = np.arange(0.0, max(max_pct * 1.05, 1.0) + PERCENT_BIN_WIDTH, PERCENT_BIN_WIDTH)
        centers = 0.5 * (bins[:-1] + bins[1:])
        densities = []
        for intensity in intensity_grid:
            pct = 100.0 * aggregate / intensity
            hist, _ = np.histogram(pct, bins=bins, density=True)
            densities.append(hist)
        density_arr = np.asarray(densities).T
        with (OUT_DIR / f"percentage_density_vs_r_{size}.csv").open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["percentage_bin_center"] + [f"r={r:.4f}" for r in r_grid])
            for pct_center, row in zip(centers, density_arr):
                writer.writerow([pct_center] + list(row))
        with (OUT_DIR / f"intensity_fitted_vs_r_{size}.csv").open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["r", "theta_deg_from_curve", "fitted_angle_window_intensity", "a_sin2", "b_cos2"])
            for r_val, theta_val, inten in zip(r_grid, theta_grid, intensity_grid):
                writer.writerow([r_val, theta_val, inten, float(data["a_sin2"]), float(data["b_cos2"])])

        fig, ax = plt.subplots(figsize=(9, 5.5))
        mesh = ax.pcolormesh(r_grid, centers, density_arr, shading="auto", cmap="viridis")
        ax.set_title(f"Background percentage density vs r ({size})")
        ax.set_xlabel("Measured r")
        ax.set_ylabel("Background / rod signal (%)")
        fig.colorbar(mesh, ax=ax, label="Probability density")
        fig.tight_layout()
        fig.savefig(OUT_DIR / f"percentage_density_vs_r_{size}.png", dpi=240)
        plt.close(fig)

        plt.figure(figsize=(7.5, 5))
        plt.scatter(r_measured, intensity_measured, s=18, alpha=0.65, label="measured rods")
        plt.plot(r_grid, intensity_grid, color="black", linewidth=2, label="fitted cos^2/sin^2 model")
        plt.title(f"Fitted 14x14 rod intensity vs r ({size})")
        plt.xlabel("Measured r")
        plt.ylabel("14x14 angle-window mean intensity")
        plt.legend()
        plt.tight_layout()
        plt.savefig(OUT_DIR / f"fitted_intensity_vs_r_{size}.png", dpi=240)
        plt.close()

        plt.figure(figsize=(7.5, 5))
        theta_line = np.linspace(0.0, 90.0, 300)
        theta_rad = np.radians(theta_line)
        intensity_line = (
            float(data["a_sin2"]) * np.sin(theta_rad) ** 2
            + float(data["b_cos2"]) * np.cos(theta_rad) ** 2
        )
        plt.scatter(theta_measured, intensity_measured, s=18, alpha=0.65, label="measured rods")
        plt.plot(theta_line, intensity_line, color="black", linewidth=2, label="fit")
        plt.title(f"Fitted intensity vs theta ({size})")
        plt.xlabel("theta (deg)")
        plt.ylabel("14x14 angle-window mean intensity")
        plt.legend()
        plt.tight_layout()
        plt.savefig(OUT_DIR / f"fitted_intensity_vs_theta_{size}.png", dpi=240)
        plt.close()


def write_recording_summary(samples_by_recording: list[dict[str, Any]]) -> None:
    with (OUT_DIR / "three_recording_mean_subtraction_summary.csv").open("w", newline="") as f:
        fields = [
            "recording",
            "shape",
            "raw_window_mean",
            "raw_window_std",
            "mean_subtracted_window_mean",
            "mean_subtracted_window_std",
            "mean_subtracted_window_p50",
            "mean_subtracted_window_p99",
            "mean_subtracted_zero_fraction",
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for record in samples_by_recording:
            raw = record["raw_window_stats"]
            res = record["mean_subtracted_window_stats"]
            writer.writerow(
                {
                    "recording": record["recording"],
                    "shape": record["shape"],
                    "raw_window_mean": raw["mean"],
                    "raw_window_std": raw["std"],
                    "mean_subtracted_window_mean": res["mean"],
                    "mean_subtracted_window_std": res["std"],
                    "mean_subtracted_window_p50": res["p50"],
                    "mean_subtracted_window_p99": res["p99"],
                    "mean_subtracted_zero_fraction": res["zero_fraction"],
                }
            )


def write_readme(samples_by_recording: list[dict[str, Any]], rod_data: dict[str, dict[str, np.ndarray | float | str]]) -> None:
    lines = [
        "Mean-subtracted three-profile background percentage distributions",
        "",
        f"Input folder: {BACKGROUND_DIR}",
        f"Output folder: {OUT_DIR}",
        "",
        "Recordings used:",
    ]
    for record in samples_by_recording:
        lines.append(f"- {record['recording']}")
    lines.extend(
        [
            "",
            "Method:",
            "- Excluded the first recording frame_stack_20260716-135510.npy.",
            "- For each of the three remaining stacks, computed the temporal mean frame and subtracted it from each sampled frame.",
            "- Negative residuals were clipped to zero.",
            "- Background values are 14x14 raw-pixel box means, matching the angle-calculation window.",
            "- Mean-intensity percentage distributions use one mean 14x14 rod intensity per rod size.",
            "- r-dependent density tables use I(theta)=a sin^2(theta)+b cos^2(theta), fitted to the stationary-rod 14x14 angle-window intensities.",
            "- theta(r) is taken from the saved stretched theta(r) curve for each rod size; no median or rolling interpolation of intensity is used.",
            "",
            "Mean rod intensities used:",
        ]
    )
    for size, data in rod_data.items():
        lines.append(
            f"- {size}: mean={float(data['mean_intensity']):.6g}; "
            f"a_sin2={float(data['a_sin2']):.6g}; b_cos2={float(data['b_cos2']):.6g}"
        )
    lines.extend(
        [
            "",
            "Main outputs:",
            "- mean_intensity_percentage_distributions_40x65nm.png",
            "- mean_intensity_percentage_distributions_25x65nm.png",
            "- mean_intensity_percentage_summary.csv",
            "- percentage_density_vs_r_40x65nm.csv/png",
            "- percentage_density_vs_r_25x65nm.csv/png",
            "- measured_intensity_vs_r_40x65nm.csv",
            "- measured_intensity_vs_r_25x65nm.csv",
            "- fitted_intensity_vs_r_40x65nm.csv/png",
            "- fitted_intensity_vs_r_25x65nm.csv/png",
            "- fitted_intensity_vs_theta_40x65nm.png",
            "- fitted_intensity_vs_theta_25x65nm.png",
            "",
            "Density table format:",
            "- First column: percentage_bin_center.",
            "- Header columns after that: r values.",
            "- Body values: probability density at that percentage bin for the corresponding r.",
        ]
    )
    (OUT_DIR / "README_three_profile_mean_subtracted_percentages.txt").write_text("\n".join(lines) + "\n")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(1607202603)
    samples_by_recording = []
    for path in STACKS:
        print(f"Analysing {path.name}")
        samples_by_recording.append(mean_subtracted_window_samples(path, rng))
    write_recording_summary(samples_by_recording)
    rod_data = load_stationary_intensity_models()
    write_mean_intensity_percentage_outputs(samples_by_recording, rod_data)
    write_density_vs_r_tables(samples_by_recording, rod_data)
    write_readme(samples_by_recording, rod_data)
    print(f"Wrote outputs to {OUT_DIR}")


if __name__ == "__main__":
    main()
