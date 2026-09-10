from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
BACKGROUND_DIR = ROOT / "datasets" / "background characterisation" / "for daping post subtraction"
OUT_DIR = ROOT / "summarise background statistics"
ANGLE_WINDOW_RAW = 14
MAX_FRAMES_PER_STACK = 15
MAX_SAMPLES_PER_KIND_PER_FILE = 250_000
R_GRID = np.linspace(0.05, 0.95, 181)

SUMMARY_JSON = ROOT / "outputs" / "40nm_vs_25nm_std_comparison_refit_new_curves" / "summary.json"
CURVE_DIR = ROOT / "theta_r_curves_for_analysis"


@dataclass
class StackStats:
    name: str
    shape: tuple[int, ...]
    dtype: str
    raw_mean: float
    raw_std: float
    raw_p50: float
    raw_p99: float
    raw_zero_fraction: float
    profile_mean: float
    profile_std: float
    profile_p50: float
    profile_p99: float
    profile_window_mean: float
    profile_window_std: float
    residual_mean: float
    residual_std: float
    residual_p50: float
    residual_p99: float
    residual_zero_fraction: float
    residual_window_mean: float
    residual_window_std: float


def ensure_out_dir() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)


def selected_frame_indices(n_frames: int) -> np.ndarray:
    n = min(MAX_FRAMES_PER_STACK, n_frames)
    return np.unique(np.linspace(0, n_frames - 1, n, dtype=int))


def sample_values(values: np.ndarray, max_count: int, rng: np.random.Generator) -> np.ndarray:
    flat = np.asarray(values).ravel()
    if flat.size <= max_count:
        return flat.astype(np.float32, copy=True)
    idx = rng.choice(flat.size, size=max_count, replace=False)
    return flat[idx].astype(np.float32, copy=False)


def box_mean_14x14(image: np.ndarray) -> np.ndarray:
    image_f = np.asarray(image, dtype=np.float32)
    return cv2.blur(
        image_f,
        (ANGLE_WINDOW_RAW, ANGLE_WINDOW_RAW),
        borderType=cv2.BORDER_REFLECT,
    )


def describe(values: np.ndarray) -> dict[str, float]:
    v = np.asarray(values, dtype=np.float64)
    mean = float(np.mean(v))
    std = float(np.std(v))
    centered = v - mean
    if std > 0:
        skew = float(np.mean(centered**3) / std**3)
        excess_kurtosis = float(np.mean(centered**4) / std**4 - 3.0)
    else:
        skew = 0.0
        excess_kurtosis = 0.0
    return {
        "mean": mean,
        "std": std,
        "p01": float(np.percentile(v, 1)),
        "p05": float(np.percentile(v, 5)),
        "p50": float(np.percentile(v, 50)),
        "p95": float(np.percentile(v, 95)),
        "p99": float(np.percentile(v, 99)),
        "min": float(np.min(v)),
        "max": float(np.max(v)),
        "zero_fraction": float(np.mean(v <= 0)),
        "skew": skew,
        "excess_kurtosis": excess_kurtosis,
    }


def load_stack_and_summarise(path: Path, rng: np.random.Generator) -> tuple[StackStats, dict[str, np.ndarray]]:
    stack = np.load(path, mmap_mode="r")
    if stack.ndim == 2:
        frame_indices = np.array([0])
        n_frames = 1
        frame_shape = stack.shape
    elif stack.ndim == 3:
        n_frames = int(stack.shape[0])
        frame_indices = selected_frame_indices(n_frames)
        frame_shape = stack.shape[1:]
    else:
        raise ValueError(f"{path} has unsupported shape {stack.shape}")

    # Mean profile is the frame-averaged background image. It is the profile
    # that would be subtracted from the stack in a static-profile correction.
    if stack.ndim == 2:
        profile = np.asarray(stack, dtype=np.float32)
    else:
        acc = np.zeros(frame_shape, dtype=np.float64)
        for i in range(n_frames):
            acc += np.asarray(stack[i], dtype=np.float64)
        profile = (acc / n_frames).astype(np.float32)

    raw_samples: list[np.ndarray] = []
    residual_samples: list[np.ndarray] = []
    residual_window_samples: list[np.ndarray] = []
    raw_window_samples: list[np.ndarray] = []
    per_frame_sample = max(1, MAX_SAMPLES_PER_KIND_PER_FILE // len(frame_indices))

    for i in frame_indices:
        frame = np.asarray(stack if stack.ndim == 2 else stack[i], dtype=np.float32)
        raw_samples.append(sample_values(frame, per_frame_sample, rng))
        raw_window_samples.append(sample_values(box_mean_14x14(frame), per_frame_sample, rng))
        residual = np.clip(frame - profile, 0.0, None)
        residual_samples.append(sample_values(residual, per_frame_sample, rng))
        residual_window_samples.append(sample_values(box_mean_14x14(residual), per_frame_sample, rng))

    profile_samples = sample_values(profile, MAX_SAMPLES_PER_KIND_PER_FILE, rng)
    profile_window_samples = sample_values(box_mean_14x14(profile), MAX_SAMPLES_PER_KIND_PER_FILE, rng)

    raw = np.concatenate(raw_samples)
    raw_window = np.concatenate(raw_window_samples)
    residual = np.concatenate(residual_samples)
    residual_window = np.concatenate(residual_window_samples)

    raw_d = describe(raw)
    profile_d = describe(profile_samples)
    profile_window_d = describe(profile_window_samples)
    residual_d = describe(residual)
    residual_window_d = describe(residual_window)

    stats = StackStats(
        name=path.name,
        shape=tuple(int(x) for x in stack.shape),
        dtype=str(stack.dtype),
        raw_mean=raw_d["mean"],
        raw_std=raw_d["std"],
        raw_p50=raw_d["p50"],
        raw_p99=raw_d["p99"],
        raw_zero_fraction=raw_d["zero_fraction"],
        profile_mean=profile_d["mean"],
        profile_std=profile_d["std"],
        profile_p50=profile_d["p50"],
        profile_p99=profile_d["p99"],
        profile_window_mean=profile_window_d["mean"],
        profile_window_std=profile_window_d["std"],
        residual_mean=residual_d["mean"],
        residual_std=residual_d["std"],
        residual_p50=residual_d["p50"],
        residual_p99=residual_d["p99"],
        residual_zero_fraction=residual_d["zero_fraction"],
        residual_window_mean=residual_window_d["mean"],
        residual_window_std=residual_window_d["std"],
    )

    samples = {
        "raw": raw,
        "raw_window": raw_window,
        "profile": profile_samples,
        "profile_window": profile_window_samples,
        "residual": residual,
        "residual_window": residual_window,
    }
    return stats, samples


def write_stack_csv(stats: list[StackStats]) -> None:
    out = OUT_DIR / "background_profile_summary_by_file.csv"
    fields = list(StackStats.__dataclass_fields__.keys())
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in stats:
            writer.writerow(row.__dict__)


def plot_file_distributions(file_samples: dict[str, dict[str, np.ndarray]]) -> None:
    plot_specs = [
        ("profile", "Mean background profile pixel distribution", "profile_pixel_distribution_by_file.png"),
        ("profile_window", "Mean profile after 14x14 window averaging", "profile_14x14_window_distribution_by_file.png"),
        ("residual", "Post-subtraction residual pixel distribution", "residual_pixel_distribution_by_file.png"),
        ("residual_window", "Residual after 14x14 window averaging", "residual_14x14_window_distribution_by_file.png"),
    ]
    for key, title, filename in plot_specs:
        plt.figure(figsize=(11, 6))
        for name, samples in file_samples.items():
            data = samples[key]
            upper = np.percentile(data, 99.5)
            bins = np.linspace(0, max(upper, 1.0), 140)
            plt.hist(data, bins=bins, histtype="step", density=True, linewidth=1.0, label=name.replace("frame_stack_", "").replace(".npy", ""))
        plt.title(title)
        plt.xlabel("8/12-bit pixel value equivalent")
        plt.ylabel("Probability density")
        plt.yscale("log")
        plt.legend(fontsize=7, ncol=2)
        plt.tight_layout()
        plt.savefig(OUT_DIR / filename, dpi=220)
        plt.close()


def plot_aggregate_distributions(aggregate: dict[str, np.ndarray]) -> None:
    pairs = [
        ("profile", "residual", "Pixel distribution: profile before vs residual after subtraction", "aggregate_pixel_before_after.png"),
        ("profile_window", "residual_window", "14x14 window-mean distribution: before vs after subtraction", "aggregate_14x14_window_before_after.png"),
    ]
    for before_key, after_key, title, filename in pairs:
        before = aggregate[before_key]
        after = aggregate[after_key]
        upper = max(np.percentile(before, 99.5), np.percentile(after, 99.5), 1.0)
        bins = np.linspace(0, upper, 180)
        plt.figure(figsize=(8, 5))
        plt.hist(before, bins=bins, density=True, histtype="stepfilled", alpha=0.35, label="Before subtraction")
        plt.hist(after, bins=bins, density=True, histtype="step", linewidth=1.8, label="After subtraction")
        plt.title(title)
        plt.xlabel("Pixel value / 14x14 window mean")
        plt.ylabel("Probability density")
        plt.yscale("log")
        plt.legend()
        plt.tight_layout()
        plt.savefig(OUT_DIR / filename, dpi=240)
        plt.close()


def load_intensity_models() -> dict[str, dict[str, float]]:
    fallback = {
        "40x65nm": {"a_sin2": 82.56508258751572, "b_cos2": 60.123991414868684, "stretch_scale": 1.0},
        "25x65nm": {"a_sin2": 36.67169824172269, "b_cos2": 21.27295902271135, "stretch_scale": 1.0},
    }
    if not SUMMARY_JSON.exists():
        return fallback
    with SUMMARY_JSON.open("r") as f:
        summary = json.load(f)
    models: dict[str, dict[str, float]] = {}
    for size, info in summary["datasets"].items():
        params = info["fit_params"]
        models[size] = {
            "a_sin2": float(params["a_sin2"]),
            "b_cos2": float(params["b_cos2"]),
            "stretch_scale": float(info.get("stretch_scale", 1.0)),
            "curve_csv": str(info["curve_csv"]),
            "good_dir": str(info["good_dir"]),
        }
    window_models = fit_angle_window_intensity_models(models)
    for size, params in window_models.items():
        models[size].update(params)
    return models


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
    x1 = x0 + win_raw
    y1 = y0 + win_raw
    x0 = max(0, x0)
    y0 = max(0, y0)
    x1 = min(gw, x1)
    y1 = min(gh, y1)
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
    raw_win = np.asarray(arr_use[:, ys, xs], dtype=np.float64)
    if raw_win.size <= 0:
        return float("nan")
    return float(np.mean(raw_win))


def fit_angle_window_intensity_models(models: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
    fitted: dict[str, dict[str, float]] = {}
    rows: list[dict[str, Any]] = []
    for size, model in models.items():
        good_dir = Path(str(model.get("good_dir", "")))
        refitted_points = ROOT / "outputs" / "40nm_vs_25nm_std_comparison_refit_new_curves" / f"{size}_refitted_points.csv"
        if not good_dir.exists() or not refitted_points.exists():
            continue
        point_rows = list(csv.DictReader(refitted_points.open("r", encoding="utf-8", newline="")))
        theta_vals: list[float] = []
        intensity_vals: list[float] = []
        for row in point_rows:
            rod_dir = good_dir / row["rod"]
            inten = rod_angle_window_mean(rod_dir)
            theta = float(row["theta_deg"])
            if np.isfinite(inten) and inten > 0.0 and np.isfinite(theta):
                theta_vals.append(theta)
                intensity_vals.append(inten)
                rows.append(
                    {
                        "rod_size": size,
                        "rod": row["rod"],
                        "theta_deg": theta,
                        "angle_window_intensity_mean": inten,
                        "previous_full_saved_array_intensity_mean": row.get("intensity_mean", ""),
                    }
                )
        if len(theta_vals) < 2:
            continue
        theta_rad = np.radians(np.asarray(theta_vals, dtype=np.float64))
        y = np.asarray(intensity_vals, dtype=np.float64)
        design = np.column_stack([np.sin(theta_rad) ** 2, np.cos(theta_rad) ** 2])
        coeffs, *_ = np.linalg.lstsq(design, y, rcond=None)
        fitted[size] = {
            "a_sin2": float(coeffs[0]),
            "b_cos2": float(coeffs[1]),
            "intensity_model_basis": "14x14_angle_window_mean",
        }
    if rows:
        with (OUT_DIR / "rod_angle_window_intensity_values.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        with (OUT_DIR / "rod_angle_window_intensity_fit_params.csv").open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["rod_size", "a_sin2", "b_cos2", "basis"])
            for size, params in fitted.items():
                writer.writerow([size, params["a_sin2"], params["b_cos2"], params["intensity_model_basis"]])
    return fitted


def theta_from_r(size: str, r_values: np.ndarray, model: dict[str, float]) -> np.ndarray:
    curve_csv = Path(model.get("curve_csv", CURVE_DIR / f"theta_r_curve_{size}_recording_bootstrap.csv"))
    curve = np.genfromtxt(curve_csv, delimiter=",", names=True)
    curve_r = np.asarray(curve["r"], dtype=np.float64) * float(model.get("stretch_scale", 1.0))
    theta = np.asarray(curve["theta_deg_center"], dtype=np.float64)
    order = np.argsort(curve_r)
    return np.interp(r_values, curve_r[order], theta[order], left=np.nan, right=np.nan)


def intensity_from_theta(theta_deg: np.ndarray, model: dict[str, float]) -> np.ndarray:
    theta = np.deg2rad(theta_deg)
    return model["a_sin2"] * np.sin(theta) ** 2 + model["b_cos2"] * np.cos(theta) ** 2


def write_percentage_vs_r(aggregate_stats: dict[str, dict[str, float]]) -> None:
    models = load_intensity_models()
    for size, model in models.items():
        theta = theta_from_r(size, R_GRID, model)
        intensity = intensity_from_theta(theta, model)
        valid = np.isfinite(theta) & (intensity > 0)
        out_csv = OUT_DIR / f"background_percentage_vs_r_{size}.csv"
        with out_csv.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "r",
                    "theta_deg_from_stretched_curve",
                    "mean_rod_intensity_model",
                    "before_profile_window_mean_percent",
                    "before_profile_window_std_percent",
                    "after_residual_window_mean_percent",
                    "after_residual_window_std_percent",
                ]
            )
            for r, t, inten, ok in zip(R_GRID, theta, intensity, valid):
                if not ok:
                    continue
                writer.writerow(
                    [
                        r,
                        t,
                        inten,
                        100.0 * aggregate_stats["profile_window"]["mean"] / inten,
                        100.0 * aggregate_stats["profile_window"]["std"] / inten,
                        100.0 * aggregate_stats["residual_window"]["mean"] / inten,
                        100.0 * aggregate_stats["residual_window"]["std"] / inten,
                    ]
                )

        data = np.genfromtxt(out_csv, delimiter=",", names=True)
        plt.figure(figsize=(8, 5))
        for prefix, label, color in [
            ("before_profile_window", "Before subtraction", "tab:orange"),
            ("after_residual_window", "After subtraction", "tab:blue"),
        ]:
            mean = data[f"{prefix}_mean_percent"]
            std = data[f"{prefix}_std_percent"]
            plt.plot(data["r"], mean, color=color, label=f"{label} mean")
            plt.fill_between(data["r"], mean - std, mean + std, color=color, alpha=0.18, label=f"{label} ±1 std")
        plt.title(f"Background as percentage of {size} rod signal vs measured r")
        plt.xlabel("Measured r")
        plt.ylabel("Background / mean rod signal (%)")
        plt.ylim(bottom=0)
        plt.legend()
        plt.tight_layout()
        plt.savefig(OUT_DIR / f"background_percentage_vs_r_{size}.png", dpi=240)
        plt.close()


def write_aggregate_stats(aggregate: dict[str, np.ndarray]) -> dict[str, dict[str, float]]:
    aggregate_stats = {key: describe(values) for key, values in aggregate.items()}
    with (OUT_DIR / "aggregate_background_distribution_statistics.json").open("w") as f:
        json.dump(aggregate_stats, f, indent=2)
    return aggregate_stats


def write_text_summary(stats: list[StackStats], aggregate_stats: dict[str, dict[str, float]]) -> None:
    profile = aggregate_stats["profile"]
    profile_window = aggregate_stats["profile_window"]
    residual = aggregate_stats["residual"]
    residual_window = aggregate_stats["residual_window"]

    lines = [
        "Background statistics summary",
        "",
        f"Input folder: {BACKGROUND_DIR}",
        f"Output folder: {OUT_DIR}",
        "",
        "Definitions used:",
        f"- Angle window: {ANGLE_WINDOW_RAW}x{ANGLE_WINDOW_RAW} raw pixels.",
        "- Before subtraction: the temporal mean background profile for each recording.",
        "- After subtraction: each sampled frame minus its own temporal mean profile, clipped at zero.",
        "- Window-convolved distribution: 14x14 box mean of the image, matching the raw ROI size used for angle calculations.",
        "- Percentage signal: 100 * background 14x14 window mean / fitted mean rod intensity.",
        "",
        "Aggregate values:",
        f"- Before subtraction profile pixels: mean {profile['mean']:.4g}, std {profile['std']:.4g}, p50 {profile['p50']:.4g}, p99 {profile['p99']:.4g}.",
        f"- Before subtraction 14x14 window mean: mean {profile_window['mean']:.4g}, std {profile_window['std']:.4g}, p50 {profile_window['p50']:.4g}, p99 {profile_window['p99']:.4g}.",
        f"- After subtraction residual pixels: mean {residual['mean']:.4g}, std {residual['std']:.4g}, zero fraction {residual['zero_fraction']:.3f}, p99 {residual['p99']:.4g}.",
        f"- After subtraction 14x14 window mean: mean {residual_window['mean']:.4g}, std {residual_window['std']:.4g}, p50 {residual_window['p50']:.4g}, p99 {residual_window['p99']:.4g}.",
        "",
        "Distribution type:",
        "- The pre-subtraction background profile is not a single pure shot-noise distribution; it is a broad spatial illumination/background profile with additional pixel noise.",
        "- The post-subtraction residual is zero-inflated and right-skewed because negative residuals are clipped to zero. A rectified/censored Gaussian-like residual with a positive tail is a better description than a normal distribution.",
        "- After 14x14 window averaging the residual becomes narrower and more nearly Gaussian by averaging, but it remains zero-inflated/right-skewed because of clipping.",
        "",
        "Rod intensity model used:",
        "- I(theta) is refitted as the mean intensity inside the same 14x14 raw-pixel angle window used for X/Y.",
        "- Fitted parameters are saved in rod_angle_window_intensity_fit_params.csv.",
        "- theta(r) is interpolated from the saved theta_r_curves_for_analysis CSV files using the stretch_scale stored in the refit summary.",
        "",
        "Main output files:",
        "- background_profile_summary_by_file.csv",
        "- aggregate_background_distribution_statistics.json",
        "- aggregate_pixel_before_after.png",
        "- aggregate_14x14_window_before_after.png",
        "- residual_pixel_distribution_by_file.png",
        "- residual_14x14_window_distribution_by_file.png",
        "- background_percentage_vs_r_40x65nm.csv/png",
        "- background_percentage_vs_r_25x65nm.csv/png",
        "",
        "Files analysed:",
    ]
    for row in stats:
        lines.append(f"- {row.name}: shape {row.shape}, dtype {row.dtype}")
    (OUT_DIR / "README_background_statistics.txt").write_text("\n".join(lines) + "\n")


def main() -> None:
    ensure_out_dir()
    rng = np.random.default_rng(12345)
    paths = sorted(BACKGROUND_DIR.glob("*.npy"))
    if not paths:
        raise FileNotFoundError(f"No .npy files found in {BACKGROUND_DIR}")

    all_stats: list[StackStats] = []
    file_samples: dict[str, dict[str, np.ndarray]] = {}
    for path in paths:
        print(f"Analysing {path.name}")
        stats, samples = load_stack_and_summarise(path, rng)
        all_stats.append(stats)
        file_samples[path.name] = samples

    write_stack_csv(all_stats)

    aggregate: dict[str, np.ndarray] = {}
    for key in ["raw", "raw_window", "profile", "profile_window", "residual", "residual_window"]:
        aggregate[key] = np.concatenate([samples[key] for samples in file_samples.values()])

    aggregate_stats = write_aggregate_stats(aggregate)
    plot_file_distributions(file_samples)
    plot_aggregate_distributions(aggregate)
    write_percentage_vs_r(aggregate_stats)
    write_text_summary(all_stats, aggregate_stats)
    print(f"Wrote summary to {OUT_DIR}")


if __name__ == "__main__":
    main()
