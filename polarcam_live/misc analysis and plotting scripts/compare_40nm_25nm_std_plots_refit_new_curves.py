from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    from scipy.optimize import least_squares  # type: ignore
except Exception:
    least_squares = None


@dataclass
class RodDatum:
    rod: str
    n_frames: int
    r_mean: float
    sigma_xy: float
    intensity_mean: float
    theta_deg: float = float("nan")
    dtheta_dr_rad: float = float("nan")
    sigma_theta_measured_deg: float = float("nan")
    sigma_phi_measured_deg: float = float("nan")


DATASETS = {
    "40x65nm": {
        "good_dir": Path("stationary rod data 01072026") / "good",
        "curve_csv": Path("theta_r_curves_for_analysis") / "theta_r_curve_40x65nm_recording_bootstrap.csv",
        "color_points_theta": "#1f77b4",
        "color_curve_theta": "#0b4f8a",
        "color_points_phi": "#2ca02c",
        "color_curve_phi": "#1b6f1b",
    },
    "25x65nm": {
        "good_dir": Path("stationary rods 25nm 02072026") / "filtered_intensity_p98_50_to_400_sigma_theta_pruned" / "good",
        "curve_csv": Path("theta_r_curves_for_analysis") / "theta_r_curve_25x65nm_recording_bootstrap.csv",
        "color_points_theta": "#d62728",
        "color_curve_theta": "#8c1d1d",
        "color_points_phi": "#ff7f0e",
        "color_curve_phi": "#b35900",
    },
}

OUTPUT_DIR = Path("outputs") / "40nm_vs_25nm_std_comparison_refit_new_curves"
STD_CURVE_MAX_DEG = 15.0


def _load_xy_series(meta_path: Path) -> np.ndarray:
    payload = json.loads(meta_path.read_text(encoding="utf-8"))
    xy_series = payload.get("xy_series")
    if not isinstance(xy_series, list):
        raise ValueError(f"No xy_series in {meta_path}")
    arr = np.asarray(xy_series, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] < 2:
        raise ValueError(f"Bad xy_series shape in {meta_path}: {arr.shape}")
    arr = arr[:, :2]
    valid = np.isfinite(arr[:, 0]) & np.isfinite(arr[:, 1])
    return arr[valid]


def _estimate_intensity_mean(npy_path: Path) -> float:
    arr = np.load(npy_path, allow_pickle=False)
    return float(np.mean(np.asarray(arr, dtype=np.float64)))


def _load_curve(curve_csv: Path) -> dict[str, np.ndarray]:
    rows = list(csv.DictReader(curve_csv.open("r", encoding="utf-8", newline="")))
    r = np.asarray([float(row["r"]) for row in rows], dtype=np.float64)
    theta_deg = np.asarray([float(row["theta_deg_center"]) for row in rows], dtype=np.float64)
    theta_rad = np.radians(theta_deg)
    dtheta_dr_rad = np.gradient(theta_rad, r)
    return {"r": r, "theta_deg": theta_deg, "dtheta_dr_rad": dtheta_dr_rad}


def _interp(arr_x: np.ndarray, arr_y: np.ndarray, x_new: float) -> float:
    return float(np.interp(float(x_new), arr_x, arr_y, left=float(arr_y[0]), right=float(arr_y[-1])))


def _stretch_curve_for_dataset(curve: dict[str, np.ndarray], dataset_r_max: float) -> tuple[dict[str, np.ndarray], float]:
    curve_r_max = float(np.max(curve["r"]))
    scale = float(dataset_r_max / max(curve_r_max, 1e-12))
    return {
        "r": curve["r"] * scale,
        "theta_deg": curve["theta_deg"].copy(),
        "dtheta_dr_rad": curve["dtheta_dr_rad"] / max(scale, 1e-12),
    }, scale


def _load_data(good_dir: Path) -> list[RodDatum]:
    out: list[RodDatum] = []
    for rod_dir in sorted([p for p in good_dir.iterdir() if p.is_dir()]):
        meta_path = rod_dir / "capture_maxfps_15x15_meta.json"
        npy_path = rod_dir / "capture_maxfps_15x15.npy"
        if not meta_path.exists() or not npy_path.exists():
            continue
        xy = _load_xy_series(meta_path)
        if xy.size == 0:
            continue
        x = xy[:, 0]
        y = xy[:, 1]
        r = np.sqrt((x * x) + (y * y))
        intensity_mean = _estimate_intensity_mean(npy_path)
        if not np.isfinite(intensity_mean) or intensity_mean <= 0.0:
            continue
        out.append(
            RodDatum(
                rod=rod_dir.name,
                n_frames=int(xy.shape[0]),
                r_mean=float(np.mean(r)),
                sigma_xy=float(np.hypot(np.std(x), np.std(y))),
                intensity_mean=float(intensity_mean),
            )
        )
    return out


def _robust_dataset_r_max(data: list[RodDatum]) -> tuple[float, float]:
    r_sorted = np.sort(np.asarray([d.r_mean for d in data], dtype=np.float64))
    if r_sorted.size == 0:
        raise ValueError("No rods available to determine dataset r_max")
    raw_max = float(r_sorted[-1])
    if r_sorted.size == 1:
        return raw_max, raw_max
    return float(r_sorted[-2]), raw_max


def _apply_curve_to_data(data: list[RodDatum], curve: dict[str, np.ndarray]) -> None:
    for d in data:
        d.theta_deg = _interp(curve["r"], curve["theta_deg"], d.r_mean)
        d.dtheta_dr_rad = _interp(curve["r"], curve["dtheta_dr_rad"], d.r_mean)
        d.sigma_theta_measured_deg = float(np.degrees(abs(d.dtheta_dr_rad) * d.sigma_xy))
        d.sigma_phi_measured_deg = float(np.degrees(d.sigma_xy / max(2.0 * d.r_mean, 1e-12)))


def _predict(data: list[RodDatum], a_sin2: float, b_cos2: float, c_over_i2: float) -> dict[str, np.ndarray]:
    theta_deg = np.asarray([d.theta_deg for d in data], dtype=np.float64)
    r_mean = np.asarray([d.r_mean for d in data], dtype=np.float64)
    dtheta_dr_deg = np.asarray([np.degrees(abs(d.dtheta_dr_rad)) for d in data], dtype=np.float64)
    theta_rad = np.radians(theta_deg)
    intensity_model = a_sin2 * (np.sin(theta_rad) ** 2) + b_cos2 * (np.cos(theta_rad) ** 2)
    intensity_model = np.maximum(intensity_model, 1e-12)
    sigma_xy = np.sqrt(c_over_i2 / (intensity_model * intensity_model))
    sigma_theta = dtheta_dr_deg * sigma_xy
    sigma_phi = np.degrees(sigma_xy / np.maximum(2.0 * r_mean, 1e-12))
    return {
        "intensity_model": intensity_model,
        "sigma_xy": sigma_xy,
        "sigma_theta": sigma_theta,
        "sigma_phi": sigma_phi,
    }


def _initial_guess(data: list[RodDatum]) -> tuple[float, float, float]:
    th = np.radians(np.asarray([d.theta_deg for d in data], dtype=np.float64))
    inten = np.asarray([d.intensity_mean for d in data], dtype=np.float64)
    X = np.column_stack([np.sin(th) ** 2, np.cos(th) ** 2])
    beta, *_ = np.linalg.lstsq(X, inten, rcond=None)
    beta = np.maximum(beta, 1e-6)
    i_model = np.maximum(beta[0] * (np.sin(th) ** 2) + beta[1] * (np.cos(th) ** 2), 1e-12)
    sigma_xy2 = np.asarray([d.sigma_xy * d.sigma_xy for d in data], dtype=np.float64)
    c0 = float(np.median(sigma_xy2 * i_model * i_model))
    return float(beta[0]), float(beta[1]), max(c0, 1e-12)


def _fit(data: list[RodDatum]) -> dict:
    if least_squares is None:
        raise RuntimeError("scipy.optimize.least_squares is required for this fit.")
    sigma_theta_meas = np.asarray([d.sigma_theta_measured_deg for d in data], dtype=np.float64)
    sigma_phi_meas = np.asarray([d.sigma_phi_measured_deg for d in data], dtype=np.float64)
    x0 = np.asarray(_initial_guess(data), dtype=np.float64)

    def residuals(params: np.ndarray) -> np.ndarray:
        pred = _predict(data, float(params[0]), float(params[1]), float(params[2]))
        return np.concatenate([pred["sigma_theta"] - sigma_theta_meas, pred["sigma_phi"] - sigma_phi_meas])

    result = least_squares(
        residuals,
        x0=x0,
        bounds=(np.zeros(3, dtype=np.float64), np.full(3, np.inf, dtype=np.float64)),
        method="trf",
    )
    pred = _predict(data, float(result.x[0]), float(result.x[1]), float(result.x[2]))
    resid = residuals(result.x)
    return {
        "a_sin2": float(result.x[0]),
        "b_cos2": float(result.x[1]),
        "c_over_i2": float(result.x[2]),
        "initial_guess": {"a_sin2": float(x0[0]), "b_cos2": float(x0[1]), "c_over_i2": float(x0[2])},
        "rmse_joint": float(np.sqrt(np.mean(resid * resid))),
        "rmse_theta_deg": float(np.sqrt(np.mean((pred["sigma_theta"] - sigma_theta_meas) ** 2))),
        "rmse_phi_deg": float(np.sqrt(np.mean((pred["sigma_phi"] - sigma_phi_meas) ** 2))),
        "pred": pred,
    }


def _clip_curve(theta_grid: np.ndarray, y_grid: np.ndarray, max_std: float) -> tuple[np.ndarray, np.ndarray]:
    valid = np.asarray(y_grid <= float(max_std), dtype=bool)
    return theta_grid[valid], y_grid[valid]


def _filter_plot_points(
    theta_vals: np.ndarray,
    y_vals: np.ndarray,
    max_std: float,
) -> tuple[np.ndarray, np.ndarray]:
    valid = np.isfinite(theta_vals) & np.isfinite(y_vals) & (y_vals <= float(max_std))
    return theta_vals[valid], y_vals[valid]


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    payloads: dict[str, dict] = {}
    theta_min = 180.0
    theta_max = -180.0

    for label, cfg in DATASETS.items():
        data = _load_data(Path.cwd() / cfg["good_dir"])
        if not data:
            raise SystemExit(f"No data found for {label}")
        dataset_r_max, raw_dataset_r_max = _robust_dataset_r_max(data)
        base_curve = _load_curve(Path.cwd() / cfg["curve_csv"])
        curve, stretch_scale = _stretch_curve_for_dataset(base_curve, dataset_r_max)
        _apply_curve_to_data(data, curve)
        fit = _fit(data)
        theta_vals = np.asarray([d.theta_deg for d in data], dtype=np.float64)
        theta_min = min(theta_min, float(np.min(theta_vals)), float(np.min(curve["theta_deg"])))
        theta_max = max(theta_max, float(np.max(theta_vals)), float(np.max(curve["theta_deg"])))

        rows = []
        for d, intensity_model, sigma_theta_pred, sigma_phi_pred, sigma_xy_pred in zip(
            data,
            fit["pred"]["intensity_model"],
            fit["pred"]["sigma_theta"],
            fit["pred"]["sigma_phi"],
            fit["pred"]["sigma_xy"],
        ):
            rows.append(
                {
                    "rod": d.rod,
                    "theta_deg": d.theta_deg,
                    "r_mean": d.r_mean,
                    "intensity_mean": d.intensity_mean,
                    "intensity_model": float(intensity_model),
                    "sigma_xy_measured": d.sigma_xy,
                    "sigma_xy_pred": float(sigma_xy_pred),
                    "sigma_theta_measured_deg": d.sigma_theta_measured_deg,
                    "sigma_theta_pred_deg": float(sigma_theta_pred),
                    "sigma_phi_measured_deg": d.sigma_phi_measured_deg,
                    "sigma_phi_pred_deg": float(sigma_phi_pred),
                }
            )
        csv_path = OUTPUT_DIR / f"{label.replace('x', 'x').replace(' ', '_').lower()}_refitted_points.csv"
        with csv_path.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

        payloads[label] = {
            "data": data,
            "curve": curve,
            "fit": fit,
            "rows_csv": csv_path,
            **cfg,
            "stretch_scale": stretch_scale,
            "dataset_r_max": dataset_r_max,
            "raw_dataset_r_max": raw_dataset_r_max,
        }

    theta_grid = np.linspace(theta_min, theta_max, 800)

    fig, ax = plt.subplots(figsize=(8.2, 5.4))
    for label, p in payloads.items():
        theta_vals = np.asarray([d.theta_deg for d in p["data"]], dtype=np.float64)
        sigma_theta_vals = np.asarray([d.sigma_theta_measured_deg for d in p["data"]], dtype=np.float64)
        theta_plot, sigma_theta_plot = _filter_plot_points(theta_vals, sigma_theta_vals, STD_CURVE_MAX_DEG)
        ax.scatter(
            theta_plot,
            sigma_theta_plot,
            s=28,
            alpha=0.78,
            color=p["color_points_theta"],
            edgecolors="none",
            label=f"{label} points",
        )
        pred_grid = _predict(
            [
                RodDatum(
                    rod="grid",
                    n_frames=0,
                    r_mean=float(np.interp(t, p["curve"]["theta_deg"], p["curve"]["r"], left=p["curve"]["r"][0], right=p["curve"]["r"][-1])),
                    sigma_xy=0.0,
                    intensity_mean=0.0,
                    theta_deg=float(t),
                    dtheta_dr_rad=float(
                        np.interp(
                            t,
                            p["curve"]["theta_deg"],
                            p["curve"]["dtheta_dr_rad"],
                            left=p["curve"]["dtheta_dr_rad"][0],
                            right=p["curve"]["dtheta_dr_rad"][-1],
                        )
                    ),
                    sigma_theta_measured_deg=0.0,
                    sigma_phi_measured_deg=0.0,
                )
                for t in theta_grid
            ],
            p["fit"]["a_sin2"],
            p["fit"]["b_cos2"],
            p["fit"]["c_over_i2"],
        )
        tx, ty = _clip_curve(theta_grid, pred_grid["sigma_theta"], STD_CURVE_MAX_DEG)
        ax.plot(tx, ty, lw=2.0, color=p["color_curve_theta"], label=f"{label} theory")
    ax.set_xlabel("theta (deg)")
    ax.set_ylabel("std dev theta (deg)")
    ax.set_title("Theta std vs theta: 40x65nm and 25x65nm refit with remade theta(r) curves")
    ax.set_ylim(0.0, STD_CURVE_MAX_DEG)
    ax.grid(True, alpha=0.22)
    ax.legend(loc="best", frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "theta_std_40nm_vs_25nm_refit_new_curves.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.2, 5.4))
    for label, p in payloads.items():
        theta_vals = np.asarray([d.theta_deg for d in p["data"]], dtype=np.float64)
        sigma_phi_vals = np.asarray([d.sigma_phi_measured_deg for d in p["data"]], dtype=np.float64)
        theta_plot, sigma_phi_plot = _filter_plot_points(theta_vals, sigma_phi_vals, STD_CURVE_MAX_DEG)
        ax.scatter(
            theta_plot,
            sigma_phi_plot,
            s=28,
            alpha=0.78,
            color=p["color_points_phi"],
            edgecolors="none",
            label=f"{label} points",
        )
        pred_grid = _predict(
            [
                RodDatum(
                    rod="grid",
                    n_frames=0,
                    r_mean=float(np.interp(t, p["curve"]["theta_deg"], p["curve"]["r"], left=p["curve"]["r"][0], right=p["curve"]["r"][-1])),
                    sigma_xy=0.0,
                    intensity_mean=0.0,
                    theta_deg=float(t),
                    dtheta_dr_rad=float(
                        np.interp(
                            t,
                            p["curve"]["theta_deg"],
                            p["curve"]["dtheta_dr_rad"],
                            left=p["curve"]["dtheta_dr_rad"][0],
                            right=p["curve"]["dtheta_dr_rad"][-1],
                        )
                    ),
                    sigma_theta_measured_deg=0.0,
                    sigma_phi_measured_deg=0.0,
                )
                for t in theta_grid
            ],
            p["fit"]["a_sin2"],
            p["fit"]["b_cos2"],
            p["fit"]["c_over_i2"],
        )
        tx, ty = _clip_curve(theta_grid, pred_grid["sigma_phi"], STD_CURVE_MAX_DEG)
        ax.plot(tx, ty, lw=2.0, color=p["color_curve_phi"], label=f"{label} theory")
    ax.set_xlabel("theta (deg)")
    ax.set_ylabel("std dev phi (deg)")
    ax.set_title("Phi std vs theta: 40x65nm and 25x65nm refit with remade theta(r) curves")
    ax.set_ylim(0.0, STD_CURVE_MAX_DEG)
    ax.grid(True, alpha=0.22)
    ax.legend(loc="best", frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "phi_std_40nm_vs_25nm_refit_new_curves.png", dpi=220)
    plt.close(fig)

    for label, p in payloads.items():
        theta_vals = np.asarray([d.theta_deg for d in p["data"]], dtype=np.float64)
        intensity_vals = np.asarray([d.intensity_mean for d in p["data"]], dtype=np.float64)
        theta_curve = np.linspace(float(np.min(theta_vals)), float(np.max(theta_vals)), 800)
        theta_curve_rad = np.radians(theta_curve)
        intensity_curve = (
            p["fit"]["a_sin2"] * (np.sin(theta_curve_rad) ** 2)
            + p["fit"]["b_cos2"] * (np.cos(theta_curve_rad) ** 2)
        )

        fig, ax = plt.subplots(figsize=(7.6, 5.2))
        ax.scatter(
            theta_vals,
            intensity_vals,
            s=28,
            alpha=0.78,
            color=p["color_points_theta"],
            edgecolors="none",
            label=f"{label} rods",
        )
        ax.plot(
            theta_curve,
            intensity_curve,
            lw=2.0,
            color=p["color_curve_theta"],
            label=f"{label} fit",
        )
        ax.set_xlabel("theta (deg)")
        ax.set_ylabel("mean intensity")
        ax.set_title(f"Intensity vs theta: {label}")
        ax.grid(True, alpha=0.22)
        ax.legend(loc="best", frameon=False)
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / f"intensity_vs_theta_{label.lower()}.png", dpi=220)
        plt.close(fig)

    summary = {
        "output_dir": str(OUTPUT_DIR.resolve()),
        "std_curve_clip_deg": STD_CURVE_MAX_DEG,
        "datasets": {
            label: {
                "good_dir": str((Path.cwd() / p["good_dir"]).resolve()),
                "curve_csv": str((Path.cwd() / p["curve_csv"]).resolve()),
                "refitted_points_csv": str(p["rows_csv"].resolve()),
                "stretch_scale": float(p["stretch_scale"]),
                "dataset_r_max": float(p["dataset_r_max"]),
                "raw_dataset_r_max": float(p["raw_dataset_r_max"]),
                "dataset_r_max_rule": "second-highest observed rod mean r in the stationary-water dataset",
                "fit_params": {
                    "a_sin2": float(p["fit"]["a_sin2"]),
                    "b_cos2": float(p["fit"]["b_cos2"]),
                    "c_over_i2": float(p["fit"]["c_over_i2"]),
                },
                "rmse_joint": float(p["fit"]["rmse_joint"]),
                "rmse_theta_deg": float(p["fit"]["rmse_theta_deg"]),
                "rmse_phi_deg": float(p["fit"]["rmse_phi_deg"]),
                "n_rods": int(len(p["data"])),
            }
            for label, p in payloads.items()
        },
        "method": "Refit the intensity model I(theta)=a sin^2(theta)+b cos^2(theta) and sigma_xy(theta)^2=c/I(theta)^2 using the same filtered stationary rod datasets as before, but with theta and dtheta/dr recomputed from the new saved theta(r) curves after stretching each curve to the second-highest observed rod mean r in the corresponding stationary-water dataset.",
    }
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
