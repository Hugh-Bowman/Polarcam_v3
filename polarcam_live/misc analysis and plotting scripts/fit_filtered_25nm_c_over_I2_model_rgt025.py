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


GOOD_DIR = Path("stationary rods 25nm 02072026") / "filtered_intensity_p98_50_to_400" / "good"
CURVE_CSV = (
    Path("glycerol suspended rods 17062026")
    / "plots"
    / "good_vs_good_plus_previous_used_center015"
    / "combined_phi_uniform_optimized"
    / "theta_vs_r_uncertainty_band.csv"
)
OUTPUT_DIR = (
    Path("stationary rods 25nm 02072026")
    / "filtered_intensity_p98_50_to_400"
    / "plots"
    / "pixel_error_c_over_I2_fit_r_gt_0p25"
)
R_MIN_FIT = 0.25


@dataclass
class RodDatum:
    rod: str
    n_frames: int
    r_mean: float
    sigma_xy: float
    intensity_mean: float
    theta_deg: float = float("nan")
    dtheta_dr_rad: float = float("nan")
    sigma_theta_xy_only_deg: float = float("nan")
    sigma_phi_deg: float = float("nan")


def _load_xy_series(meta_path: Path) -> np.ndarray:
    payload = json.loads(meta_path.read_text(encoding="utf-8"))
    xy_series = payload.get("xy_series")
    arr = np.asarray(xy_series, dtype=np.float64)
    arr = arr[:, :2]
    valid = np.isfinite(arr[:, 0]) & np.isfinite(arr[:, 1])
    return arr[valid]


def _load_curve(curve_csv: Path) -> dict[str, np.ndarray]:
    rows = list(csv.DictReader(curve_csv.open("r", encoding="utf-8", newline="")))
    r = np.asarray([float(row["r"]) for row in rows], dtype=np.float64)
    theta_deg = np.asarray([float(row["theta_deg_center"]) for row in rows], dtype=np.float64)
    theta_rad = np.radians(theta_deg)
    dtheta_dr_rad = np.gradient(theta_rad, r)
    return {"r": r, "theta_deg": theta_deg, "dtheta_dr_rad": dtheta_dr_rad}


def _interp(arr_x: np.ndarray, arr_y: np.ndarray, x_new: float) -> float:
    return float(np.interp(float(x_new), arr_x, arr_y, left=float(arr_y[0]), right=float(arr_y[-1])))


def _estimate_intensity_mean(npy_path: Path) -> float:
    arr = np.load(npy_path, allow_pickle=False)
    return float(np.mean(np.asarray(arr, dtype=np.float64)))


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
        sigma_xy = float(np.hypot(np.std(x), np.std(y)))
        out.append(
            RodDatum(
                rod=rod_dir.name,
                n_frames=int(xy.shape[0]),
                r_mean=float(np.mean(r)),
                sigma_xy=sigma_xy,
                intensity_mean=_estimate_intensity_mean(npy_path),
            )
        )
    return out


def _stretch_curve_for_dataset(curve: dict[str, np.ndarray], r_max_dataset: float) -> tuple[dict[str, np.ndarray], float]:
    scale = float(r_max_dataset / max(float(np.max(curve["r"])), 1e-12))
    return (
        {
            "r": curve["r"] * scale,
            "theta_deg": curve["theta_deg"].copy(),
            "dtheta_dr_rad": curve["dtheta_dr_rad"] / max(scale, 1e-12),
        },
        scale,
    )


def _apply_curve_to_data(data: list[RodDatum], curve: dict[str, np.ndarray]) -> None:
    for d in data:
        d.theta_deg = _interp(curve["r"], curve["theta_deg"], d.r_mean)
        d.dtheta_dr_rad = _interp(curve["r"], curve["dtheta_dr_rad"], d.r_mean)
        d.sigma_theta_xy_only_deg = float(np.degrees(abs(d.dtheta_dr_rad) * d.sigma_xy))
        d.sigma_phi_deg = float(np.degrees(d.sigma_xy / max(2.0 * d.r_mean, 1e-12)))


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
    sigma_theta_meas = np.asarray([d.sigma_theta_xy_only_deg for d in data], dtype=np.float64)
    sigma_phi_meas = np.asarray([d.sigma_phi_deg for d in data], dtype=np.float64)
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


def main() -> None:
    out_dir = Path.cwd() / OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    curve = _load_curve(Path.cwd() / CURVE_CSV)
    data_all = _load_data(Path.cwd() / GOOD_DIR)
    if not data_all:
        raise SystemExit("No rods found.")
    stretched_curve, stretch_scale = _stretch_curve_for_dataset(curve, float(max(d.r_mean for d in data_all)))
    _apply_curve_to_data(data_all, stretched_curve)
    data_fit = [d for d in data_all if d.r_mean > R_MIN_FIT]
    if not data_fit:
        raise SystemExit("No rods satisfy r_mean > 0.25.")
    fit = _fit(data_fit)
    pred = _predict(data_all, fit["a_sin2"], fit["b_cos2"], fit["c_over_i2"])

    rows = []
    for datum, intensity_model, sigma_xy_pred, sigma_theta_pred, sigma_phi_pred in zip(
        data_all, pred["intensity_model"], pred["sigma_xy"], pred["sigma_theta"], pred["sigma_phi"]
    ):
        rows.append(
            {
                "rod": datum.rod,
                "r_mean": datum.r_mean,
                "theta_deg": datum.theta_deg,
                "used_in_fit": datum.r_mean > R_MIN_FIT,
                "intensity_mean": datum.intensity_mean,
                "intensity_model": float(intensity_model),
                "sigma_theta_measured_deg": datum.sigma_theta_xy_only_deg,
                "sigma_theta_pred_deg": float(sigma_theta_pred),
                "sigma_phi_measured_deg": datum.sigma_phi_deg,
                "sigma_phi_pred_deg": float(sigma_phi_pred),
                "sigma_xy_measured": datum.sigma_xy,
                "sigma_xy_pred": float(sigma_xy_pred),
            }
        )
    with (out_dir / "filtered_25nm_r_gt_0p25_fitted_points.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    th_all = np.asarray([d.theta_deg for d in data_all], dtype=np.float64)
    inten_all = np.asarray([d.intensity_mean for d in data_all], dtype=np.float64)
    sth_meas_all = np.asarray([d.sigma_theta_xy_only_deg for d in data_all], dtype=np.float64)
    sth_pred_all = np.asarray([float(r["sigma_theta_pred_deg"]) for r in rows], dtype=np.float64)
    sphi_meas_all = np.asarray([d.sigma_phi_deg for d in data_all], dtype=np.float64)
    sphi_pred_all = np.asarray([float(r["sigma_phi_pred_deg"]) for r in rows], dtype=np.float64)
    used_mask = np.asarray([d.r_mean > R_MIN_FIT for d in data_all], dtype=bool)

    theta_grid = np.linspace(float(np.min(stretched_curve["theta_deg"])), float(np.max(stretched_curve["theta_deg"])), 500)
    r_curve = stretched_curve["r"]
    theta_curve = stretched_curve["theta_deg"]
    dtheta_dr_deg_curve = np.degrees(np.abs(stretched_curve["dtheta_dr_rad"]))
    r_grid = np.interp(theta_grid, theta_curve, r_curve, left=float(r_curve[0]), right=float(r_curve[-1]))
    dtheta_dr_deg_grid = np.interp(theta_grid, theta_curve, dtheta_dr_deg_curve, left=float(dtheta_dr_deg_curve[0]), right=float(dtheta_dr_deg_curve[-1]))
    tg = np.radians(theta_grid)
    I_grid = fit["a_sin2"] * (np.sin(tg) ** 2) + fit["b_cos2"] * (np.cos(tg) ** 2)
    I_grid = np.maximum(I_grid, 1e-12)
    sigma_xy_grid = np.sqrt(fit["c_over_i2"] / (I_grid * I_grid))
    sigma_theta_grid = dtheta_dr_deg_grid * sigma_xy_grid
    sigma_phi_grid = np.degrees(sigma_xy_grid / np.maximum(2.0 * r_grid, 1e-12))

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.scatter(th_all[~used_mask], inten_all[~used_mask], s=34, alpha=0.55, color="#b8a6d9", edgecolors="none", label="Not used in fit")
    ax.scatter(th_all[used_mask], inten_all[used_mask], s=40, alpha=0.85, color="#9467bd", edgecolors="none", label="Used in fit")
    ax.plot(theta_grid, I_grid, color="#111111", lw=2.0)
    ax.set_xlabel("theta (deg)")
    ax.set_ylabel("Mean pixel intensity I")
    ax.set_title("25nm rods filtered, r>0.25 fit: intensity vs theta")
    ax.grid(True, alpha=0.22)
    ax.legend(loc="best", frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / "intensity_vs_theta.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.scatter(th_all[~used_mask], sth_meas_all[~used_mask], s=34, alpha=0.55, color="#8fb8df", edgecolors="none", label="Not used in fit")
    ax.scatter(th_all[used_mask], sth_meas_all[used_mask], s=40, alpha=0.85, color="#1f77b4", edgecolors="none", label="Used in fit")
    ax.plot(theta_grid, sigma_theta_grid, color="#d62728", lw=2.0)
    ax.set_xlabel("theta (deg)")
    ax.set_ylabel("std dev theta (deg)")
    ax.set_title("25nm rods filtered, r>0.25 fit: theta std vs theta")
    ax.grid(True, alpha=0.22)
    ax.legend(loc="best", frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / "theta_error_model_fit.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.scatter(th_all[~used_mask], sphi_meas_all[~used_mask], s=34, alpha=0.55, color="#97d497", edgecolors="none", label="Not used in fit")
    ax.scatter(th_all[used_mask], sphi_meas_all[used_mask], s=40, alpha=0.85, color="#2ca02c", edgecolors="none", label="Used in fit")
    ax.plot(theta_grid, sigma_phi_grid, color="#ff7f0e", lw=2.0)
    ax.set_xlabel("theta (deg)")
    ax.set_ylabel("std dev phi (deg)")
    ax.set_title("25nm rods filtered, r>0.25 fit: phi std vs theta")
    ax.grid(True, alpha=0.22)
    ax.legend(loc="best", frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / "phi_error_model_fit.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.0, 6.0))
    lo = float(min(np.min(sth_meas_all), np.min(sth_pred_all)))
    hi = float(max(np.max(sth_meas_all), np.max(sth_pred_all)))
    ax.scatter(sth_meas_all[~used_mask], sth_pred_all[~used_mask], s=34, alpha=0.55, color="#f1a3a3", edgecolors="none", label="Not used in fit")
    ax.scatter(sth_meas_all[used_mask], sth_pred_all[used_mask], s=40, alpha=0.82, color="#d62728", edgecolors="none", label="Used in fit")
    ax.plot([lo, hi], [lo, hi], color="#111111", lw=1.5, ls="--")
    ax.set_xlabel("Actual propagated std dev theta (deg)")
    ax.set_ylabel("Predicted std dev theta (deg)")
    ax.set_title("25nm rods filtered, r>0.25 fit: predicted vs actual theta std")
    ax.grid(True, alpha=0.22)
    ax.legend(loc="best", frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / "theta_predicted_vs_actual.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.0, 6.0))
    lo = float(min(np.min(sphi_meas_all), np.min(sphi_pred_all)))
    hi = float(max(np.max(sphi_meas_all), np.max(sphi_pred_all)))
    ax.scatter(sphi_meas_all[~used_mask], sphi_pred_all[~used_mask], s=34, alpha=0.55, color="#f7c38e", edgecolors="none", label="Not used in fit")
    ax.scatter(sphi_meas_all[used_mask], sphi_pred_all[used_mask], s=40, alpha=0.82, color="#ff7f0e", edgecolors="none", label="Used in fit")
    ax.plot([lo, hi], [lo, hi], color="#111111", lw=1.5, ls="--")
    ax.set_xlabel("Actual propagated std dev phi (deg)")
    ax.set_ylabel("Predicted std dev phi (deg)")
    ax.set_title("25nm rods filtered, r>0.25 fit: predicted vs actual phi std")
    ax.grid(True, alpha=0.22)
    ax.legend(loc="best", frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / "phi_predicted_vs_actual.png", dpi=220)
    plt.close(fig)

    summary = {
        "good_dir": str((Path.cwd() / GOOD_DIR).resolve()),
        "output_dir": str(out_dir.resolve()),
        "curve_csv": str((Path.cwd() / CURVE_CSV).resolve()),
        "n_rods_total_scored": len(data_all),
        "n_rods_used_in_fit": len(data_fit),
        "r_mean_min_fit": R_MIN_FIT,
        "curve_r_stretch": {
            "dataset_r_max": float(max(d.r_mean for d in data_all)),
            "empirical_r_max": float(np.max(curve["r"])),
            "stretch_scale": float(stretch_scale),
        },
        "fit_model": "I(theta) = a sin^2(theta) + b cos^2(theta), sigma_xy(theta)^2 = c / I(theta)^2",
        "ignores_theta_r_curve_uncertainty": True,
        "params": {
            "a_sin2": fit["a_sin2"],
            "b_cos2": fit["b_cos2"],
            "c_over_I2": fit["c_over_i2"],
        },
        "initial_guess": fit["initial_guess"],
        "rmse_joint": fit["rmse_joint"],
        "rmse_theta_deg": fit["rmse_theta_deg"],
        "rmse_phi_deg": fit["rmse_phi_deg"],
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
