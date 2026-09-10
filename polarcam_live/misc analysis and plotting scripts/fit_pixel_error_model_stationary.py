from __future__ import annotations

import csv
import itertools
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    import matplotlib.pyplot as plt
except Exception as exc:  # pragma: no cover
    raise RuntimeError(f"matplotlib is required to run this script: {exc}")

try:
    from scipy.optimize import nnls  # type: ignore
except Exception:
    nnls = None

try:
    from scipy.optimize import least_squares  # type: ignore
except Exception:
    least_squares = None


GOOD_DIR = Path("stationary rod data 01072026") / "good"
CURVE_CSV = (
    Path("glycerol suspended rods 17062026")
    / "plots"
    / "good_vs_good_plus_previous_used_center015"
    / "combined_phi_uniform_optimized"
    / "theta_vs_r_uncertainty_band.csv"
)
OUTPUT_DIR = Path("stationary rod data 01072026") / "plots" / "pixel_error_model_fit"
TERM_ORDER = ("a", "b_over_I", "c_over_I2")


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
    intensity_model: float = float("nan")


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


def _load_curve(curve_csv: Path) -> dict[str, np.ndarray]:
    rows = list(csv.DictReader(curve_csv.open("r", encoding="utf-8", newline="")))
    r = np.asarray([float(row["r"]) for row in rows], dtype=np.float64)
    theta_deg = np.asarray([float(row["theta_deg_center"]) for row in rows], dtype=np.float64)
    theta_rad = np.radians(theta_deg)
    dtheta_dr_rad = np.gradient(theta_rad, r)
    return {
        "r": r,
        "theta_deg": theta_deg,
        "dtheta_dr_rad": dtheta_dr_rad,
    }


def _interp(arr_x: np.ndarray, arr_y: np.ndarray, x_new: float) -> float:
    return float(np.interp(float(x_new), arr_x, arr_y, left=float(arr_y[0]), right=float(arr_y[-1])))


def _estimate_intensity_mean(npy_path: Path) -> float:
    arr = np.load(npy_path)
    x = np.asarray(arr, dtype=np.float64)
    return float(np.mean(x))


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
        r_mean = float(np.mean(r))
        sigma_x = float(np.std(x))
        sigma_y = float(np.std(y))
        sigma_xy = float(np.hypot(sigma_x, sigma_y))
        intensity_mean = _estimate_intensity_mean(npy_path)
        if not np.isfinite(intensity_mean) or intensity_mean <= 0.0:
            continue
        out.append(
            RodDatum(
                rod=rod_dir.name,
                n_frames=int(xy.shape[0]),
                r_mean=r_mean,
                sigma_xy=sigma_xy,
                intensity_mean=intensity_mean,
            )
        )
    return out


def _apply_curve_to_data(data: list[RodDatum], curve: dict[str, np.ndarray]) -> None:
    for d in data:
        d.theta_deg = _interp(curve["r"], curve["theta_deg"], d.r_mean)
        d.dtheta_dr_rad = _interp(curve["r"], curve["dtheta_dr_rad"], d.r_mean)
        d.sigma_theta_xy_only_deg = float(np.degrees(abs(d.dtheta_dr_rad) * d.sigma_xy))
        d.sigma_phi_deg = float(np.degrees(d.sigma_xy / max(2.0 * d.r_mean, 1e-12)))


def _stretch_curve_for_dataset(curve: dict[str, np.ndarray], r_max_dataset: float) -> tuple[dict[str, np.ndarray], float]:
    r_max_empirical = float(np.max(curve["r"]))
    scale = float(r_max_dataset / max(r_max_empirical, 1e-12))
    stretched_r = curve["r"] * scale
    stretched_dtheta_dr_rad = curve["dtheta_dr_rad"] / max(scale, 1e-12)
    return (
        {
            "r": stretched_r,
            "theta_deg": curve["theta_deg"].copy(),
            "dtheta_dr_rad": stretched_dtheta_dr_rad,
        },
        scale,
    )


def _all_term_combinations() -> list[tuple[str, ...]]:
    combos: list[tuple[str, ...]] = []
    for k in range(1, len(TERM_ORDER) + 1):
        combos.extend(itertools.combinations(TERM_ORDER, k))
    return combos


def _select_intensity_outlier(data: list[RodDatum], theta_lo: float = 60.0, theta_hi: float = 70.0) -> RodDatum:
    cand = [d for d in data if float(theta_lo) <= d.theta_deg <= float(theta_hi)]
    if not cand:
        raise RuntimeError("No rods in requested theta window for outlier removal.")
    return max(cand, key=lambda d: d.intensity_mean)

def _pixel_variance_from_model(theta_deg: np.ndarray, inten_sin2: float, inten_cos2: float, coeffs: dict[str, float]) -> np.ndarray:
    theta_rad = np.radians(theta_deg)
    I_theta = (inten_sin2 * (np.sin(theta_rad) ** 2)) + (inten_cos2 * (np.cos(theta_rad) ** 2))
    I_theta = np.maximum(I_theta, 1e-12)
    var_xy = np.zeros_like(I_theta, dtype=np.float64)
    var_xy += float(coeffs.get("a", 0.0))
    var_xy += float(coeffs.get("b", 0.0)) / I_theta
    var_xy += float(coeffs.get("c", 0.0)) / (I_theta * I_theta)
    return var_xy


def _coeffs_from_terms(terms: tuple[str, ...], free_params: np.ndarray) -> dict[str, float]:
    coeffs = {"a": 0.0, "b": 0.0, "c": 0.0}
    for i, term in enumerate(terms, start=2):
        if term == "a":
            coeffs["a"] = float(free_params[i])
        elif term == "b_over_I":
            coeffs["b"] = float(free_params[i])
        elif term == "c_over_I2":
            coeffs["c"] = float(free_params[i])
        else:
            raise ValueError(f"Unknown term {term}")
    return coeffs


def _predict_from_params(data: list[RodDatum], inten_sin2: float, inten_cos2: float, coeffs: dict[str, float]) -> dict[str, np.ndarray]:
    theta_deg = np.asarray([d.theta_deg for d in data], dtype=np.float64)
    r_mean = np.asarray([d.r_mean for d in data], dtype=np.float64)
    dtheta_dr_deg = np.asarray([np.degrees(abs(d.dtheta_dr_rad)) for d in data], dtype=np.float64)
    I_model = (inten_sin2 * (np.sin(np.radians(theta_deg)) ** 2)) + (inten_cos2 * (np.cos(np.radians(theta_deg)) ** 2))
    I_model = np.maximum(I_model, 1e-12)
    var_xy = _pixel_variance_from_model(theta_deg, inten_sin2, inten_cos2, coeffs)
    sigma_xy = np.sqrt(np.maximum(var_xy, 0.0))
    sigma_theta = dtheta_dr_deg * sigma_xy
    sigma_phi = np.degrees(sigma_xy / np.maximum(2.0 * r_mean, 1e-12))
    return {
        "intensity_model": I_model,
        "var_xy": var_xy,
        "sigma_xy": sigma_xy,
        "sigma_theta": sigma_theta,
        "sigma_phi": sigma_phi,
    }


def _predict_from_actual_intensity(data: list[RodDatum], coeffs: dict[str, float]) -> dict[str, np.ndarray]:
    theta_deg = np.asarray([d.theta_deg for d in data], dtype=np.float64)
    r_mean = np.asarray([d.r_mean for d in data], dtype=np.float64)
    dtheta_dr_deg = np.asarray([np.degrees(abs(d.dtheta_dr_rad)) for d in data], dtype=np.float64)
    intensity_actual = np.asarray([d.intensity_mean for d in data], dtype=np.float64)
    intensity_actual = np.maximum(intensity_actual, 1e-12)
    var_xy = np.zeros_like(intensity_actual, dtype=np.float64)
    var_xy += float(coeffs.get("a", 0.0))
    var_xy += float(coeffs.get("b", 0.0)) / intensity_actual
    var_xy += float(coeffs.get("c", 0.0)) / (intensity_actual * intensity_actual)
    sigma_xy = np.sqrt(np.maximum(var_xy, 0.0))
    sigma_theta = dtheta_dr_deg * sigma_xy
    sigma_phi = np.degrees(sigma_xy / np.maximum(2.0 * r_mean, 1e-12))
    return {
        "theta_deg": theta_deg,
        "intensity_actual": intensity_actual,
        "var_xy": var_xy,
        "sigma_xy": sigma_xy,
        "sigma_theta": sigma_theta,
        "sigma_phi": sigma_phi,
    }


def _initial_intensity_guess(data: list[RodDatum], outlier_rod: str) -> tuple[float, float]:
    fit_data = [d for d in data if d.rod != outlier_rod]
    th = np.radians(np.asarray([d.theta_deg for d in fit_data], dtype=np.float64))
    y = np.asarray([d.intensity_mean for d in fit_data], dtype=np.float64)
    X = np.column_stack([np.sin(th) ** 2, np.cos(th) ** 2])
    if nnls is not None:
        beta, _ = nnls(X, y)
    else:
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        beta = np.maximum(beta, 0.0)
    return float(beta[0]), float(beta[1])


def _fit_joint_model(data: list[RodDatum], terms: tuple[str, ...], init_intensity: tuple[float, float]) -> dict:
    if least_squares is None:
        raise RuntimeError("scipy.optimize.least_squares is required for the joint fit.")

    sigma_theta_meas = np.asarray([d.sigma_theta_xy_only_deg for d in data], dtype=np.float64)
    sigma_phi_meas = np.asarray([d.sigma_phi_deg for d in data], dtype=np.float64)
    x0 = np.asarray([max(init_intensity[0], 1e-6), max(init_intensity[1], 1e-6)] + [1e-3] * len(terms), dtype=np.float64)
    lb = np.zeros_like(x0)
    ub = np.full_like(x0, np.inf)

    def residuals(params: np.ndarray) -> np.ndarray:
        inten_sin2 = float(params[0])
        inten_cos2 = float(params[1])
        coeffs = _coeffs_from_terms(terms, params)
        pred = _predict_from_params(data, inten_sin2, inten_cos2, coeffs)
        return np.concatenate(
            [
                pred["sigma_theta"] - sigma_theta_meas,
                pred["sigma_phi"] - sigma_phi_meas,
            ]
        )

    result = least_squares(residuals, x0=x0, bounds=(lb, ub), method="trf")
    resid = residuals(result.x)
    rss = float(np.sum(resid * resid))
    n_obs = int(resid.size)
    rmse_joint = float(np.sqrt(np.mean(resid * resid)))
    pred = _predict_from_params(data, float(result.x[0]), float(result.x[1]), _coeffs_from_terms(terms, result.x))
    rmse_theta = float(np.sqrt(np.mean((pred["sigma_theta"] - sigma_theta_meas) ** 2)))
    rmse_phi = float(np.sqrt(np.mean((pred["sigma_phi"] - sigma_phi_meas) ** 2)))
    k = int(2 + len(terms))
    rss_safe = max(rss, 1e-30)
    aic = float(n_obs * math.log(rss_safe / n_obs) + (2.0 * k))
    bic = float(n_obs * math.log(rss_safe / n_obs) + (k * math.log(n_obs)))
    coeffs = _coeffs_from_terms(terms, result.x)
    return {
        "terms": ",".join(terms),
        "k": k,
        "rss": rss,
        "rmse_joint": rmse_joint,
        "rmse_theta_deg": rmse_theta,
        "rmse_phi_deg": rmse_phi,
        "aic": aic,
        "bic": bic,
        "intensity_a_sin2": float(result.x[0]),
        "intensity_b_cos2": float(result.x[1]),
        "a": float(coeffs["a"]),
        "b": float(coeffs["b"]),
        "c": float(coeffs["c"]),
    }


def main() -> None:
    good_dir = Path.cwd() / GOOD_DIR
    curve = _load_curve(Path.cwd() / CURVE_CSV)
    out_dir = Path.cwd() / OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    data = _load_data(good_dir)
    if not data:
        raise SystemExit("No usable rods found in the good folder.")
    r_max_dataset = float(max(d.r_mean for d in data))
    stretched_curve, r_stretch_scale = _stretch_curve_for_dataset(curve, r_max_dataset)
    _apply_curve_to_data(data, stretched_curve)

    intensity_outlier = _select_intensity_outlier(data, theta_lo=60.0, theta_hi=70.0)
    init_intensity = _initial_intensity_guess(data, intensity_outlier.rod)
    model_rows: list[dict] = []
    best_row: dict | None = None

    for terms in _all_term_combinations():
        row = _fit_joint_model(data, terms, init_intensity)
        model_rows.append(row)
        if best_row is None or row["bic"] < best_row["bic"]:
            best_row = row

    if best_row is None:
        raise RuntimeError("No models were fit.")

    best_terms = tuple(best_row["terms"].split(","))
    inten_a = float(best_row["intensity_a_sin2"])
    inten_b = float(best_row["intensity_b_cos2"])
    coeffs = {
        "a": float(best_row["a"]),
        "b": float(best_row["b"]),
        "c": float(best_row["c"]),
    }
    pred_best = _predict_from_params(data, inten_a, inten_b, coeffs)
    pred_actual_intensity = _predict_from_actual_intensity(data, coeffs)

    fitted_rows = []
    for datum, intensity_model, var_xy_pred, sigma_xy_pred, sigma_theta_pred_deg, sigma_phi_pred_deg, sigma_xy_pred_actual, sigma_theta_pred_actual, sigma_phi_pred_actual in zip(
        data,
        pred_best["intensity_model"],
        pred_best["var_xy"],
        pred_best["sigma_xy"],
        pred_best["sigma_theta"],
        pred_best["sigma_phi"],
        pred_actual_intensity["sigma_xy"],
        pred_actual_intensity["sigma_theta"],
        pred_actual_intensity["sigma_phi"],
    ):
        datum.intensity_model = float(intensity_model)
        fitted_rows.append(
            {
                "rod": datum.rod,
                "theta_deg": datum.theta_deg,
                "r_mean": datum.r_mean,
                "intensity_mean": datum.intensity_mean,
                "intensity_model": float(intensity_model),
                "sigma_xy_measured": datum.sigma_xy,
                "sigma_xy_pred": float(sigma_xy_pred),
                "sigma_xy_pred_actual_intensity": float(sigma_xy_pred_actual),
                "sigma_phi_measured_deg": datum.sigma_phi_deg,
                "sigma_phi_pred_deg": float(sigma_phi_pred_deg),
                "sigma_phi_pred_actual_intensity_deg": float(sigma_phi_pred_actual),
                "sigma_theta_measured_deg": datum.sigma_theta_xy_only_deg,
                "sigma_theta_pred_deg": float(sigma_theta_pred_deg),
                "sigma_theta_pred_actual_intensity_deg": float(sigma_theta_pred_actual),
                "sigma_xy2_pred": float(var_xy_pred),
            }
        )

    _write_csv(
        out_dir / "pixel_error_model_comparison.csv",
        sorted(model_rows, key=lambda r: r["bic"]),
        ["terms", "k", "rss", "rmse_joint", "rmse_theta_deg", "rmse_phi_deg", "aic", "bic", "intensity_a_sin2", "intensity_b_cos2", "a", "b", "c"],
    )
    _write_csv(
        out_dir / "pixel_error_model_fitted_points.csv",
        fitted_rows,
        [
            "rod",
            "theta_deg",
            "r_mean",
            "intensity_mean",
            "intensity_model",
            "sigma_xy_measured",
            "sigma_xy_pred",
            "sigma_xy_pred_actual_intensity",
            "sigma_phi_measured_deg",
            "sigma_phi_pred_deg",
            "sigma_phi_pred_actual_intensity_deg",
            "sigma_theta_measured_deg",
            "sigma_theta_pred_deg",
            "sigma_theta_pred_actual_intensity_deg",
            "sigma_xy2_pred",
        ],
    )

    th = np.asarray([r["theta_deg"] for r in fitted_rows], dtype=np.float64)
    theta_min_data = float(np.min(th))
    sphi_meas = np.asarray([r["sigma_phi_measured_deg"] for r in fitted_rows], dtype=np.float64)
    sphi_pred_actual = np.asarray([r["sigma_phi_pred_actual_intensity_deg"] for r in fitted_rows], dtype=np.float64)
    sth_meas = np.asarray([r["sigma_theta_measured_deg"] for r in fitted_rows], dtype=np.float64)
    sth_pred_model = np.asarray([r["sigma_theta_pred_deg"] for r in fitted_rows], dtype=np.float64)
    sth_pred_actual = np.asarray([r["sigma_theta_pred_actual_intensity_deg"] for r in fitted_rows], dtype=np.float64)
    sphi_pred_model = np.asarray([r["sigma_phi_pred_deg"] for r in fitted_rows], dtype=np.float64)
    I_model_pts = np.asarray([r["intensity_model"] for r in fitted_rows], dtype=np.float64)
    sxy2_meas = np.asarray([r["sigma_xy_measured"] ** 2 for r in fitted_rows], dtype=np.float64)

    theta_grid = np.linspace(float(np.min(stretched_curve["theta_deg"])), float(np.max(stretched_curve["theta_deg"])), 500)
    theta_curve = stretched_curve["theta_deg"]
    r_curve = stretched_curve["r"]
    dtheta_dr_deg_curve = np.degrees(np.abs(stretched_curve["dtheta_dr_rad"]))
    r_grid = np.interp(theta_grid, theta_curve, r_curve, left=float(r_curve[0]), right=float(r_curve[-1]))
    dtheta_dr_deg_grid = np.interp(
        theta_grid,
        theta_curve,
        dtheta_dr_deg_curve,
        left=float(dtheta_dr_deg_curve[0]),
        right=float(dtheta_dr_deg_curve[-1]),
    )
    var_xy_grid = _pixel_variance_from_model(theta_grid, inten_a, inten_b, coeffs)
    sigma_xy_grid = np.sqrt(np.maximum(var_xy_grid, 0.0))
    sigma_theta_grid = dtheta_dr_deg_grid * sigma_xy_grid
    sigma_phi_grid = np.degrees(sigma_xy_grid / np.maximum(2.0 * r_grid, 1e-12))
    I_grid = (inten_a * (np.sin(np.radians(theta_grid)) ** 2)) + (inten_b * (np.cos(np.radians(theta_grid)) ** 2))

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.scatter(th, sth_meas, s=40, alpha=0.85, color="#1f77b4", label="Measured propagated points")
    ax.plot(theta_grid, sigma_theta_grid, color="#d62728", lw=2.0, label=f"Predicted using I(theta) ({best_row['terms']})")
    ax.set_xlabel("theta (deg)")
    ax.set_ylabel("std dev theta (deg)")
    ax.set_title("Theta std vs theta")
    ax.grid(True, alpha=0.22)
    ax.legend(loc="best", frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / "theta_error_model_fit.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.scatter(th, sphi_meas, s=40, alpha=0.85, color="#2ca02c", label="Measured propagated points")
    valid_phi_model = theta_grid >= theta_min_data
    ax.plot(theta_grid[valid_phi_model], sigma_phi_grid[valid_phi_model], color="#ff7f0e", lw=2.0, label=f"Predicted using I(theta) ({best_row['terms']})")
    y_lo, y_hi = ax.get_ylim()
    ax.axvspan(float(theta_grid[0]), theta_min_data, facecolor="#bbbbbb", alpha=0.2, hatch="///", edgecolor="#888888")
    ax.text(
        float(theta_grid[0] + 0.45 * (theta_min_data - float(theta_grid[0]))),
        float(y_lo + 0.9 * (y_hi - y_lo)),
        "Model diverges",
        rotation=90,
        ha="center",
        va="top",
        color="#555555",
    )
    ax.set_xlabel("theta (deg)")
    ax.set_ylabel("std dev phi (deg)")
    ax.set_title("Phi std vs theta")
    ax.grid(True, alpha=0.22)
    ax.legend(loc="best", frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / "phi_error_model_fit.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.scatter(I_model_pts, sxy2_meas, s=40, alpha=0.85, color="#1f77b4", label="Measured sigma_xy^2")
    order_Im = np.argsort(I_grid)
    ax.plot(I_grid[order_Im], var_xy_grid[order_Im], color="#d62728", lw=2.0, label=f"Model {best_row['terms']}")
    ax.set_xlabel("Pixel intensity I")
    ax.set_ylabel("sigma_xy^2")
    ax.set_title("Pixel-level variance model fit")
    ax.grid(True, alpha=0.22)
    ax.legend(loc="best", frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / "pixel_variance_model_fit.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    th_fit = np.asarray([d.theta_deg for d in data if d.rod != intensity_outlier.rod], dtype=np.float64)
    I_fit = np.asarray([d.intensity_mean for d in data if d.rod != intensity_outlier.rod], dtype=np.float64)
    ax.scatter(th_fit, I_fit, s=40, alpha=0.88, color="#9467bd", label="Intensity data")
    ax.scatter([intensity_outlier.theta_deg], [intensity_outlier.intensity_mean], s=70, alpha=0.95, color="#d62728", label="Removed outlier")
    ax.plot(theta_grid, I_grid, color="#111111", lw=2.0, label=r"Fit: $a\sin^2\theta + b\cos^2\theta$")
    ax.set_xlabel("theta (deg)")
    ax.set_ylabel("Mean pixel intensity I")
    ax.set_title("Intensity vs theta")
    ax.grid(True, alpha=0.22)
    ax.legend(loc="best", frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / "intensity_vs_theta.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.0, 6.0))
    theta_all = np.concatenate([sth_meas, sth_pred_model, sth_pred_actual])
    theta_lo = float(np.min(theta_all))
    theta_hi = float(np.max(theta_all))
    ax.scatter(sth_meas, sth_pred_model, s=38, alpha=0.82, color="#d62728", label="Using I(theta)")
    ax.scatter(sth_meas, sth_pred_actual, s=38, alpha=0.78, color="#9467bd", label="Using rod intensity")
    ax.plot([theta_lo, theta_hi], [theta_lo, theta_hi], color="#111111", lw=1.5, ls="--")
    ax.set_xlabel("Measured propagated std dev theta (deg)")
    ax.set_ylabel("Predicted std dev theta (deg)")
    ax.set_title("Predicted vs measured theta std")
    ax.grid(True, alpha=0.22)
    ax.legend(loc="best", frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / "theta_predicted_vs_actual.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.0, 6.0))
    phi_all = np.concatenate([sphi_meas, sphi_pred_model, sphi_pred_actual])
    phi_lo = float(np.min(phi_all))
    phi_hi = float(np.max(phi_all))
    ax.scatter(sphi_meas, sphi_pred_model, s=38, alpha=0.82, color="#ff7f0e", label="Using I(theta)")
    ax.scatter(sphi_meas, sphi_pred_actual, s=38, alpha=0.78, color="#8c564b", label="Using rod intensity")
    ax.plot([phi_lo, phi_hi], [phi_lo, phi_hi], color="#111111", lw=1.5, ls="--")
    ax.set_xlabel("Measured propagated std dev phi (deg)")
    ax.set_ylabel("Predicted std dev phi (deg)")
    ax.set_title("Predicted vs measured phi std")
    ax.grid(True, alpha=0.22)
    ax.legend(loc="best", frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / "phi_predicted_vs_actual.png", dpi=220)
    plt.close(fig)

    summary = {
        "good_dir": str(good_dir),
        "curve_csv": str(Path.cwd() / CURVE_CSV),
        "output_dir": str(out_dir),
        "n_rods": int(len(data)),
        "curve_r_stretch": {
            "empirical_r_max": float(np.max(curve["r"])),
            "dataset_r_max": r_max_dataset,
            "stretch_scale": r_stretch_scale,
        },
        "intensity_model": "I(theta) = a sin^2(theta) + b cos^2(theta)",
        "intensity_model_params": {
            "a_sin2": float(inten_a),
            "b_cos2": float(inten_b),
        },
        "intensity_model_fit_mode": "joint with error model against sigma_theta(theta) and sigma_phi(theta)",
        "initial_intensity_guess": {
            "a_sin2": float(init_intensity[0]),
            "b_cos2": float(init_intensity[1]),
        },
        "intensity_fit_removed_outlier": {
            "rod": intensity_outlier.rod,
            "theta_deg": float(intensity_outlier.theta_deg),
            "intensity_mean": float(intensity_outlier.intensity_mean),
        },
        "fit_target": "joint sigma_theta(theta) and sigma_phi(theta)",
        "fit_model": "sigma_xy(theta)^2 = a + b/I_model(theta) + c/I_model(theta)^2, with sigma_theta and sigma_phi derived from sigma_xy",
        "pixel_variance_model": "var_pixel(theta) = a + b/I_model(theta) + c/I_model(theta)^2",
        "additional_outputs": [
            "theta_error_model_fit.png uses the stretched theta(r) calibration and the smooth I(theta) prediction",
            "phi_error_model_fit.png uses the stretched theta(r) calibration and the smooth I(theta) prediction",
            "theta_predicted_vs_actual.png compares predictions against the same measured propagated theta std values",
            "phi_predicted_vs_actual.png compares predictions against the same measured propagated phi std values",
        ],
        "best_model_by_bic": best_row,
        "all_models_sorted_by_bic": sorted(model_rows, key=lambda r: r["bic"]),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
