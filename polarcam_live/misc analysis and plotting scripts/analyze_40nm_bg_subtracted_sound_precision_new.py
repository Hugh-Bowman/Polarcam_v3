from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares, lsq_linear


DATASETS = {
    "0.6 ms": Path(r"C:\Polarcam Software\Polarcam_v3\polarcam_live\datasets\attempt240nmsoundonbuffer\pending"),
    "1.2 ms": Path(r"C:\Polarcam Software\Polarcam_v3\polarcam_live\datasets\40nm stationary buffer sound on\precision 12ms"),
}
OUTPUT_DIR = Path(
    r"C:\Polarcam Software\Polarcam_v3\polarcam_live\datasets\40nm stationary buffer sound on\precision_error_model_bg_subtracted_attempt240"
)

# Fourkas finite-NA water/buffer curve from the GUI angle reconstruction model.
FOURKAS_WATER = {
    "label": "Fourkas finite-NA water/buffer",
    "J1": 0.65235,
    "J2": 0.03744,
    "J3": 0.10765,
    "r_max": 0.8914452224590091,
}

THETA_FIT_MIN_DEG = 0.0
PLOT_THETA_MIN_DEG = 0.0
THETA_BIN_WIDTH_DEG = 5.0
TOP_N_PER_BIN = 1
HIGH_ANGLE_TOP_N_PER_BIN = 1
VERY_HIGH_ANGLE_FILTER_START_DEG = 80.0
VERY_HIGH_ANGLE_TOP_N_PER_BIN = 3
EXTRA_SELECTION_FRACTION = 0.5
HIGH_ANGLE_FILTER_START_DEG = 75.0
MODEL_ERROR_PLOT_MAX_DEG = 10.0
R_SIGMA_LO_PCT = 16.0
R_SIGMA_HI_PCT = 84.0
R_SIGMA_DIVISOR = 2.0


@dataclass
class RodDatum:
    label: str
    rod: str
    path: Path
    n_frames: int
    theta_bin_deg: int
    r_mean: float
    sigma_r: float
    r_p16: float
    r_p84: float
    sigma_r_percentile: float
    phi_mean_deg: float
    sigma_xy: float
    theta_deg: float
    dtheta_dr_rad: float
    r_clipped_to_90: bool
    theta_from_r_p16_deg: float
    theta_from_r_p84_deg: float
    sigma_theta_deg: float
    sigma_phi_deg: float
    intensity_mean: float
    background_subtracted: bool
    background_profile_path: str
    selected_for_theta: bool = False
    selected_for_phi: bool = False
    used_for_theta_fit: bool = False
    used_for_phi_fit: bool = False


def fourkas_r_from_theta(theta_rad: np.ndarray) -> np.ndarray:
    j1 = float(FOURKAS_WATER["J1"])
    j2 = float(FOURKAS_WATER["J2"])
    j3 = float(FOURKAS_WATER["J3"])
    a = j1 - j2
    b = j1 + j2
    s2 = np.sin(theta_rad) ** 2
    c2 = np.cos(theta_rad) ** 2
    return (a * s2) / np.maximum((b * s2) + (2.0 * j3 * c2), 1e-15)


def build_curve() -> dict[str, np.ndarray]:
    theta_deg = np.linspace(0.0, 89.999, 30000)
    theta_rad = np.radians(theta_deg)
    r = fourkas_r_from_theta(theta_rad)
    dtheta_dr_rad = np.gradient(theta_rad, r)
    return {"theta_deg": theta_deg, "theta_rad": theta_rad, "r": r, "dtheta_dr_rad": dtheta_dr_rad}


def theta_from_r(curve: dict[str, np.ndarray], r_value: float) -> tuple[float, float, bool] | None:
    r_max = float(FOURKAS_WATER["r_max"])
    if not np.isfinite(r_value) or r_value <= 0.0:
        return None
    if r_value >= r_max:
        return 90.0, float(curve["dtheta_dr_rad"][-1]), True
    theta = float(np.interp(r_value, curve["r"], curve["theta_deg"]))
    dtheta = float(np.interp(r_value, curve["r"], curve["dtheta_dr_rad"]))
    return theta, dtheta, False


def strip_phase_marker(arr: np.ndarray, meta: dict) -> np.ndarray:
    if arr.shape[0] <= 1:
        return arr
    if bool(meta.get("actual", {}).get("phase_marker_appended", False)):
        return arr[:-1]
    if bool(meta.get("phase_marker_appended", False)):
        return arr[:-1]
    return arr


def raw_square_bounds(frame_shape: tuple[int, int], roi_meta: dict) -> tuple[int, int, int, int]:
    h, w = int(frame_shape[0]), int(frame_shape[1])
    win_raw = int(roi_meta.get("win_raw", roi_meta.get("w", min(h, w))))
    win_raw = max(2, win_raw)
    if win_raw % 2:
        win_raw -= 1
    cx = (w - 1) / 2.0
    cy = (h - 1) / 2.0
    try:
        cx = float(roi_meta["cx"]) - float(roi_meta["x"])
        cy = float(roi_meta["cy"]) - float(roi_meta["y"])
    except Exception:
        pass
    x0 = int(round(cx)) - (win_raw // 2)
    y0 = int(round(cy)) - (win_raw // 2)
    x0 = max(0, min(w - win_raw, x0))
    y0 = max(0, min(h - win_raw, y0))
    return x0, y0, x0 + win_raw, y0 + win_raw


def mean_raw_square_intensity(npy_path: Path, meta: dict) -> float:
    arr = np.load(npy_path, mmap_mode="r", allow_pickle=False)
    arr = strip_phase_marker(arr, meta)
    roi_meta = dict(meta.get("actual", {}).get("roi") or meta.get("roi") or {})
    vals = []
    for frame in arr:
        g = np.asarray(frame)
        if g.ndim != 2:
            g = g[..., 0]
        if roi_meta:
            x0, y0, x1, y1 = raw_square_bounds((g.shape[0], g.shape[1]), roi_meta)
            win = np.asarray(g[y0:y1, x0:x1], dtype=np.float64)
        else:
            win = np.asarray(g, dtype=np.float64)
        if win.size:
            vals.append(float(np.mean(win)))
    return float(np.mean(vals)) if vals else float("nan")


def load_xy(meta: dict, meta_path: Path) -> np.ndarray:
    xy = meta.get("xy_series")
    if not isinstance(xy, list):
        raise ValueError(f"No xy_series in {meta_path}")
    arr = np.asarray(xy, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] < 2:
        raise ValueError(f"Bad xy_series shape {arr.shape} in {meta_path}")
    arr = arr[:, :2]
    target_frames = meta.get("requested", {}).get("target_frames")
    actual_frames = meta.get("actual", {}).get("frames")
    try:
        expected = int(actual_frames if actual_frames is not None else target_frames)
    except Exception:
        expected = None
    if bool(meta.get("actual", {}).get("phase_marker_appended", False)) and expected is not None:
        arr = arr[:expected]
    elif bool(meta.get("phase_marker_appended", False)) and arr.shape[0] > 1:
        arr = arr[:-1]
    ok = np.isfinite(arr[:, 0]) & np.isfinite(arr[:, 1])
    return arr[ok]


def theta_series_from_r(curve: dict[str, np.ndarray], r: np.ndarray) -> np.ndarray:
    r_arr = np.asarray(r, dtype=np.float64)
    r_max = float(FOURKAS_WATER["r_max"])
    theta = np.full(r_arr.shape, np.nan, dtype=np.float64)
    finite = np.isfinite(r_arr) & (r_arr > 0.0)
    high = finite & (r_arr >= r_max)
    mid = finite & (r_arr < r_max)
    theta[high] = 90.0
    theta[mid] = np.interp(r_arr[mid], curve["r"], curve["theta_deg"])
    return theta


def theta_value_from_r(curve: dict[str, np.ndarray], r_value: float) -> float:
    arr = theta_series_from_r(curve, np.asarray([r_value], dtype=np.float64))
    return float(arr[0])


def circular_phi_stats_from_xy(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    phi = 0.5 * np.arctan2(y, x)
    z = np.exp(2.0j * phi)
    z_mean = np.mean(z)
    mean_phi = 0.5 * float(np.angle(z_mean))
    resultant = float(np.clip(np.abs(z_mean), 1e-12, 1.0))
    sigma_phi = 0.5 * math.sqrt(max(0.0, -2.0 * math.log(resultant)))
    return float(np.degrees(mean_phi)), float(np.degrees(sigma_phi))


def find_rod_dirs(root: Path) -> list[Path]:
    if not root.exists():
        return []
    out = []
    for meta_path in root.rglob("capture_maxfps_15x15_meta.json"):
        if (meta_path.parent / "capture_maxfps_15x15.npy").exists():
            out.append(meta_path.parent)
    return sorted(set(out))


def load_dataset(label: str, root: Path, curve: dict[str, np.ndarray]) -> list[RodDatum]:
    rows: list[RodDatum] = []
    for rod_dir in find_rod_dirs(root):
        meta_path = rod_dir / "capture_maxfps_15x15_meta.json"
        npy_path = rod_dir / "capture_maxfps_15x15.npy"
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            xy = load_xy(meta, meta_path)
            if xy.shape[0] < 5:
                continue
            x = xy[:, 0]
            y = xy[:, 1]
            r = np.hypot(x, y)
            r_mean = float(np.mean(r))
            theta_pair = theta_from_r(curve, r_mean)
            if theta_pair is None:
                continue
            theta_deg, dtheta_dr_rad, clipped_to_90 = theta_pair
            sigma_r = float(np.std(r))
            r_p16 = float(np.percentile(r, R_SIGMA_LO_PCT))
            r_p84 = float(np.percentile(r, R_SIGMA_HI_PCT))
            sigma_r_percentile = float(max(0.0, (r_p84 - r_p16) / R_SIGMA_DIVISOR))
            theta_p16 = theta_value_from_r(curve, r_p16)
            theta_p84 = theta_value_from_r(curve, r_p84)
            sigma_theta_percentile = float(abs(theta_p84 - theta_p16) / R_SIGMA_DIVISOR)
            phi_mean_deg, sigma_phi_deg = circular_phi_stats_from_xy(x, y)
            sigma_xy = float(np.hypot(np.std(x), np.std(y)))
            intensity = mean_raw_square_intensity(npy_path, meta)
            if not np.isfinite(intensity) or intensity <= 0.0:
                continue
            background = dict(meta.get("background") or {})
            rows.append(
                RodDatum(
                    label=label,
                    rod=rod_dir.name,
                    path=rod_dir,
                    n_frames=int(xy.shape[0]),
                    theta_bin_deg=int(math.floor(theta_deg / THETA_BIN_WIDTH_DEG) * THETA_BIN_WIDTH_DEG),
                    r_mean=r_mean,
                    sigma_r=sigma_r,
                    r_p16=r_p16,
                    r_p84=r_p84,
                    sigma_r_percentile=sigma_r_percentile,
                    phi_mean_deg=phi_mean_deg,
                    sigma_xy=sigma_xy,
                    theta_deg=float(theta_deg),
                    dtheta_dr_rad=float(dtheta_dr_rad),
                    r_clipped_to_90=bool(clipped_to_90),
                    theta_from_r_p16_deg=float(theta_p16),
                    theta_from_r_p84_deg=float(theta_p84),
                    sigma_theta_deg=float(sigma_theta_percentile),
                    sigma_phi_deg=float(sigma_phi_deg),
                    intensity_mean=float(intensity),
                    background_subtracted=bool(background.get("background_subtracted", meta.get("actual", {}).get("background_subtracted", False))),
                    background_profile_path=str(background.get("background_profile_path") or meta.get("actual", {}).get("background_profile_path") or ""),
                )
            )
        except Exception as exc:
            print(f"Skipping {rod_dir}: {exc}")
    return rows


def select_best_per_theta_bin(rows: list[RodDatum], metric: str, flag: str) -> list[RodDatum]:
    bins: dict[int, list[RodDatum]] = {}
    for row in rows:
        bins.setdefault(row.theta_bin_deg, []).append(row)
    selected: list[RodDatum] = []
    second_pass: list[RodDatum] = []
    for b in sorted(bins):
        if float(b) >= VERY_HIGH_ANGLE_FILTER_START_DEG:
            top_n = VERY_HIGH_ANGLE_TOP_N_PER_BIN
        elif float(b) >= HIGH_ANGLE_FILTER_START_DEG:
            top_n = HIGH_ANGLE_TOP_N_PER_BIN
        else:
            top_n = TOP_N_PER_BIN
        if 0.0 <= float(b) < 10.0:
            top_n = max(top_n, 3)
        ordered = sorted(bins[b], key=lambda r: (float(getattr(r, metric)), r.sigma_xy, r.rod))
        for row in ordered[:top_n]:
            setattr(row, flag, True)
            selected.append(row)
        if EXTRA_SELECTION_FRACTION > 0.0 and len(ordered) > top_n:
            second_pass.append(ordered[top_n])
    n_extra = int(round(EXTRA_SELECTION_FRACTION * len(selected)))
    for row in sorted(second_pass, key=lambda r: (float(getattr(r, metric)), r.sigma_xy, r.rod))[:n_extra]:
        setattr(row, flag, True)
        selected.append(row)
    return selected


def prune_theta_selection(label: str, rows: list[RodDatum]) -> tuple[list[RodDatum], list[RodDatum]]:
    kept = []
    removed = []
    for row in rows:
        if not np.isfinite(row.sigma_theta_deg) or float(row.sigma_theta_deg) <= 0.0:
            row.selected_for_theta = False
            removed.append(row)
        else:
            kept.append(row)
    return kept, removed


def prune_phi_selection(label: str, rows: list[RodDatum]) -> tuple[list[RodDatum], list[RodDatum]]:
    kept = []
    removed = []
    for row in rows:
        theta = float(row.theta_deg)
        sigma_phi = float(row.sigma_phi_deg)
        reject = False
        if theta < 10.0 and sigma_phi >= 10.0:
            reject = True
        if label == "0.6 ms" and theta < 10.0 and sigma_phi > 6.0:
            reject = True
        if label == "0.6 ms" and 70.0 < theta < 75.0 and sigma_phi > 0.9:
            reject = True
        if label == "0.6 ms" and 80.0 <= theta < 90.0 and sigma_phi > 1.0:
            reject = True
        if label == "1.2 ms" and 80.0 <= theta < 90.0 and sigma_phi > 1.0:
            reject = True
        if reject:
            row.selected_for_phi = False
            removed.append(row)
        else:
            kept.append(row)
    return kept, removed


def fit_intensity(rows: list[RodDatum]) -> tuple[float, float]:
    theta = np.radians(np.asarray([r.theta_deg for r in rows], dtype=np.float64))
    intensity = np.asarray([r.intensity_mean for r in rows], dtype=np.float64)
    design = np.column_stack([np.cos(theta) ** 2, np.sin(theta) ** 2])
    beta, *_ = np.linalg.lstsq(design, intensity, rcond=None)
    return float(max(beta[0], 1e-12)), float(max(beta[1], 1e-12))


def predict(
    theta_deg: np.ndarray,
    curve: dict[str, np.ndarray],
    b_cos2: float,
    c_sin2: float,
    a_var: float | None = None,
    noise_a_const: float = 0.0,
    noise_b_over_I: float = 0.0,
    noise_c_over_I2: float | None = None,
    sigma_xy_const: float | None = None,
) -> dict[str, np.ndarray]:
    theta_rad = np.radians(theta_deg)
    r = fourkas_r_from_theta(theta_rad)
    dtheta_dr = np.interp(theta_deg, curve["theta_deg"], curve["dtheta_dr_rad"])
    intensity = np.maximum(b_cos2 * np.cos(theta_rad) ** 2 + c_sin2 * np.sin(theta_rad) ** 2, 1e-12)
    if sigma_xy_const is not None:
        sigma_xy = np.full(theta_rad.shape, max(float(sigma_xy_const), 0.0), dtype=np.float64)
    else:
        if noise_c_over_I2 is None:
            noise_c_over_I2 = 0.0 if a_var is None else float(a_var)
        var_xy = (
            max(float(noise_a_const), 0.0)
            + max(float(noise_b_over_I), 0.0) / intensity
            + max(float(noise_c_over_I2), 0.0) / (intensity * intensity)
        )
        sigma_xy = np.sqrt(np.maximum(var_xy, 0.0))
    return {
        "r": r,
        "intensity": intensity,
        "sigma_xy": sigma_xy,
        "sigma_theta_deg": np.degrees(np.abs(dtheta_dr) * sigma_xy),
        "sigma_phi_deg": np.degrees(sigma_xy / np.maximum(2.0 * r, 1e-12)),
    }


def unique_rows(rows: list[RodDatum]) -> list[RodDatum]:
    seen = set()
    out = []
    for row in rows:
        if row.path in seen:
            continue
        seen.add(row.path)
        out.append(row)
    return out


def predict_from_params(theta_deg: np.ndarray, curve: dict[str, np.ndarray], params: dict) -> dict[str, np.ndarray]:
    return predict(
        theta_deg,
        curve,
        params["b_cos2_intensity"],
        params["c_sin2_intensity"],
        sigma_xy_const=params.get("sigma_xy_const"),
        noise_a_const=params.get("noise_a_const", 0.0),
        noise_b_over_I=params.get("noise_b_over_I", 0.0),
        noise_c_over_I2=params.get("noise_c_over_I2", params.get("a_var_xy_over_I2", 0.0)),
    )


def fit_variance(
    rows: list[RodDatum],
    curve: dict[str, np.ndarray],
    b_cos2: float,
    c_sin2: float,
    kind: str,
) -> float:
    theta = np.asarray([r.theta_deg for r in rows], dtype=np.float64)
    pred_unit = predict(theta, curve, b_cos2, c_sin2, a_var=1.0)
    if kind == "theta":
        base = np.asarray(pred_unit["sigma_theta_deg"], dtype=np.float64)
        target = np.asarray([r.sigma_theta_deg for r in rows], dtype=np.float64)
    elif kind == "phi":
        base = np.asarray(pred_unit["sigma_phi_deg"], dtype=np.float64)
        target = np.asarray([r.sigma_phi_deg for r in rows], dtype=np.float64)
    else:
        raise ValueError(f"Unknown fit kind: {kind}")
    ok = np.isfinite(base) & np.isfinite(target) & (base > 0.0) & (target > 0.0)
    # Fit the model in fractional-error space so large-uncertainty points do not dominate.
    ratio = base[ok] / target[ok]
    scale = float(np.sum(ratio) / max(np.sum(ratio * ratio), 1e-30))
    return float(max(scale * scale, 0.0))


def fit_shared_noise_model(
    theta_rows: list[RodDatum],
    phi_rows: list[RodDatum],
    curve: dict[str, np.ndarray],
    b_cos2: float,
    c_sin2: float,
) -> dict[str, float]:
    design_rows = []
    targets = []
    for row in theta_rows:
        if not np.isfinite(row.sigma_theta_deg) or row.sigma_theta_deg <= 0.0:
            continue
        if not np.isfinite(row.dtheta_dr_rad):
            continue
        theta_rad = math.radians(row.theta_deg)
        intensity = max(b_cos2 * math.cos(theta_rad) ** 2 + c_sin2 * math.sin(theta_rad) ** 2, 1e-12)
        factor = math.degrees(abs(row.dtheta_dr_rad))
        if factor <= 0.0:
            continue
        # Weighted relative residual on sigma is approximated as a relative residual on variance.
        design_rows.append([1.0, 1.0 / intensity, 1.0 / (intensity * intensity)])
        targets.append((row.sigma_theta_deg / factor) ** 2)
    for row in phi_rows:
        if not np.isfinite(row.sigma_phi_deg) or row.sigma_phi_deg <= 0.0:
            continue
        theta_rad = math.radians(row.theta_deg)
        intensity = max(b_cos2 * math.cos(theta_rad) ** 2 + c_sin2 * math.sin(theta_rad) ** 2, 1e-12)
        r = max(fourkas_r_from_theta(np.asarray([theta_rad], dtype=np.float64))[0], 1e-12)
        factor = math.degrees(1.0 / (2.0 * r))
        design_rows.append([1.0, 1.0 / intensity, 1.0 / (intensity * intensity)])
        targets.append((row.sigma_phi_deg / factor) ** 2)
    design = np.asarray(design_rows, dtype=np.float64)
    target = np.asarray(targets, dtype=np.float64)
    ok = np.all(np.isfinite(design), axis=1) & np.isfinite(target) & (target > 0.0)
    if np.count_nonzero(ok) < 3:
        raise ValueError("Too few finite points for shared noise model fit")
    design = design[ok]
    target = target[ok]
    weights = 1.0 / np.maximum(target, 1e-30)
    weighted_design = design * weights[:, None]
    weighted_target = target * weights
    fit = lsq_linear(weighted_design, weighted_target, bounds=(0.0, np.inf), method="trf")
    a_const, b_over_i, c_over_i2 = [float(max(v, 0.0)) for v in fit.x]
    return {
        "noise_a_const": a_const,
        "noise_b_over_I": b_over_i,
        "noise_c_over_I2": c_over_i2,
        "noise_model": "var_xy = a + b/I(theta) + c/I(theta)^2",
        "noise_fit_objective": "non-negative weighted least squares on inferred var_xy with weight 1/var_xy_data, approximating fractional sigma residuals",
    }


def fit_constant_sigma_xy(
    theta_rows: list[RodDatum],
    phi_rows: list[RodDatum],
    curve: dict[str, np.ndarray],
) -> dict[str, float | str]:
    bases = []
    targets = []
    for row in theta_rows:
        if not np.isfinite(row.sigma_theta_deg) or row.sigma_theta_deg <= 0.0:
            continue
        if not np.isfinite(row.dtheta_dr_rad):
            continue
        base = math.degrees(abs(row.dtheta_dr_rad))
        if base <= 0.0:
            continue
        bases.append(base)
        targets.append(float(row.sigma_theta_deg))
    for row in phi_rows:
        if not np.isfinite(row.sigma_phi_deg) or row.sigma_phi_deg <= 0.0:
            continue
        theta_rad = math.radians(row.theta_deg)
        r = max(float(fourkas_r_from_theta(np.asarray([theta_rad], dtype=np.float64))[0]), 1e-12)
        bases.append(math.degrees(1.0 / (2.0 * r)))
        targets.append(float(row.sigma_phi_deg))
    base = np.asarray(bases, dtype=np.float64)
    target = np.asarray(targets, dtype=np.float64)
    ok = np.isfinite(base) & np.isfinite(target) & (base > 0.0) & (target > 0.0)
    if np.count_nonzero(ok) < 1:
        raise ValueError("Too few finite points for constant sigma_xy fit")
    def residual(p: np.ndarray) -> np.ndarray:
        return (float(p[0]) * base[ok]) - target[ok]

    fit = least_squares(residual, x0=np.asarray([0.015], dtype=np.float64), bounds=(0.0, np.inf))
    sigma_xy = float(fit.x[0])
    return {
        "sigma_xy_const": max(sigma_xy, 0.0),
        "sigma_xy_model": "sigma_xy = constant",
        "sigma_xy_initial_guess": 0.015,
        "sigma_xy_fit_objective": "joint theta/phi fit minimizing absolute sigma residuals, (model - data)^2",
    }


def rows_to_dicts(rows: list[RodDatum]) -> list[dict]:
    return [
        {
            "label": r.label,
            "rod": r.rod,
            "path": str(r.path),
            "n_frames": r.n_frames,
            "theta_bin_deg": r.theta_bin_deg,
            "selected_for_theta": r.selected_for_theta,
            "selected_for_phi": r.selected_for_phi,
            "used_for_theta_fit": r.used_for_theta_fit,
            "used_for_phi_fit": r.used_for_phi_fit,
            "theta_deg": r.theta_deg,
            "r_mean": r.r_mean,
            "r_clipped_to_90": r.r_clipped_to_90,
            "sigma_r": r.sigma_r,
            "r_p16": r.r_p16,
            "r_p84": r.r_p84,
            "sigma_r_percentile_half_p84_minus_p16": r.sigma_r_percentile,
            "theta_from_r_p16_deg": r.theta_from_r_p16_deg,
            "theta_from_r_p84_deg": r.theta_from_r_p84_deg,
            "phi_mean_deg": r.phi_mean_deg,
            "sigma_xy": r.sigma_xy,
            "sigma_theta_deg": r.sigma_theta_deg,
            "sigma_phi_deg": r.sigma_phi_deg,
            "intensity_mean_raw_square": r.intensity_mean,
            "background_subtracted": r.background_subtracted,
            "background_profile_path": r.background_profile_path,
        }
        for r in rows
    ]


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def model_visible_until_10_deg(theta_grid: np.ndarray, model_y: np.ndarray) -> np.ndarray:
    y = np.asarray(model_y, dtype=np.float64)
    out = np.full(y.shape, np.nan, dtype=np.float64)
    ok = np.isfinite(y) & (y <= MODEL_ERROR_PLOT_MAX_DEG)
    if not np.any(ok):
        return out
    idx = np.where(ok)[0]
    first = int(idx[0])
    last = int(idx[-1])
    contiguous = ok[first : last + 1]
    if not np.all(contiguous):
        bad = np.where(~contiguous)[0]
        if bad.size:
            last = first + int(bad[0]) - 1
    out[first : last + 1] = y[first : last + 1]
    return out


def add_low_theta_divergence_region(ax: plt.Axes, label: bool = True) -> None:
    ax.axvspan(
        0.0,
        10.0,
        facecolor="#777777",
        alpha=0.08,
        hatch="///",
        edgecolor="#777777",
        linewidth=0.0,
        label=r"error diverges at low $\theta$" if label else None,
        zorder=0,
    )


def plot_single_kind(
    label: str,
    kind: str,
    selected: list[RodDatum],
    fit_rows: list[RodDatum],
    curve: dict[str, np.ndarray],
    params: dict,
    out_dir: Path,
) -> None:
    fit_set = {r.path for r in fit_rows}

    if kind == "theta":
        attr = "sigma_theta_deg"
        pred_key = "sigma_theta_deg"
        ylabel = "std theta (deg)"
        color = "#1f77b4"
    elif kind == "phi":
        attr = "sigma_phi_deg"
        pred_key = "sigma_phi_deg"
        ylabel = "std phi (deg)"
        color = "#2ca02c"
    else:
        raise ValueError(f"Unknown plot kind: {kind}")

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    add_low_theta_divergence_region(ax)
    excluded = [r for r in selected if r.path not in fit_set]
    if excluded:
        ax.scatter(
            [r.theta_deg for r in excluded],
            [getattr(r, attr) for r in excluded],
            s=34,
            color="#999999",
            alpha=0.45,
            edgecolors="none",
            label="selected, excluded from fit",
        )
    ax.scatter(
        [r.theta_deg for r in fit_rows],
        [getattr(r, attr) for r in fit_rows],
        s=42,
        color=color,
        alpha=0.88,
        edgecolors="none",
        label="used for fit",
    )
    plotted_y = np.asarray([getattr(r, attr) for r in fit_rows], dtype=np.float64)
    finite_y = plotted_y[np.isfinite(plotted_y)]
    y_cap = float(np.max(finite_y) * 1.12) if finite_y.size else MODEL_ERROR_PLOT_MAX_DEG
    if not np.isfinite(y_cap) or y_cap <= 0.0:
        y_cap = 1.0
    if y_cap >= MODEL_ERROR_PLOT_MAX_DEG:
        ax.axhline(MODEL_ERROR_PLOT_MAX_DEG, color="#666666", lw=0.8, ls="--", alpha=0.45)
    ax.set_xlim(PLOT_THETA_MIN_DEG, 90.0)
    ax.set_ylim(0.0, y_cap)
    ax.set_xlabel("theta (deg)")
    ax.set_ylabel(ylabel)
    ax.set_title(f"40nm rods {label}: {ylabel} vs theta")
    ax.grid(True, alpha=0.24)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / f"{label.replace(' ', '').replace('.', 'p')}_std_{kind}_vs_theta.png", dpi=220)
    plt.close(fig)

    if "sigma_xy_const" in params:
        return

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    theta_rad = np.radians(theta_grid)
    i_grid = params["b_cos2_intensity"] * np.cos(theta_rad) ** 2 + params["c_sin2_intensity"] * np.sin(theta_rad) ** 2
    excluded = [r for r in selected if r.path not in fit_set]
    if excluded:
        ax.scatter(
            [r.theta_deg for r in excluded],
            [r.intensity_mean for r in excluded],
            s=34,
            color="#999999",
            alpha=0.45,
            edgecolors="none",
            label="selected, excluded from fit",
        )
    ax.scatter(
        [r.theta_deg for r in fit_rows],
        [r.intensity_mean for r in fit_rows],
        s=42,
        color="#9467bd",
        alpha=0.88,
        edgecolors="none",
        label="used for fit",
    )
    ax.plot(theta_grid, i_grid, color="#111111", lw=2.0, label="fit: b cos^2(theta) + c sin^2(theta)")
    ax.set_xlabel("theta (deg)")
    ax.set_ylabel("mean background-subtracted raw-square intensity")
    ax.set_title(f"40nm rods {label}: intensity vs theta ({kind} subset)")
    ax.grid(True, alpha=0.24)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / f"{label.replace(' ', '').replace('.', 'p')}_{kind}_subset_intensity_vs_theta.png", dpi=220)
    plt.close(fig)


def plot_combined(results: dict[str, dict], curve: dict[str, np.ndarray], out_dir: Path) -> None:
    colors = {"0.6 ms": "#1f77b4", "1.2 ms": "#ff7f0e"}
    markers = {"0.6 ms": "o", "1.2 ms": "s"}
    for kind, attr, pred_key, ylabel, fname in [
        ("theta", "sigma_theta_deg", "sigma_theta_deg", "std theta (deg)", "combined_std_theta_vs_theta.png"),
        ("phi", "sigma_phi_deg", "sigma_phi_deg", "std phi (deg)", "combined_std_phi_vs_theta.png"),
    ]:
        fig, ax = plt.subplots(figsize=(7.5, 5.0))
        add_low_theta_divergence_region(ax)
        plotted_y_parts = []
        for label, res in results.items():
            rows = res[kind]["fit_rows"]
            color = colors.get(label, None)
            plotted_y_parts.append(np.asarray([getattr(r, attr) for r in rows], dtype=np.float64))
            ax.scatter(
                [r.theta_deg for r in rows],
                [getattr(r, attr) for r in rows],
                s=44,
                alpha=0.78,
                edgecolors="white",
                linewidths=0.45,
                color=color,
                marker=markers.get(label, "o"),
                label=f"{label} data",
            )
        plotted_y = np.concatenate([p for p in plotted_y_parts if p.size]) if plotted_y_parts else np.asarray([])
        finite_y = plotted_y[np.isfinite(plotted_y)]
        y_cap = float(np.max(finite_y) * 1.12) if finite_y.size else MODEL_ERROR_PLOT_MAX_DEG
        if not np.isfinite(y_cap) or y_cap <= 0.0:
            y_cap = 1.0
        if y_cap >= MODEL_ERROR_PLOT_MAX_DEG:
            ax.axhline(MODEL_ERROR_PLOT_MAX_DEG, color="#666666", lw=0.8, ls="--", alpha=0.45)
        ax.set_xlim(PLOT_THETA_MIN_DEG, 90.0)
        ax.set_ylim(0.0, y_cap)
        ax.set_xlabel("theta (deg)")
        ax.set_ylabel(ylabel)
        ax.set_title(f"40nm rods: {ylabel} vs theta")
        ax.grid(True, alpha=0.24)
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(out_dir / fname, dpi=220)
        plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    curve = build_curve()
    results: dict[str, dict] = {}
    summaries = []
    for label, root in DATASETS.items():
        rows = load_dataset(label, root, curve)
        selected_theta = select_best_per_theta_bin(rows, metric="sigma_r_percentile", flag="selected_for_theta")
        selected_phi = select_best_per_theta_bin(rows, metric="sigma_phi_deg", flag="selected_for_phi")
        selected_theta, theta_pruned = prune_theta_selection(label, selected_theta)
        selected_phi, phi_pruned = prune_phi_selection(label, selected_phi)
        fit_rows_theta = [
            r
            for r in selected_theta
            if THETA_FIT_MIN_DEG <= float(r.theta_deg) <= 90.0
            and np.isfinite(r.sigma_theta_deg)
        ]
        fit_rows_phi = [
            r
            for r in selected_phi
            if THETA_FIT_MIN_DEG <= float(r.theta_deg) <= 90.0
            and np.isfinite(r.sigma_phi_deg)
        ]
        for row in fit_rows_theta:
            row.used_for_theta_fit = True
        for row in fit_rows_phi:
            row.used_for_phi_fit = True
        if len(fit_rows_theta) < 3:
            raise SystemExit(f"Too few theta fit rows for {label}: {len(fit_rows_theta)}")
        if len(fit_rows_phi) < 3:
            raise SystemExit(f"Too few phi fit rows for {label}: {len(fit_rows_phi)}")

        shared_noise_params = fit_constant_sigma_xy(fit_rows_theta, fit_rows_phi, curve)
        shared_params = {
            "b_cos2_intensity": 1.0,
            "c_sin2_intensity": 1.0,
            **shared_noise_params,
        }
        theta_params = dict(shared_params)
        phi_params = dict(shared_params)

        results[label] = {
            "all_rows": rows,
            "theta": {
                "selected": selected_theta,
                "fit_rows": fit_rows_theta,
                "fit_params": theta_params,
            },
            "phi": {
                "selected": selected_phi,
                "fit_rows": fit_rows_phi,
                "fit_params": phi_params,
            },
        }
        plot_single_kind(label, "theta", selected_theta, fit_rows_theta, curve, theta_params, OUTPUT_DIR)
        plot_single_kind(label, "phi", selected_phi, fit_rows_phi, curve, phi_params, OUTPUT_DIR)
        safe_label = label.replace(" ", "").replace(".", "p")
        write_csv(OUTPUT_DIR / f"{safe_label}_all_rods.csv", rows_to_dicts(rows))
        write_csv(OUTPUT_DIR / f"{safe_label}_theta_selected_rods.csv", rows_to_dicts(selected_theta))
        write_csv(OUTPUT_DIR / f"{safe_label}_theta_fit_rods.csv", rows_to_dicts(fit_rows_theta))
        write_csv(OUTPUT_DIR / f"{safe_label}_theta_pruned_rods.csv", rows_to_dicts(theta_pruned))
        write_csv(OUTPUT_DIR / f"{safe_label}_phi_selected_rods.csv", rows_to_dicts(selected_phi))
        write_csv(OUTPUT_DIR / f"{safe_label}_phi_fit_rods.csv", rows_to_dicts(fit_rows_phi))
        write_csv(OUTPUT_DIR / f"{safe_label}_phi_pruned_rods.csv", rows_to_dicts(phi_pruned))
        summary = {
            "label": label,
            "source_dir": str(root),
            "n_rods_loaded": len(rows),
            "theta_selection": {
                "n_selected_best_per_bin": len(selected_theta),
                "n_used_for_fit": len(fit_rows_theta),
                "selection": f"best {TOP_N_PER_BIN} rod per occupied {THETA_BIN_WIDTH_DEG:g} degree theta bin, plus the next-best rod from the best {EXTRA_SELECTION_FRACTION * 100:g}% of bins",
                "low_theta_selection": "for theta 0-9.999 deg can include up to 3 best rods if available",
                "pruned_after_selection": len(theta_pruned),
                "prune_rule": "remove theta points with non-finite or zero sigma_theta, typically clipped r>=r_max points at theta=90 with no measurable theta spread",
                "fit_params": theta_params,
            },
            "phi_selection": {
                "n_selected_best_per_bin": len(selected_phi),
                "n_used_for_fit": len(fit_rows_phi),
                "selection": f"best {TOP_N_PER_BIN} rod per occupied {THETA_BIN_WIDTH_DEG:g} degree theta bin, plus the next-best rod from the best {EXTRA_SELECTION_FRACTION * 100:g}% of bins",
                "low_theta_selection": "for phi only, theta 0-9.999 deg can include up to 3 best rods if available",
                "pruned_after_selection": len(phi_pruned),
                "prune_rule": "phi only: remove 0.6 ms point with 70<theta<75 and sigma_phi>0.9 deg; remove 0.6 ms and 1.2 ms points with 80<=theta<90 and sigma_phi>1 deg",
                "fit_params": phi_params,
            },
            "fit_theta_window_deg": [THETA_FIT_MIN_DEG, 90.0],
            "plot_theta_window_deg": [PLOT_THETA_MIN_DEG, 90.0],
            "high_angle_filter": f"from {HIGH_ANGLE_FILTER_START_DEG:g} deg upward, select top {HIGH_ANGLE_TOP_N_PER_BIN} rods per bin; from {VERY_HIGH_ANGLE_FILTER_START_DEG:g} deg upward, select top {VERY_HIGH_ANGLE_TOP_N_PER_BIN} rods per bin before adding the extra {EXTRA_SELECTION_FRACTION * 100:g}% second-pass rods",
            "model_line_display": "fitted model curves are not plotted; y-axis limits are set from the data points used in each plot",
            "theta_uncertainty_method": f"measured theta std = abs(theta(r_p{R_SIGMA_HI_PCT:g}) - theta(r_p{R_SIGMA_LO_PCT:g}))/{R_SIGMA_DIVISOR:g}; model theta std uses smooth local gradient propagation dtheta/dr",
            "sigma_model": "same per-exposure one-parameter sigma_xy = constant model used for theta and phi",
            "sigma_model_fit_objective": "fit sigma_xy jointly to theta and phi by minimizing absolute sigma residuals, (model - data)^2",
            "r_above_rmax_handling": "rods with mean r >= Fourkas r_max are retained, plotted at theta=90 deg, and included in theta/phi fitting",
            "phase_marker_handling": "NPY marker frame is stripped for intensity; xy_series is limited to the real frame count when phase_marker_appended=true",
            "theta_curve": FOURKAS_WATER,
            "background_subtraction": "No additional subtraction applied; files report background_subtracted=true.",
            "largest_mean_r_loaded": max((r.r_mean for r in rows), default=float("nan")),
        }
        summaries.append(summary)
        print(
            f"{label}: loaded={len(rows)} "
            f"theta_selected={len(selected_theta)} theta_fit={len(fit_rows_theta)} "
            f"theta_pruned={len(theta_pruned)} "
            f"phi_selected={len(selected_phi)} phi_fit={len(fit_rows_phi)} "
            f"phi_pruned={len(phi_pruned)} "
            f"max_r={summary['largest_mean_r_loaded']:.6g} "
            f"sigma_xy={shared_noise_params['sigma_xy_const']:.6g}"
        )
    plot_combined(results, curve, OUTPUT_DIR)
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    print(f"Output: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
