from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from matplotlib.patches import ConnectionPatch, Patch

try:
    import matplotlib.pyplot as plt
except Exception as exc:  # pragma: no cover
    raise RuntimeError(f"matplotlib is required to run this script: {exc}")

try:
    from scipy.optimize import least_squares  # type: ignore
except Exception:
    least_squares = None


# User-specified hole+fresnel conversion:
# theta(r) = asin(sqrt((0.1866*r)/(0.5577 - 0.4216*r))), valid for 0 <= r < 0.9170
THETA_A = 0.1866
THETA_B = 0.5577
THETA_C = 0.4216
THETA_R_MAX = 0.9170
THETA_THEORY_PLOT_MAX_DEG = 89.9
DATA_THETA_MIN_DEG = 10.0
DATA_THETA_MAX_DEG = 80.0
PHI_OUTLIER_HIGH_THETA_DEG = 40.0
PHI_OUTLIER_HIGH_ERR_DEG = 1.4
PHI_OUTLIER_LOW_THETA_DEG = 20.0
PHI_OUTLIER_LOW_ERR_DEG = 1.2
LOW_THETA_WEIGHT_MAX_DEG = 30.0
LOW_THETA_WEIGHT_FACTOR = 3.0
FORCE_BEST_FIXED_FIT = True
FORCE_SIGMA_LIGHT = 0.01226
FORCE_SIGMA_SOURCE_XY = 0.01358


@dataclass
class RodResult:
    rod_id: str
    rod_key: str
    mode: str
    n_frames: int
    theta_deg: float
    theta_err_deg: float
    phi_err_deg: float
    r_mean: float
    xy_var_score: float
    brightness_mean: float
    brightness_abs_norm: float
    npy_path: str


def _phi_main_x(theta_deg: float) -> float:
    # Reserve [7,10] display segment as compact marker for true [0,10] inset region.
    return float(theta_deg)


def _theta_main_x(theta_deg: float) -> float:
    # Compress true [80,90] into display [80,83] for compact right-side extension.
    t = float(theta_deg)
    if t <= 80.0:
        return t
    if t >= 90.0:
        return 83.0
    return 80.0 + (t - 80.0) * (3.0 / 10.0)


@dataclass
class ModeFit:
    mode: str
    a_param: float
    b_param: float
    sigma_bg: float
    k_intensity: float
    n_used: int


@dataclass
class IntensityThetaFit:
    i_const_77: float
    k_77: float
    i_const_11: float
    k_11: float
    n_used: int


@dataclass
class IntensityThetaErrorFit:
    a_param: float
    b_param: float
    sigma_bg: float
    n_used: int


@dataclass
class IntensityThetaExtremaFit:
    i0_77: float
    k_77: float
    i0_11: float
    k_11: float
    n_used: int


@dataclass
class SharedNoiseTwoKFit:
    a_40: float
    b_40: float
    a_25: float
    b_25: float
    sigma_bg: float
    k_line: np.ndarray
    n_like_line: np.ndarray
    n_used: int


def theta_hole_fresnel_from_r(r: np.ndarray) -> np.ndarray:
    r = np.asarray(r, dtype=np.float64)
    out = np.full(r.shape, np.nan, dtype=np.float64)
    finite = np.isfinite(r)
    sat = finite & (r >= THETA_R_MAX)
    if np.any(sat):
        out[sat] = 0.5 * np.pi

    valid = finite & (r >= 0.0) & (r < THETA_R_MAX)
    if not np.any(valid):
        return out

    rv = r[valid]
    den = THETA_B - (THETA_C * rv)
    ok = den > 0.0
    if not np.any(ok):
        return out

    ratio = np.full(rv.shape, np.nan, dtype=np.float64)
    ratio[ok] = (THETA_A * rv[ok]) / den[ok]
    ok2 = np.isfinite(ratio) & (ratio >= 0.0) & (ratio <= 1.0)
    if np.any(ok2):
        theta = np.arcsin(np.sqrt(ratio[ok2]))
        rv_out = np.full(rv.shape, np.nan, dtype=np.float64)
        rv_out[ok2] = theta
        out_idx = np.where(valid)[0]
        out[out_idx] = rv_out
    return out


def dtheta_dr_hole_fresnel(r: np.ndarray) -> np.ndarray:
    """
    d/dr of theta(r) with theta=asin(sqrt(q)), q=a*r/(b-c*r)
    dtheta/dr = (dq/dr) / (2*sqrt(q)*sqrt(1-q)) with dq/dr = a*b/(b-c*r)^2
    """
    r = np.asarray(r, dtype=np.float64)
    out = np.full(r.shape, np.nan, dtype=np.float64)

    den = THETA_B - (THETA_C * r)
    valid = np.isfinite(r) & np.isfinite(den) & (den > 0.0)
    if not np.any(valid):
        return out

    rv = r[valid]
    denv = den[valid]
    q = (THETA_A * rv) / denv
    dq = (THETA_A * THETA_B) / (denv * denv)

    ok = np.isfinite(q) & (q > 0.0) & (q < 1.0) & np.isfinite(dq)
    if not np.any(ok):
        return out

    deriv = np.full(rv.shape, np.nan, dtype=np.float64)
    deriv[ok] = dq[ok] / (2.0 * np.sqrt(q[ok]) * np.sqrt(1.0 - q[ok]))

    idx = np.where(valid)[0]
    out[idx] = deriv
    return out


def r_from_theta_hole_fresnel(theta_rad: np.ndarray) -> np.ndarray:
    """
    Inverse of theta(r) model:
      q = sin(theta)^2
      q = a*r/(b-c*r)  =>  r = b*q / (a + c*q)
    """
    th = np.asarray(theta_rad, dtype=np.float64)
    q = np.sin(th) ** 2
    den = THETA_A + (THETA_C * q)
    out = np.full(th.shape, np.nan, dtype=np.float64)
    ok = np.isfinite(q) & np.isfinite(den) & (den > 0.0)
    if np.any(ok):
        out[ok] = (THETA_B * q[ok]) / den[ok]
    return out


def robust_sigma(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size < 3:
        return float("nan")
    q16, q84 = np.percentile(x, [16.0, 84.0])
    return float(0.5 * (q84 - q16))


def circular_sigma_phi(phi: np.ndarray) -> tuple[float, float]:
    """
    Returns (phi_center_rad, phi_sigma_rad) for phi with pi-periodicity.

    Uses doubled-angle circular mean and wrapped residuals:
    - mu = 0.5*atan2(mean(sin(2phi)), mean(cos(2phi)))
    - residual = 0.5*angle(exp(i*2*(phi-mu)))
    - sigma = 0.5*(P84(residual)-P16(residual))
    """
    phi = np.asarray(phi, dtype=np.float64)
    phi = phi[np.isfinite(phi)]
    if phi.size < 3:
        return (float("nan"), float("nan"))

    s2 = np.sin(2.0 * phi)
    c2 = np.cos(2.0 * phi)
    mu = 0.5 * np.arctan2(np.mean(s2), np.mean(c2))

    resid = 0.5 * np.angle(np.exp(1j * 2.0 * (phi - mu)))
    sig = robust_sigma(resid)
    return (float(mu), float(sig))


def load_json(path: Path) -> Optional[dict]:
    try:
        if not path.exists():
            return None
        with path.open("r", encoding="utf-8") as f:
            d = json.load(f)
        return d if isinstance(d, dict) else None
    except Exception:
        return None


def roi_pixels_for_mode(mode: str) -> float:
    # maxfps mode uses 11x11 capture ROI; 77fps mode uses 15x15 capture ROI.
    if mode == "maxfps_11x11":
        return float(11 * 11)
    if mode == "77fps":
        return float(15 * 15)
    return float(11 * 11)


def mode_meta_path(rod_dir: Path, mode: str) -> Path:
    if mode == "77fps":
        return rod_dir / "capture_77fps_meta.json"
    return rod_dir / "capture_maxfps_11x11_meta.json"


def analyze_mode(rod_dir: Path, mode: str, min_frames: int) -> Optional[RodResult]:
    meta = load_json(mode_meta_path(rod_dir, mode))
    if not meta:
        return None

    xy_raw = meta.get("xy_series", [])
    if not isinstance(xy_raw, list):
        return None

    xy = []
    for v in xy_raw:
        if isinstance(v, (list, tuple)) and len(v) >= 2:
            try:
                x = float(v[0])
                y = float(v[1])
            except Exception:
                continue
            if np.isfinite(x) and np.isfinite(y):
                xy.append((x, y))

    if len(xy) < int(min_frames):
        return None

    arr = np.asarray(xy, dtype=np.float64)
    x = arr[:, 0]
    y = arr[:, 1]
    r = np.sqrt((x * x) + (y * y))
    sx = float(np.std(x))
    sy = float(np.std(y))
    xy_var_score = float(np.hypot(sx, sy))

    theta_rad = theta_hole_fresnel_from_r(r)
    theta_ok = np.isfinite(theta_rad)
    if int(np.count_nonzero(theta_ok)) < int(min_frames):
        return None

    theta_deg = np.degrees(theta_rad[theta_ok])
    theta_center_deg = float(np.median(theta_deg))
    theta_err_deg = float(robust_sigma(theta_deg))

    phi = 0.5 * np.arctan2(y, x)
    _, phi_err_rad = circular_sigma_phi(phi)
    phi_err_deg = float(np.degrees(phi_err_rad)) if np.isfinite(phi_err_rad) else float("nan")

    rod_meta = load_json(rod_dir / "meta.json") or {}
    rod_id = str(rod_meta.get("rod_id", rod_dir.name))
    theta_center_rad = float(np.radians(theta_center_deg))
    s2 = float(np.sin(theta_center_rad) ** 2)
    s2 = max(s2, 1e-6)
    brightness_mean = float("nan")
    npy_path: Optional[Path] = None
    try:
        npy_name = str(meta.get("npy_file", ""))
        npy_path = rod_dir / npy_name if npy_name else None
        if npy_path is not None and npy_path.exists():
            arr_stack = np.load(npy_path, mmap_mode="r", allow_pickle=False)
            if getattr(arr_stack, "size", 0) > 0:
                brightness_mean = float(np.mean(arr_stack))
    except Exception:
        brightness_mean = float("nan")
    brightness_abs_norm = (
        float(brightness_mean / s2) if np.isfinite(brightness_mean) else float("nan")
    )

    return RodResult(
        rod_id=rod_id,
        rod_key=rod_dir.name,
        mode=mode,
        n_frames=int(arr.shape[0]),
        theta_deg=theta_center_deg,
        theta_err_deg=theta_err_deg,
        phi_err_deg=phi_err_deg,
        r_mean=float(np.nanmean(r)),
        xy_var_score=xy_var_score,
        brightness_mean=brightness_mean,
        brightness_abs_norm=brightness_abs_norm,
        npy_path=str(npy_path) if (npy_path is not None and npy_path.exists()) else "",
    )


def fit_noise_model_for_mode(
    rows: list[RodResult],
    mode: Optional[str],
    theta_weight_bandwidth_deg: float,
    use_density_weight: bool = True,
) -> Optional[ModeFit]:
    sub = [
        r
        for r in rows
        if (mode is None or r.mode == mode)
        and np.isfinite(r.r_mean)
        and np.isfinite(r.theta_err_deg)
        and np.isfinite(r.phi_err_deg)
        and np.isfinite(r.theta_deg)
        and np.isfinite(r.brightness_mean)
        and (r.brightness_mean > 0.0)
    ]
    if len(sub) < 2:
        return None

    rr = np.asarray([r.r_mean for r in sub], dtype=np.float64)
    sig_theta = np.radians(np.asarray([r.theta_err_deg for r in sub], dtype=np.float64))
    sig_phi = np.radians(np.asarray([r.phi_err_deg for r in sub], dtype=np.float64))
    th_deg = np.asarray([r.theta_deg for r in sub], dtype=np.float64)
    th_rad = np.radians(th_deg)
    npix = np.asarray([roi_pixels_for_mode(r.mode) for r in sub], dtype=np.float64)

    ok = (
        np.isfinite(rr)
        & np.isfinite(sig_theta)
        & np.isfinite(sig_phi)
        & np.isfinite(th_deg)
        & (rr > 1e-6)
        & (sig_theta > 0.0)
        & (sig_phi > 0.0)
    )
    rr = rr[ok]
    sig_theta = sig_theta[ok]
    sig_phi = sig_phi[ok]
    th_deg = th_deg[ok]
    th_rad = th_rad[ok]
    if rr.size < 2:
        return None

    dth = dtheta_dr_hole_fresnel(rr)
    d2 = dth * dth
    ok_d = np.isfinite(d2) & (d2 > 0.0)
    rr = rr[ok_d]
    sig_theta = sig_theta[ok_d]
    sig_phi = sig_phi[ok_d]
    th_deg = th_deg[ok_d]
    th_rad = th_rad[ok_d]
    d2 = d2[ok_d]
    if rr.size < 2:
        return None

    y_theta = sig_theta * sig_theta
    y_phi = sig_phi * sig_phi

    if use_density_weight:
        # Theta-density weighting: inverse local density so crowded theta neighborhoods
        # do not dominate the fit.
        bw = max(1e-6, float(theta_weight_bandwidth_deg))
        d = (th_deg[:, None] - th_deg[None, :]) / bw
        local_density = np.sum(np.exp(-0.5 * d * d), axis=1)
        w_base = 1.0 / np.maximum(local_density, 1e-12)
        # Normalize weights to keep scale stable.
        w_base = w_base * (float(w_base.size) / float(np.sum(w_base)))
    else:
        w_base = np.ones(rr.shape, dtype=np.float64)

    # Extra emphasis for low-theta region where fits are currently weak.
    low_mask = np.isfinite(th_deg) & (th_deg >= 0.0) & (th_deg <= LOW_THETA_WEIGHT_MAX_DEG)
    if np.any(low_mask):
        w_base[low_mask] = w_base[low_mask] * float(max(1.0, LOW_THETA_WEIGHT_FACTOR))

    # Phi residual magnitudes are typically much smaller than theta in variance-space,
    # so add a scale-compensation factor so phi contributes comparably in the joint fit.
    med_theta = float(np.nanmedian(y_theta)) if y_theta.size else float("nan")
    med_phi = float(np.nanmedian(y_phi)) if y_phi.size else float("nan")
    if np.isfinite(med_theta) and np.isfinite(med_phi) and (med_theta > 0.0) and (med_phi > 0.0):
        w_phi_scale = float(np.clip(med_theta / med_phi, 1.0, 100.0))
    else:
        w_phi_scale = 1.0
    a_param = 1.0
    b_param = 0.1
    sigma_bg = 1.0
    k_intensity = float(np.nanmedian([r.brightness_abs_norm for r in sub if np.isfinite(r.brightness_abs_norm) and r.brightness_abs_norm > 0.0]))
    k_intensity = max(1e-6, k_intensity if np.isfinite(k_intensity) else 10.0)

    if least_squares is not None:
        def _resid(params: np.ndarray) -> np.ndarray:
            a_v = float(params[0])
            b_v = float(params[1])
            s_bg = float(params[2])
            k_v = float(params[3])
            i_theta = np.maximum((s_bg * s_bg) + (k_v * (np.sin(th_rad) ** 2)), 1e-9)
            var_xy = ((a_v / i_theta) + (0.5 * b_v * b_v) + ((2.0 * s_bg * s_bg) / np.maximum(i_theta * i_theta, 1e-18))) / np.maximum(npix, 1.0)
            pred_theta = d2 * var_xy
            pred_phi = var_xy / (4.0 * rr * rr)
            e_theta = np.sqrt(w_base) * (pred_theta - y_theta)
            e_phi = np.sqrt(w_base * w_phi_scale) * (pred_phi - y_phi)
            return np.concatenate([e_theta, e_phi], axis=0)

        seeds = [
            np.array([1.0, 0.1, 1.0, k_intensity], dtype=np.float64),
            np.array([0.5, 0.2, 0.7, 0.5 * k_intensity], dtype=np.float64),
            np.array([2.0, 0.05, 1.5, 2.0 * k_intensity], dtype=np.float64),
        ]

        best = None
        best_cost = float("inf")
        lb = np.array([1e-12, 1e-12, 1e-12, 1e-12], dtype=np.float64)
        ub = np.array([np.inf, np.inf, np.inf, np.inf], dtype=np.float64)
        for x0 in seeds:
            try:
                x0_use = np.minimum(np.maximum(x0, lb), ub)
                res = least_squares(_resid, x0=x0_use, bounds=(lb, ub), method="trf", max_nfev=4000)
                if res.success and np.isfinite(res.cost) and float(res.cost) < best_cost:
                    best = res
                    best_cost = float(res.cost)
            except Exception:
                continue
        if best is not None and best.x.size >= 3:
            a_param = float(max(0.0, best.x[0]))
            b_param = float(max(0.0, best.x[1]))
            sigma_bg = float(max(0.0, best.x[2]))
            k_intensity = float(max(0.0, best.x[3]))

    return ModeFit(
        mode=mode if mode is not None else "combined",
        a_param=max(0.0, a_param),
        b_param=max(0.0, b_param),
        sigma_bg=max(0.0, sigma_bg),
        k_intensity=max(1e-9, k_intensity),
        n_used=int(rr.size),
    )


def fit_shared_noise_two_datasets(
    rows_old: list[RodResult],
    rows_new: list[RodResult],
    theta_weight_bandwidth_deg: float,
    use_density_weight: bool = True,
) -> Optional[SharedNoiseTwoKFit]:
    combo: list[tuple[int, int, RodResult]] = []
    for r in rows_old:
        combo.append((0, 0 if r.mode == "77fps" else 1, r))
    for r in rows_new:
        combo.append((1, 0 if r.mode == "77fps" else 1, r))

    sub = [
        (ds, m, r)
        for ds, m, r in combo
        if np.isfinite(r.r_mean)
        and np.isfinite(r.theta_err_deg)
        and np.isfinite(r.phi_err_deg)
        and np.isfinite(r.theta_deg)
        and np.isfinite(r.brightness_mean)
        and (r.brightness_mean > 0.0)
    ]
    if len(sub) < 4:
        return None

    line_idx = np.asarray([2 * ds + m for ds, m, _ in sub], dtype=np.int64)
    rr = np.asarray([r.r_mean for _, _, r in sub], dtype=np.float64)
    sig_theta = np.radians(np.asarray([r.theta_err_deg for _, _, r in sub], dtype=np.float64))
    sig_phi = np.radians(np.asarray([r.phi_err_deg for _, _, r in sub], dtype=np.float64))
    th_deg = np.asarray([r.theta_deg for _, _, r in sub], dtype=np.float64)
    th_rad = np.radians(th_deg)

    ok = (
        np.isfinite(rr)
        & np.isfinite(sig_theta)
        & np.isfinite(sig_phi)
        & np.isfinite(th_deg)
        & (rr > 1e-6)
        & (sig_theta > 0.0)
        & (sig_phi > 0.0)
    )
    line_idx = line_idx[ok]
    rr = rr[ok]
    sig_theta = sig_theta[ok]
    sig_phi = sig_phi[ok]
    th_deg = th_deg[ok]
    th_rad = th_rad[ok]
    if rr.size < 4:
        return None

    dth = dtheta_dr_hole_fresnel(rr)
    d2 = dth * dth
    ok_d = np.isfinite(d2) & (d2 > 0.0)
    line_idx = line_idx[ok_d]
    rr = rr[ok_d]
    sig_theta = sig_theta[ok_d]
    sig_phi = sig_phi[ok_d]
    th_deg = th_deg[ok_d]
    th_rad = th_rad[ok_d]
    d2 = d2[ok_d]
    if rr.size < 4:
        return None

    y_theta = sig_theta * sig_theta
    y_phi = sig_phi * sig_phi

    if use_density_weight:
        bw = max(1e-6, float(theta_weight_bandwidth_deg))
        d = (th_deg[:, None] - th_deg[None, :]) / bw
        local_density = np.sum(np.exp(-0.5 * d * d), axis=1)
        w_base = 1.0 / np.maximum(local_density, 1e-12)
        w_base = w_base * (float(w_base.size) / float(np.sum(w_base)))
    else:
        w_base = np.ones(rr.shape, dtype=np.float64)

    low_mask = np.isfinite(th_deg) & (th_deg >= 0.0) & (th_deg <= LOW_THETA_WEIGHT_MAX_DEG)
    if np.any(low_mask):
        w_base[low_mask] *= float(max(1.0, LOW_THETA_WEIGHT_FACTOR))

    med_theta = float(np.nanmedian(y_theta)) if y_theta.size else float("nan")
    med_phi = float(np.nanmedian(y_phi)) if y_phi.size else float("nan")
    if np.isfinite(med_theta) and np.isfinite(med_phi) and (med_theta > 0.0) and (med_phi > 0.0):
        w_phi_scale = float(np.clip(med_theta / med_phi, 1.0, 100.0))
    else:
        w_phi_scale = 1.0

    # Initial guess: seed per-line fits, then share a,b by rod size.
    a_line0 = np.full(4, 0.1, dtype=np.float64)
    b_line0 = np.full(4, 0.2, dtype=np.float64)
    k0 = np.full(4, 10.0, dtype=np.float64)
    n0 = np.asarray([225.0, 121.0, 225.0, 121.0], dtype=np.float64)
    s0 = 1.0

    for li, (ds_name, mode_name) in enumerate(
        [("40nm", "77fps"), ("40nm", "maxfps_11x11"), ("25nm", "77fps"), ("25nm", "maxfps_11x11")]
    ):
        src_rows = rows_old if ds_name == "40nm" else rows_new
        sub_rows = [r for r in src_rows if r.mode == mode_name]
        fit_li = fit_noise_model_for_mode(sub_rows, mode=mode_name, theta_weight_bandwidth_deg=theta_weight_bandwidth_deg, use_density_weight=use_density_weight) if len(sub_rows) >= 3 else None
        if fit_li is not None:
            a_line0[li] = max(1e-12, float(fit_li.a_param))
            b_line0[li] = max(1e-12, float(fit_li.b_param))
            k0[li] = float(np.clip(fit_li.k_intensity, 1.0, 100.0))
            s0 = max(s0, float(fit_li.sigma_bg))
        else:
            kval = np.nanmedian(
                [
                    r.brightness_abs_norm
                    for r in sub_rows
                    if np.isfinite(r.brightness_abs_norm) and (r.brightness_abs_norm > 0.0)
                ]
            )
            if np.isfinite(kval):
                k0[li] = float(np.clip(kval, 1.0, 100.0))

    a40_0 = float(np.nanmedian(a_line0[:2]))
    b40_0 = float(np.nanmedian(b_line0[:2]))
    a25_0 = float(np.nanmedian(a_line0[2:]))
    b25_0 = float(np.nanmedian(b_line0[2:]))

    if least_squares is not None:
        def _resid(params: np.ndarray) -> np.ndarray:
            s_v = float(params[0])
            a40_v = float(params[1])
            b40_v = float(params[2])
            a25_v = float(params[3])
            b25_v = float(params[4])
            k_v = np.asarray(params[5:9], dtype=np.float64)
            n_v = np.asarray(params[9:13], dtype=np.float64)
            a_use = np.where(line_idx < 2, a40_v, a25_v)
            b_use = np.where(line_idx < 2, b40_v, b25_v)
            k_use = k_v[line_idx]
            n_use = n_v[line_idx]

            i_theta = np.maximum((s_v * s_v) + (k_use * (np.sin(th_rad) ** 2)), 1e-9)
            var_xy = (
                (a_use / i_theta)
                + (0.5 * b_use * b_use)
                + ((2.0 * s_v * s_v) / np.maximum(i_theta * i_theta, 1e-18))
            ) / np.maximum(n_use, 1.0)
            pred_theta = d2 * var_xy
            pred_phi = var_xy / (4.0 * rr * rr)
            e_theta = np.sqrt(w_base) * (pred_theta - y_theta)
            e_phi = np.sqrt(w_base * w_phi_scale) * (pred_phi - y_phi)
            return np.concatenate([e_theta, e_phi], axis=0)

        p0 = np.concatenate(
            [
                np.array([s0, a40_0, b40_0, a25_0, b25_0], dtype=np.float64),
                k0,
                n0,
            ]
        )
        seeds = [
            p0,
            p0 * np.array([0.8, 0.9, 1.1, 0.9, 1.1] + [1.0] * 8, dtype=np.float64),
            p0 * np.array([1.2, 1.1, 0.9, 1.1, 0.9] + [1.0] * 8, dtype=np.float64),
        ]
        lb = np.array(
            [1e-12] + [1e-12] * 4 + [1.0] * 4 + [10.0] * 4,
            dtype=np.float64,
        )
        ub = np.array(
            [np.inf] + [np.inf] * 4 + [100.0] * 4 + [5000.0] * 4,
            dtype=np.float64,
        )

        best = None
        best_cost = float("inf")
        for x0 in seeds:
            try:
                x0_use = np.minimum(np.maximum(x0, lb), ub)
                res = least_squares(_resid, x0=x0_use, bounds=(lb, ub), method="trf", max_nfev=14000)
                if res.success and np.isfinite(res.cost) and float(res.cost) < best_cost:
                    best = res
                    best_cost = float(res.cost)
            except Exception:
                continue

        if best is not None and best.x.size >= 13:
            x = np.maximum(np.asarray(best.x, dtype=np.float64), lb)
            s0 = float(x[0])
            a40_0 = float(x[1])
            b40_0 = float(x[2])
            a25_0 = float(x[3])
            b25_0 = float(x[4])
            k0 = np.clip(np.asarray(x[5:9], dtype=np.float64), 1.0, 100.0)
            n0 = np.maximum(np.asarray(x[9:13], dtype=np.float64), 1.0)

    return SharedNoiseTwoKFit(
        a_40=float(a40_0),
        b_40=float(b40_0),
        a_25=float(a25_0),
        b_25=float(b25_0),
        sigma_bg=max(0.0, s0),
        k_line=np.asarray(k0, dtype=np.float64),
        n_like_line=np.asarray(n0, dtype=np.float64),
        n_used=int(rr.size),
    )


def fit_intensity_theta_model(rows: list[RodResult]) -> Optional[IntensityThetaFit]:
    sub = [
        r
        for r in rows
        if np.isfinite(r.theta_deg)
        and np.isfinite(r.brightness_mean)
        and (r.brightness_mean > 0.0)
        and (r.mode in ("77fps", "maxfps_11x11"))
    ]
    if len(sub) < 3:
        return None

    def _fit_mode(mode_name: str) -> tuple[float, float]:
        msub = [r for r in sub if r.mode == mode_name]
        if len(msub) < 3:
            return (1.0, 1e-3)
        th = np.radians(np.asarray([r.theta_deg for r in msub], dtype=np.float64))
        s2 = np.sin(th) ** 2
        y = np.asarray([r.brightness_mean for r in msub], dtype=np.float64)
        A = np.column_stack([np.ones_like(s2), s2])
        x0, *_ = np.linalg.lstsq(A, y, rcond=None)
        x0 = np.maximum(x0, 1e-9)
        if least_squares is not None:
            def _resid(p: np.ndarray) -> np.ndarray:
                c0 = float(p[0])
                k0 = float(p[1])
                return (c0 + (k0 * s2)) - y
            lb = np.array([1e-9, 1e-6], dtype=np.float64)
            ub = np.array([np.inf, np.inf], dtype=np.float64)
            try:
                res = least_squares(_resid, x0=np.asarray(x0, dtype=np.float64), bounds=(lb, ub), method="trf", max_nfev=3000)
                if res.success and res.x.size >= 2:
                    x0 = np.maximum(np.asarray(res.x, dtype=np.float64), lb)
            except Exception:
                pass
        return (float(x0[0]), float(x0[1]))

    c77, k77 = _fit_mode("77fps")
    c11, k11 = _fit_mode("maxfps_11x11")
    return IntensityThetaFit(
        i_const_77=c77,
        k_77=k77,
        i_const_11=c11,
        k_11=k11,
        n_used=len(sub),
    )


def fit_intensity_theta_from_r_extrema(rows: list[RodResult]) -> Optional[IntensityThetaExtremaFit]:
    sub = [
        r
        for r in rows
        if np.isfinite(r.theta_deg)
        and np.isfinite(r.r_mean)
        and np.isfinite(r.brightness_mean)
        and (r.mode in ("77fps", "maxfps_11x11"))
    ]
    if len(sub) < 2:
        return None

    def _fit_mode(mode_name: str) -> tuple[float, float]:
        msub = [r for r in sub if r.mode == mode_name]
        if len(msub) < 2:
            return (float("nan"), float("nan"))
        r_min = min(msub, key=lambda x: float(x.r_mean))
        r_max = max(msub, key=lambda x: float(x.r_mean))
        i0 = float(r_min.brightness_mean)
        i_max = float(r_max.brightness_mean)
        return (i0, i_max - i0)

    i0_77, k_77 = _fit_mode("77fps")
    i0_11, k_11 = _fit_mode("maxfps_11x11")
    if not (np.isfinite(i0_77) and np.isfinite(k_77) and np.isfinite(i0_11) and np.isfinite(k_11)):
        return None

    return IntensityThetaExtremaFit(
        i0_77=float(i0_77),
        k_77=float(k_77),
        i0_11=float(i0_11),
        k_11=float(k_11),
        n_used=int(len(sub)),
    )


def make_intensity_theta_extrema_plot(
    out_dir: Path,
    rows: list[RodResult],
    fit_ext: Optional[IntensityThetaExtremaFit],
    stem_suffix: str = "",
) -> None:
    if fit_ext is None:
        return
    pts = [
        r
        for r in rows
        if np.isfinite(r.theta_deg)
        and np.isfinite(r.brightness_mean)
        and np.isfinite(r.r_mean)
        and (r.mode in ("77fps", "maxfps_11x11"))
    ]
    if not pts:
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.2, 4.8), dpi=130)
    colors = {"77fps": "tab:blue", "maxfps_11x11": "tab:orange"}
    labels = {"77fps": "77fps 15x15 pixels", "maxfps_11x11": "1600fps"}

    for mode in ("77fps", "maxfps_11x11"):
        sub = [r for r in pts if r.mode == mode]
        if not sub:
            continue
        x = np.asarray([float(r.theta_deg) for r in sub], dtype=np.float64)
        y = np.asarray([float(r.brightness_mean) for r in sub], dtype=np.float64)
        ax.scatter(x, y, s=20, alpha=0.75, color=colors[mode], edgecolors="none", label=f"Data {labels[mode]}")

        th = np.linspace(0.0, 90.0, 600)
        s2 = np.sin(np.radians(th)) ** 2
        if mode == "77fps":
            i0 = fit_ext.i0_77
            k = fit_ext.k_77
        else:
            i0 = fit_ext.i0_11
            k = fit_ext.k_11
        ax.plot(th, i0 + (k * s2), "--", lw=2.0, color=colors[mode], label=f"Extrema anchor {labels[mode]}")

    ax.set_xlim(0.0, 90.0)
    ax.set_xlabel("theta from theta(r) method (deg)")
    ax.set_ylabel("mean intensity (raw units)")
    ax.set_title("I(theta) using smallest-r and largest-r anchors")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / f"intensity_vs_theta_extrema_anchor{stem_suffix}.png")
    plt.close(fig)


def _load_frame_means(npy_path: str) -> Optional[np.ndarray]:
    if not npy_path:
        return None
    p = Path(npy_path)
    if not p.exists():
        return None
    try:
        arr = np.load(p, mmap_mode="r", allow_pickle=False)
    except Exception:
        return None
    if not hasattr(arr, "ndim") or getattr(arr, "size", 0) == 0:
        return None
    try:
        if arr.ndim >= 3:
            means = np.asarray(arr, dtype=np.float64).reshape(arr.shape[0], -1).mean(axis=1)
        elif arr.ndim == 2:
            means = np.asarray([float(np.mean(arr))], dtype=np.float64)
        else:
            means = np.asarray(arr, dtype=np.float64).reshape(-1)
    except Exception:
        return None
    means = means[np.isfinite(means)]
    if means.size < 3:
        return None
    return means


def _theta_from_intensity_series(i_series: np.ndarray, i_const: float, k_mode: float) -> np.ndarray:
    x = (np.asarray(i_series, dtype=np.float64) - float(i_const)) / max(float(k_mode), 1e-12)
    x = np.clip(x, 0.0, 1.0)
    return np.arcsin(np.sqrt(x))


def build_intensity_theta_points(
    rows: list[RodResult],
    fit_i: IntensityThetaFit,
) -> list[dict]:
    pts: list[dict] = []
    for r in rows:
        if r.mode not in ("77fps", "maxfps_11x11"):
            continue
        if r.mode == "77fps":
            k_mode = fit_i.k_77
            i_const_mode = fit_i.i_const_77
        else:
            k_mode = fit_i.k_11
            i_const_mode = fit_i.i_const_11
        if not np.isfinite(k_mode) or k_mode <= 1e-12:
            continue
        means = _load_frame_means(r.npy_path)
        if means is None or means.size < 3:
            continue
        th_series = _theta_from_intensity_series(means, i_const_mode, k_mode)
        th_deg = np.degrees(th_series)
        th_deg = th_deg[np.isfinite(th_deg)]
        if th_deg.size < 3:
            continue
        th_center = float(np.median(th_deg))
        th_err = float(robust_sigma(th_deg))
        if not np.isfinite(th_center) or not np.isfinite(th_err) or th_err <= 0.0:
            continue
        pts.append(
            {
                "mode": r.mode,
                "theta_deg": th_center,
                "theta_err_deg": th_err,
            }
        )
    return pts


def fit_intensity_theta_error_model(
    points: list[dict],
    fit_i: IntensityThetaFit,
) -> Optional[IntensityThetaErrorFit]:
    if least_squares is None or not points:
        return None
    th_deg = np.asarray([p["theta_deg"] for p in points], dtype=np.float64)
    sig = np.radians(np.asarray([p["theta_err_deg"] for p in points], dtype=np.float64))
    modes = [str(p["mode"]) for p in points]
    ok = np.isfinite(th_deg) & np.isfinite(sig) & (sig > 0.0)
    if np.count_nonzero(ok) < 5:
        return None
    th = np.radians(th_deg[ok])
    sig = sig[ok]
    m_ok = [modes[i] for i in np.where(ok)[0]]
    s = np.sin(th)
    c = np.cos(th)
    sc_ok = np.abs(s * c) > 1e-9
    if np.count_nonzero(sc_ok) < 5:
        return None
    th = th[sc_ok]
    sig = sig[sc_ok]
    m_ok2 = [m_ok[i] for i in np.where(sc_ok)[0]]

    npix = np.asarray([roi_pixels_for_mode(m) for m in m_ok2], dtype=np.float64)
    k_mode = np.asarray([fit_i.k_77 if m == "77fps" else fit_i.k_11 for m in m_ok2], dtype=np.float64)
    i_const_mode = np.asarray([fit_i.i_const_77 if m == "77fps" else fit_i.i_const_11 for m in m_ok2], dtype=np.float64)
    s = np.sin(th)
    c = np.cos(th)

    def _pred_sigma(a_p: float, b_p: float, s_bg: float) -> np.ndarray:
        i_mean = i_const_mode + (k_mode * (s * s))
        i_mean = np.maximum(i_mean, 1e-9)
        var_i_mean = ((a_p * i_mean) + ((b_p * b_p) * i_mean) + (s_bg * s_bg)) / np.maximum(npix, 1.0)
        var_i_mean = np.maximum(var_i_mean, 0.0)
        dth_dI = 1.0 / np.maximum(2.0 * k_mode * np.abs(s * c), 1e-12)
        return dth_dI * np.sqrt(var_i_mean)

    def _resid(p: np.ndarray) -> np.ndarray:
        a_p = float(p[0])
        b_p = float(p[1])
        s_bg = float(p[2])
        pred = _pred_sigma(a_p, b_p, s_bg)
        return pred - sig

    x0 = np.asarray([1.0, 0.1, 1.0], dtype=np.float64)
    lb = np.asarray([1e-12, 1e-12, 1e-12], dtype=np.float64)
    ub = np.asarray([np.inf, np.inf, np.inf], dtype=np.float64)
    try:
        res = least_squares(_resid, x0=x0, bounds=(lb, ub), method="trf", max_nfev=5000)
        if not res.success or res.x.size < 3:
            return None
        xx = np.maximum(np.asarray(res.x, dtype=np.float64), 1e-12)
        return IntensityThetaErrorFit(
            a_param=float(xx[0]),
            b_param=float(xx[1]),
            sigma_bg=float(xx[2]),
            n_used=int(sig.size),
        )
    except Exception:
        return None


def sigma_xy_model(theta_rad: np.ndarray, fit: ModeFit, mode: Optional[str] = None) -> np.ndarray:
    theta_rad = np.asarray(theta_rad, dtype=np.float64)
    i_use = np.maximum((fit.sigma_bg * fit.sigma_bg) + (fit.k_intensity * (np.sin(theta_rad) ** 2)), 1e-9)
    npix = roi_pixels_for_mode(mode if mode is not None else "maxfps_11x11")
    return np.sqrt(
        (
            (fit.a_param / i_use)
            + (0.5 * fit.b_param * fit.b_param)
            + ((2.0 * fit.sigma_bg * fit.sigma_bg) / np.maximum(i_use * i_use, 1e-18))
        )
        / max(1.0, npix)
    )


def predict_phi_err_rad(r: np.ndarray, theta_rad: np.ndarray, fit: ModeFit, mode: Optional[str] = None) -> np.ndarray:
    r = np.asarray(r, dtype=np.float64)
    theta_rad = np.asarray(theta_rad, dtype=np.float64)
    out = np.full(r.shape, np.nan, dtype=np.float64)
    ok = np.isfinite(r) & (r > 1e-9)
    if not np.any(ok):
        return out
    mode_use = mode if mode is not None else fit.mode
    sxy = sigma_xy_model(theta_rad[ok], fit, mode=mode_use)
    out[ok] = sxy / (2.0 * r[ok])
    return out


def predict_theta_err_rad(r: np.ndarray, theta_rad: np.ndarray, fit: ModeFit, mode: Optional[str] = None) -> np.ndarray:
    r = np.asarray(r, dtype=np.float64)
    theta_rad = np.asarray(theta_rad, dtype=np.float64)
    dth = dtheta_dr_hole_fresnel(r)
    mode_use = mode if mode is not None else fit.mode
    sxy = sigma_xy_model(theta_rad, fit, mode=mode_use)
    out = np.abs(dth) * sxy
    return out


def predict_theta_err_intensity_deg(
    theta_deg: np.ndarray,
    mode: str,
    fit_xy: ModeFit,
    fit_i: IntensityThetaFit,
) -> np.ndarray:
    th_deg = np.asarray(theta_deg, dtype=np.float64)
    th = np.radians(th_deg)
    s = np.sin(th)
    c = np.cos(th)

    if mode == "77fps":
        k_mode = float(fit_i.k_77)
        i_const_mode = float(fit_i.i_const_77)
    else:
        k_mode = float(fit_i.k_11)
        i_const_mode = float(fit_i.i_const_11)
    i_mean = i_const_mode + (k_mode * (s * s))
    npix = roi_pixels_for_mode(mode)

    out = np.full(th.shape, np.nan, dtype=np.float64)
    ok = np.isfinite(i_mean) & (i_mean > 1e-12) & np.isfinite(s) & np.isfinite(c) & (np.abs(s * c) > 1e-9) & (k_mode > 1e-12)
    if not np.any(ok):
        return out

    var_i_mean = (
        (fit_xy.a_param * i_mean)
        + ((fit_xy.b_param * fit_xy.b_param) * i_mean)
        + (fit_xy.sigma_bg * fit_xy.sigma_bg)
    ) / max(1.0, npix)
    var_i_mean = np.maximum(var_i_mean, 0.0)
    dth_dI = 1.0 / (2.0 * k_mode * s * c)
    sig_th = np.abs(dth_dI) * np.sqrt(var_i_mean)
    out[ok] = np.degrees(sig_th[ok])
    return out


def theta_theory_grid_deg() -> np.ndarray:
    # Full model range up to 90 deg (exclude exact endpoints to avoid singular behavior at r=0 and theta=90).
    return np.linspace(0.1, 89.9, 360)


def write_csv(path: Path, rows: list[RodResult]) -> None:
    lines = [
        "rod_id,rod_key,mode,n_frames,theta_deg,theta_err_deg,phi_err_deg,r_mean,xy_var_score,brightness_mean,brightness_abs_norm",
    ]
    for r in rows:
        lines.append(
            f"{r.rod_id},{r.rod_key},{r.mode},{r.n_frames},{r.theta_deg:.8g},{r.theta_err_deg:.8g},{r.phi_err_deg:.8g},{r.r_mean:.8g},{r.xy_var_score:.8g},{r.brightness_mean:.8g},{r.brightness_abs_norm:.8g}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def select_best_rows_by_theta_bin(
    rows: list[RodResult],
    bin_width_deg: float = 10.0,
    top_k_per_bin: int = 2,
    theta_min_deg: float = 10.0,
    theta_max_deg: float = 80.0,
) -> tuple[list[RodResult], list[tuple[str, float, float, int]]]:
    if not rows:
        return ([], [])
    bw = max(1e-6, float(bin_width_deg))
    tmin = float(theta_min_deg)
    tmax = float(theta_max_deg)
    if tmax <= tmin:
        return ([], [])
    nbins = int(np.ceil((tmax - tmin) / bw))
    selected_mode_rods: set[tuple[str, str]] = set()
    bin_counts: list[tuple[str, float, float, int]] = []

    modes = sorted({r.mode for r in rows})
    for mode in modes:
        mode_rows = [r for r in rows if r.mode == mode]
        by_rod: dict[str, list[RodResult]] = {}
        for r in mode_rows:
            by_rod.setdefault(r.rod_key, []).append(r)

        rod_meta: list[tuple[str, float, float]] = []
        for rk, grp in by_rod.items():
            th = np.asarray([g.theta_deg for g in grp], dtype=np.float64)
            sc = np.asarray([g.xy_var_score for g in grp], dtype=np.float64)
            th = th[np.isfinite(th)]
            sc = sc[np.isfinite(sc)]
            if th.size == 0 or sc.size == 0:
                continue
            theta_rep = float(np.median(th))
            if theta_rep < tmin or theta_rep > tmax:
                continue
            score_rep = float(np.median(sc))
            rod_meta.append((rk, theta_rep, score_rep))

        bins: list[list[tuple[str, float]]] = [[] for _ in range(nbins)]
        for rk, theta_rep, score_rep in rod_meta:
            idx = int((theta_rep - tmin) // bw)
            idx = max(0, min(idx, nbins - 1))
            bins[idx].append((rk, score_rep))

        for i in range(nbins):
            start = tmin + (i * bw)
            end = min(tmax, tmin + ((i + 1) * bw))
            picked = sorted(bins[i], key=lambda t: (t[1], t[0]))[: int(top_k_per_bin)]
            for rk, _ in picked:
                selected_mode_rods.add((mode, rk))
            bin_counts.append((mode, start, end, len(picked)))

    out = [r for r in rows if (r.mode, r.rod_key) in selected_mode_rods]
    return (out, bin_counts)


def remove_phi_theta_anomalies(rows: list[RodResult]) -> tuple[list[RodResult], int]:
    kept: list[RodResult] = []
    removed = 0
    for r in rows:
        drop_hi = (r.theta_deg > PHI_OUTLIER_HIGH_THETA_DEG) and (r.phi_err_deg > PHI_OUTLIER_HIGH_ERR_DEG)
        drop_lo = (r.theta_deg < PHI_OUTLIER_LOW_THETA_DEG) and (r.phi_err_deg < PHI_OUTLIER_LOW_ERR_DEG)
        if drop_hi or drop_lo:
            removed += 1
            continue
        kept.append(r)
    return (kept, removed)


def remove_phi_theta_anomalies_keep_low_theta(rows: list[RodResult]) -> tuple[list[RodResult], int]:
    kept: list[RodResult] = []
    removed = 0
    for r in rows:
        drop_hi = (r.theta_deg > PHI_OUTLIER_HIGH_THETA_DEG) and (r.phi_err_deg > PHI_OUTLIER_HIGH_ERR_DEG)
        if drop_hi:
            removed += 1
            continue
        kept.append(r)
    return (kept, removed)


def write_fit_json(path: Path, fits: dict[str, ModeFit]) -> None:
    payload = {
        "model": "I(theta)=sigma_bg^2+K*sin(theta)^2; sigma_xy^2=(a/I + b^2/2 + 2*sigma_bg^2/I^2)/Npix(mode); sigma_phi^2=sigma_xy^2/(4r^2); sigma_theta^2=(dtheta/dr)^2*sigma_xy^2",
        "fits": {
            m: {
                "a_param": float(f.a_param),
                "b_param": float(f.b_param),
                "sigma_bg": float(f.sigma_bg),
                "k_intensity": float(f.k_intensity),
                "roi_pixels_mode": (
                    "mixed(11x11,15x15)" if f.mode == "combined" else float(roi_pixels_for_mode(f.mode))
                ),
                "n_used": int(f.n_used),
            }
            for m, f in fits.items()
        },
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def theta_plot_limits(rows: list[RodResult]) -> tuple[float, float]:
    vals = np.asarray([r.theta_deg for r in rows], dtype=np.float64)
    ok = np.isfinite(vals)
    if not np.any(ok):
        return (0.0, 90.0)
    lo = float(np.min(vals[ok]))
    hi = float(np.max(vals[ok]))
    if hi <= lo:
        pad = 0.5
        return (max(0.0, lo - pad), min(90.0, hi + pad))
    return (max(0.0, lo), min(90.0, hi))


def add_scatter_by_mode(
    ax,
    rows: list[RodResult],
    y_attr: str,
    y_label: str,
    xlim: tuple[float, float],
    x_mapper=None,
) -> None:
    modes = ["77fps", "maxfps_11x11"]
    colors = {"77fps": "tab:blue", "maxfps_11x11": "tab:orange"}
    labels = {"77fps": "77fps 15x15 pixels", "maxfps_11x11": "1600fps"}
    used_any = False
    for m in modes:
        sub = [r for r in rows if r.mode == m and np.isfinite(getattr(r, y_attr)) and np.isfinite(r.theta_deg)]
        if not sub:
            continue
        used_any = True
        x_raw = np.asarray([r.theta_deg for r in sub], dtype=np.float64)
        if x_mapper is None:
            x = x_raw
        else:
            x = np.asarray([x_mapper(float(v)) for v in x_raw], dtype=np.float64)
        y = np.asarray([getattr(r, y_attr) for r in sub], dtype=np.float64)
        ax.scatter(
            x,
            y,
            s=20,
            alpha=0.85,
            label=labels.get(m, m),
            color=colors.get(m, None),
            edgecolors="none",
        )

    if used_any:
        ax.legend(frameon=False, fontsize=9)
    ax.set_xlabel("theta (deg) [hole+fresnel from r]")
    ax.set_ylabel(y_label)
    ax.set_xlim(*xlim)
    ax.grid(True, alpha=0.3)


def add_theory_curves(
    ax,
    rows: list[RodResult],
    fits: dict[str, ModeFit],
    which: str,
    xlim: Optional[tuple[float, float]] = None,
    y_max_deg: Optional[float] = None,
    x_mapper=None,
) -> None:
    if not fits:
        return

    fit = next(iter(fits.values()))
    theta_deg_grid = theta_theory_grid_deg()
    keep = theta_deg_grid <= THETA_THEORY_PLOT_MAX_DEG
    if xlim is not None:
        keep = keep & (theta_deg_grid >= float(xlim[0])) & (theta_deg_grid <= float(xlim[1]))
    theta_deg_grid = theta_deg_grid[keep]
    if theta_deg_grid.size < 2:
        return
    theta_rad_grid = np.radians(theta_deg_grid)
    r_grid = r_from_theta_hole_fresnel(theta_rad_grid)
    ok = np.isfinite(r_grid) & (r_grid >= 0.0) & (r_grid < THETA_R_MAX)
    if np.count_nonzero(ok) < 2:
        return
    mode_styles = [
        ("77fps", "tab:blue", "Fit (77fps 15x15 pixels)"),
        ("maxfps_11x11", "tab:orange", "Fit (1600fps)"),
    ]
    for mode_name, color, label in mode_styles:
        if which == "phi":
            y_deg = np.degrees(predict_phi_err_rad(r_grid[ok], theta_rad_grid[ok], fit, mode=mode_name))
        else:
            y_deg = np.degrees(predict_theta_err_rad(r_grid[ok], theta_rad_grid[ok], fit, mode=mode_name))

        x_plot = theta_deg_grid[ok]
        ok2 = np.isfinite(x_plot) & np.isfinite(y_deg)
        if y_max_deg is not None:
            ok2 = ok2 & (y_deg <= float(y_max_deg))
        if np.count_nonzero(ok2) < 2:
            continue
        x_raw = x_plot[ok2]
        if x_mapper is None:
            x_use = x_raw
        else:
            x_use = np.asarray([x_mapper(float(v)) for v in x_raw], dtype=np.float64)
        y_use = y_deg[ok2]
        order = np.argsort(x_use)
        ax.plot(x_use[order], y_use[order], color=color, lw=2.0, alpha=0.95, label=label)


def _draw_overflow_points(ax, rows: list[RodResult], y_attr: str, xlim: tuple[float, float], y_cap: float) -> None:
    modes = ["77fps", "maxfps_11x11"]
    colors = {"77fps": "tab:blue", "maxfps_11x11": "tab:orange"}
    for m in modes:
        sub = [
            r
            for r in rows
            if r.mode == m
            and np.isfinite(r.theta_deg)
            and np.isfinite(getattr(r, y_attr))
            and (r.theta_deg >= xlim[0])
            and (r.theta_deg <= xlim[1])
            and (getattr(r, y_attr) > y_cap)
        ]
        if not sub:
            continue
        x = np.asarray([r.theta_deg for r in sub], dtype=np.float64)
        y = np.full_like(x, y_cap, dtype=np.float64)
        ax.scatter(x, y, s=28, marker="^", color=colors.get(m, "k"), edgecolors="none", alpha=0.9, zorder=4)


def _draw_compact_range_marker(ax, x0: float, x1: float, label0: str, label1: str) -> None:
    xmin, xmax = ax.get_xlim()
    xr = max(1e-9, float(xmax - xmin))
    compact = 0.08 * xr
    xc = 0.5 * (x0 + x1)
    a = max(xmin, xc - 0.5 * compact)
    b = min(xmax, xc + 0.5 * compact)
    trans = ax.get_xaxis_transform()
    y0 = -0.065
    ax.plot([a, b], [y0, y0], transform=trans, color="0.25", lw=1.2, clip_on=False)
    ax.plot([a, a], [y0 - 0.02, y0 + 0.02], transform=trans, color="0.25", lw=1.2, clip_on=False)
    ax.plot([b, b], [y0 - 0.02, y0 + 0.02], transform=trans, color="0.25", lw=1.2, clip_on=False)
    ax.text(a, y0 - 0.04, label0, transform=trans, ha="center", va="top", fontsize=8)
    ax.text(b, y0 - 0.04, label1, transform=trans, ha="center", va="top", fontsize=8)
    return (a, b)


def y_plot_limits(
    rows: list[RodResult], fits: dict[str, ModeFit], which: str, xlim: tuple[float, float]
) -> tuple[float, float]:
    vals: list[np.ndarray] = []
    y_attr = "phi_err_deg" if which == "phi" else "theta_err_deg"
    y_sc = np.asarray([getattr(r, y_attr) for r in rows], dtype=np.float64)
    x_sc = np.asarray([r.theta_deg for r in rows], dtype=np.float64)
    ok_sc = np.isfinite(x_sc) & np.isfinite(y_sc) & (x_sc >= xlim[0]) & (x_sc <= xlim[1])
    if np.any(ok_sc):
        vals.append(y_sc[ok_sc])

    theta_deg_grid = theta_theory_grid_deg()
    use = (
        (theta_deg_grid >= xlim[0])
        & (theta_deg_grid <= xlim[1])
        & (theta_deg_grid <= THETA_THEORY_PLOT_MAX_DEG)
    )
    if np.any(use):
        theta_rad_grid = np.radians(theta_deg_grid[use])
        r_grid = r_from_theta_hole_fresnel(theta_rad_grid)
        ok_r = np.isfinite(r_grid) & (r_grid >= 0.0) & (r_grid < THETA_R_MAX)
        if np.any(ok_r):
            for fit in fits.values():
                for mode_name in ("77fps", "maxfps_11x11"):
                    if which == "phi":
                        y_deg = np.degrees(predict_phi_err_rad(r_grid[ok_r], theta_rad_grid[ok_r], fit, mode=mode_name))
                    else:
                        y_deg = np.degrees(predict_theta_err_rad(r_grid[ok_r], theta_rad_grid[ok_r], fit, mode=mode_name))
                    y_deg = np.asarray(y_deg, dtype=np.float64)
                    y_deg = y_deg[np.isfinite(y_deg)]
                    if y_deg.size:
                        vals.append(y_deg)

    if not vals:
        return (0.0, 1.0)
    y_all = np.concatenate(vals, axis=0)
    y_all = y_all[np.isfinite(y_all)]
    if y_all.size == 0:
        return (0.0, 1.0)
    lo = float(np.min(y_all))
    hi = float(np.max(y_all))
    if hi <= lo:
        pad = max(0.1, abs(hi) * 0.1)
        return (max(0.0, lo - pad), hi + pad)
    pad = max(0.05 * (hi - lo), 0.05)
    return (max(0.0, lo - pad), hi + pad)


def make_plots(
    rows: list[RodResult],
    out_dir: Path,
    fits: dict[str, ModeFit],
    intensity_theta_fit: Optional[IntensityThetaFit] = None,
    rows_inset: Optional[list[RodResult]] = None,
    stem_suffix: str = "",
    title_suffix: str = "",
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows_in = rows if rows_inset is None else rows_inset
    xlim_full = (0.0, 90.0)

    # Plot 1: phi error vs theta
    fig1, ax1 = plt.subplots(figsize=(7.2, 4.8), dpi=130)
    add_scatter_by_mode(
        ax1,
        rows_in,
        y_attr="phi_err_deg",
        y_label="phi error (deg, 1-sigma robust)",
        xlim=xlim_full,
    )
    add_theory_curves(ax1, rows_in, fits, which="phi", xlim=xlim_full)
    ax1.set_ylim(0.0, 5.0)
    ax1.set_title(f"Phi Estimation Error vs Theta{title_suffix}")
    # Mark low-theta range where BLUE model curve predicts phi error > 5 deg.
    theta_grid_phi = np.linspace(0.1, 89.9, 1200)
    theta_rad_grid_phi = np.radians(theta_grid_phi)
    r_grid_phi = r_from_theta_hole_fresnel(theta_rad_grid_phi)
    ok_phi = np.isfinite(r_grid_phi) & (r_grid_phi > 0.0) & (r_grid_phi < THETA_R_MAX)
    phi_diverge_drawn = False
    if np.any(ok_phi) and fits:
        fit0 = next(iter(fits.values()))
        y_blue_phi = np.full(theta_grid_phi.shape, np.nan, dtype=np.float64)
        y_calc_phi = np.degrees(
            predict_phi_err_rad(r_grid_phi[ok_phi], theta_rad_grid_phi[ok_phi], fit0, mode="77fps")
        )
        y_blue_phi[ok_phi] = y_calc_phi
        left_phi = np.isfinite(y_blue_phi) & (y_blue_phi > 5.0) & (theta_grid_phi <= 45.0)
        if np.any(left_phi):
            left_end_phi = float(np.max(theta_grid_phi[left_phi]))
            ax1.axvspan(
                0.0,
                left_end_phi,
                facecolor="none",
                hatch="///",
                edgecolor="0.45",
                linewidth=0.0,
                zorder=0,
            )
            phi_diverge_drawn = True
    h1, l1 = ax1.get_legend_handles_labels()
    if phi_diverge_drawn:
        h1.append(Patch(facecolor="none", edgecolor="0.45", hatch="///", label="Model error diverges"))
        l1.append("Model error diverges")
    ax1.legend(h1, l1, frameon=False, fontsize=8)
    fig1.tight_layout()
    fig1.savefig(out_dir / f"phi_error_vs_theta{stem_suffix}.png")
    plt.close(fig1)

    # Plot 2: theta error vs theta (clip display to 0..10 deg)
    fig2, ax2 = plt.subplots(figsize=(7.2, 4.8), dpi=130)
    add_scatter_by_mode(
        ax2,
        rows_in,
        y_attr="theta_err_deg",
        y_label="theta error (deg, 1-sigma robust)",
        xlim=xlim_full,
    )
    add_theory_curves(ax2, rows_in, fits, which="theta", xlim=xlim_full)
    if intensity_theta_fit is not None and fits:
        fit0 = next(iter(fits.values()))
        th_grid = np.linspace(0.1, 89.9, 900)
        for mode_name, color in (("77fps", "tab:blue"), ("maxfps_11x11", "tab:orange")):
            y_alt = predict_theta_err_intensity_deg(th_grid, mode_name, fit0, intensity_theta_fit)
            ok_alt = np.isfinite(y_alt)
            if np.count_nonzero(ok_alt) >= 2:
                ax2.plot(
                    th_grid[ok_alt],
                    y_alt[ok_alt],
                    color=color,
                    lw=1.6,
                    ls="--",
                    alpha=0.95,
                    label=f"Intensity-theta fit ({'77fps 15x15 pixels' if mode_name == '77fps' else '1600fps'})",
                )
    ax2.set_xlim(*xlim_full)
    ax2.set_ylim(0.0, 10.0)
    ax2.set_title(f"Theta Estimation Error vs Theta{title_suffix}")
    # Mark upper-theta range where BLUE model curve predicts theta error > 10 deg.
    theta_grid = np.linspace(0.1, 89.9, 1200)
    theta_rad_grid = np.radians(theta_grid)
    r_grid = r_from_theta_hole_fresnel(theta_rad_grid)
    ok = np.isfinite(r_grid) & (r_grid > 0.0) & (r_grid < THETA_R_MAX)
    theta_diverge_drawn = False
    if np.any(ok) and fits:
        fit0 = next(iter(fits.values()))
        y_blue_th = np.full(theta_grid.shape, np.nan, dtype=np.float64)
        y_calc_th = np.degrees(
            predict_theta_err_rad(r_grid[ok], theta_rad_grid[ok], fit0, mode="77fps")
        )
        y_blue_th[ok] = y_calc_th
        right_mask = np.isfinite(y_blue_th) & (y_blue_th > 10.0) & (theta_grid >= 45.0)
        if np.any(right_mask):
            right_start = float(np.min(theta_grid[right_mask]))
            ax2.axvspan(
                right_start,
                90.0,
                facecolor="none",
                hatch="///",
                edgecolor="0.45",
                linewidth=0.0,
                zorder=0,
            )
            theta_diverge_drawn = True
    h2, l2 = ax2.get_legend_handles_labels()
    if theta_diverge_drawn:
        h2.append(Patch(facecolor="none", edgecolor="0.45", hatch="///", label="Model error diverges"))
        l2.append("Model error diverges")
    ax2.legend(h2, l2, frameon=False, fontsize=8)

    fig2.tight_layout()
    fig2.savefig(out_dir / f"theta_error_vs_theta{stem_suffix}.png")
    plt.close(fig2)


def make_intensity_theta_method_plot(
    out_dir: Path,
    points: list[dict],
    old_fit: Optional[ModeFit],
    fit_i: Optional[IntensityThetaFit],
    fit_err_i: Optional[IntensityThetaErrorFit],
    stem_suffix: str = "",
) -> None:
    if not points or fit_i is None or fit_err_i is None:
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.2, 4.8), dpi=130)
    colors = {"77fps": "tab:blue", "maxfps_11x11": "tab:orange"}
    labels = {"77fps": "77fps 15x15 pixels", "maxfps_11x11": "1600fps"}

    for mode in ("77fps", "maxfps_11x11"):
        sub = [p for p in points if p["mode"] == mode and np.isfinite(p["theta_deg"]) and np.isfinite(p["theta_err_deg"])]
        if not sub:
            continue
        x = np.asarray([p["theta_deg"] for p in sub], dtype=np.float64)
        y = np.asarray([p["theta_err_deg"] for p in sub], dtype=np.float64)
        ax.scatter(x, y, s=20, alpha=0.85, color=colors[mode], edgecolors="none", label=f"Data (theta(I)) {labels[mode]}")

    th_grid = np.linspace(0.1, 89.9, 900)
    th = np.radians(th_grid)
    s = np.sin(th)
    c = np.cos(th)
    y_model_vals: list[np.ndarray] = []
    for mode in ("77fps", "maxfps_11x11"):
        if mode == "77fps":
            k_mode = fit_i.k_77
            i_const_mode = fit_i.i_const_77
        else:
            k_mode = fit_i.k_11
            i_const_mode = fit_i.i_const_11
        npix = roi_pixels_for_mode(mode)
        if not np.isfinite(k_mode) or k_mode <= 1e-12:
            continue
        i_mean = i_const_mode + (k_mode * (s * s))
        i_mean = np.maximum(i_mean, 1e-9)
        var_i_mean = (
            (fit_err_i.a_param * i_mean)
            + ((fit_err_i.b_param * fit_err_i.b_param) * i_mean)
            + (fit_err_i.sigma_bg * fit_err_i.sigma_bg)
        ) / max(1.0, npix)
        dth_dI = 1.0 / np.maximum(2.0 * k_mode * np.abs(s * c), 1e-12)
        y_deg = np.degrees(dth_dI * np.sqrt(np.maximum(var_i_mean, 0.0)))
        ok = np.isfinite(y_deg)
        if np.count_nonzero(ok) >= 2:
            y_model_vals.append(y_deg[ok])
            ax.plot(th_grid[ok], y_deg[ok], "--", lw=2.0, color=colors[mode], alpha=0.95, label=f"Intensity-theta theory {labels[mode]}")

    # Old theta(r)-based theory curves for comparison.
    if old_fit is not None:
        theta_rad_grid = np.radians(th_grid)
        r_grid = r_from_theta_hole_fresnel(theta_rad_grid)
        okr = np.isfinite(r_grid) & (r_grid > 0.0) & (r_grid < THETA_R_MAX)
        for mode in ("77fps", "maxfps_11x11"):
            y_old = np.degrees(predict_theta_err_rad(r_grid[okr], theta_rad_grid[okr], old_fit, mode=mode))
            y_all = np.full(th_grid.shape, np.nan, dtype=np.float64)
            y_all[okr] = y_old
            ok = np.isfinite(y_all)
            if np.count_nonzero(ok) >= 2:
                y_model_vals.append(y_all[ok])
                ax.plot(th_grid[ok], y_all[ok], "-", lw=1.5, color=colors[mode], alpha=0.6, label=f"Old theta(r) theory {labels[mode]}")

    ax.set_xlim(DATA_THETA_MIN_DEG, DATA_THETA_MAX_DEG)
    # Set y-range from model values in the displayed theta window.
    y_hi = None
    if y_model_vals:
        y_all = np.concatenate(y_model_vals)
        xmask = (th_grid >= DATA_THETA_MIN_DEG) & (th_grid <= DATA_THETA_MAX_DEG)
        # Recompute robust top bound directly on plotted x-window samples.
        y_window: list[float] = []
        for mode in ("77fps", "maxfps_11x11"):
            if mode == "77fps":
                k_mode = fit_i.k_77
                i_const_mode = fit_i.i_const_77
            else:
                k_mode = fit_i.k_11
                i_const_mode = fit_i.i_const_11
            npix = roi_pixels_for_mode(mode)
            if np.isfinite(k_mode) and k_mode > 1e-12:
                i_mean = i_const_mode + (k_mode * (s * s))
                i_mean = np.maximum(i_mean, 1e-9)
                var_i_mean = (
                    (fit_err_i.a_param * i_mean)
                    + ((fit_err_i.b_param * fit_err_i.b_param) * i_mean)
                    + (fit_err_i.sigma_bg * fit_err_i.sigma_bg)
                ) / max(1.0, npix)
                dth_dI = 1.0 / np.maximum(2.0 * k_mode * np.abs(s * c), 1e-12)
                y_deg = np.degrees(dth_dI * np.sqrt(np.maximum(var_i_mean, 0.0)))
                y_ok = y_deg[xmask & np.isfinite(y_deg)]
                if y_ok.size:
                    y_window.extend(y_ok.tolist())
        if y_window:
            y_max = float(np.max(np.asarray(y_window, dtype=np.float64)))
            if np.isfinite(y_max) and y_max > 0.0:
                y_hi = 1.1 * y_max
    if y_hi is None:
        y_hi = 5.0
    ax.set_ylim(0.0, y_hi)
    ax.set_xlabel("theta from intensity reconstruction (deg)")
    ax.set_ylabel("theta error (deg, robust 1-sigma)")
    ax.set_title("Theta Error vs Theta using Intensity-only Reconstruction")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / f"theta_error_vs_theta_intensity_method{stem_suffix}.png")
    plt.close(fig)


def _collect_rows_from_good_dirs(good_dirs: list[Path], modes: list[str], min_frames: int) -> list[RodResult]:
    out: list[RodResult] = []
    rod_dirs: list[Path] = []
    for gd in good_dirs:
        if gd.exists() and gd.is_dir():
            rod_dirs.extend(sorted([p for p in gd.iterdir() if p.is_dir()]))
    for rd in rod_dirs:
        for mode in modes:
            res = analyze_mode(rd, mode=mode, min_frames=min_frames)
            if res is not None:
                out.append(res)
    return out


def _remove_mid_theta_high_err(rows: list[RodResult]) -> tuple[list[RodResult], int]:
    kept: list[RodResult] = []
    removed = 0
    for r in rows:
        if (
            np.isfinite(r.theta_deg)
            and np.isfinite(r.theta_err_deg)
            and (25.0 <= float(r.theta_deg) <= 35.0)
            and (float(r.theta_err_deg) > 2.1)
        ):
            removed += 1
            continue
        kept.append(r)
    return (kept, removed)


def _remove_local_high_error_points(
    rows: list[RodResult],
    frac_remove: float = 0.20,
    theta_lo_deg: float = 20.0,
    theta_hi_deg: float = 65.0,
    bandwidth_deg: float = 6.0,
) -> tuple[list[RodResult], int]:
    if not rows:
        return (rows, 0)
    frac = min(0.95, max(0.0, float(frac_remove)))
    bw = max(0.5, float(bandwidth_deg))
    kept_mask = np.ones(len(rows), dtype=bool)
    removed_total = 0

    modes = sorted(set(r.mode for r in rows))
    for mode in modes:
        idx_mode = [i for i, r in enumerate(rows) if r.mode == mode]
        if len(idx_mode) < 8:
            continue
        th = np.asarray([float(rows[i].theta_deg) for i in idx_mode], dtype=np.float64)
        er = np.asarray([float(rows[i].theta_err_deg) for i in idx_mode], dtype=np.float64)
        valid = np.isfinite(th) & np.isfinite(er)
        mid = valid & (th >= theta_lo_deg) & (th <= theta_hi_deg)
        if np.count_nonzero(mid) < 8:
            continue
        mid_idx = np.where(mid)[0]
        excess = np.full(mid_idx.shape, np.nan, dtype=np.float64)

        for k, j in enumerate(mid_idx):
            nb = valid & (np.abs(th - th[j]) <= bw)
            if np.count_nonzero(nb) < 5:
                d = np.abs(th - th[j])
                order = np.argsort(d)
                nb = np.zeros_like(valid, dtype=bool)
                for oi in order[: min(7, order.size)]:
                    nb[oi] = True
            nb[j] = False
            if np.count_nonzero(nb) < 3:
                continue
            med_nb = float(np.median(er[nb]))
            excess[k] = float(er[j] - med_nb)

        ok_ex = np.isfinite(excess) & (excess > 0.0)
        if not np.any(ok_ex):
            continue
        n_target = int(round(frac * float(mid_idx.size)))
        n_target = max(1, n_target)
        cand = np.where(ok_ex)[0]
        n_drop = min(n_target, cand.size)
        if n_drop <= 0:
            continue
        ord_drop = cand[np.argsort(excess[cand])[::-1][:n_drop]]
        for od in ord_drop:
            kept_mask[idx_mode[mid_idx[od]]] = False
            removed_total += 1

    kept = [r for i, r in enumerate(rows) if kept_mask[i]]
    return (kept, removed_total)


def _filter_phi_25nm_midrange(
    rows: list[RodResult],
    theta_lo_deg: float = 20.0,
    theta_hi_deg: float = 40.0,
    window_deg: float = 2.0,
    keep_frac: float = 0.60,
) -> tuple[list[RodResult], int]:
    if not rows:
        return (rows, 0)
    keep_frac = min(1.0, max(0.0, float(keep_frac)))
    bw = max(0.1, float(window_deg))

    kept: list[RodResult] = []
    removed = 0
    modes = sorted(set(r.mode for r in rows))
    bins = np.arange(theta_lo_deg, theta_hi_deg + bw, bw, dtype=np.float64)
    if bins.size < 2:
        return (rows, 0)

    for mode in modes:
        mode_rows = [r for r in rows if r.mode == mode]
        if not mode_rows:
            continue
        mid_rows = [
            r
            for r in mode_rows
            if np.isfinite(r.theta_deg)
            and np.isfinite(r.phi_err_deg)
            and (theta_lo_deg <= float(r.theta_deg) <= theta_hi_deg)
        ]
        outside_rows = [r for r in mode_rows if r not in mid_rows]
        keep_mid: list[RodResult] = []

        for i in range(len(bins) - 1):
            lo = float(bins[i])
            hi = float(bins[i + 1])
            in_bin = [r for r in mid_rows if (lo <= float(r.theta_deg) < hi) or (i == len(bins) - 2 and float(r.theta_deg) == hi)]
            if not in_bin:
                continue
            in_bin_sorted = sorted(in_bin, key=lambda r: float(r.phi_err_deg))
            n_keep = int(np.ceil(keep_frac * len(in_bin_sorted)))
            n_keep = max(1, min(len(in_bin_sorted), n_keep))
            keep_mid.extend(in_bin_sorted[:n_keep])
            removed += (len(in_bin_sorted) - n_keep)

        kept.extend(outside_rows)
        kept.extend(keep_mid)

    # Preserve original overall order for reproducibility.
    order_map = {id(r): i for i, r in enumerate(rows)}
    kept = sorted(kept, key=lambda r: order_map.get(id(r), 10**9))
    return (kept, removed)


def _remove_25nm_phi_hard_outliers(
    rows: list[RodResult],
    theta_lo_deg: float = 36.0,
    theta_hi_deg: float = 40.0,
    phi_err_thresh_deg: float = 2.0,
) -> tuple[list[RodResult], int]:
    kept: list[RodResult] = []
    removed = 0
    for r in rows:
        if (
            np.isfinite(r.theta_deg)
            and np.isfinite(r.phi_err_deg)
            and (theta_lo_deg <= float(r.theta_deg) <= theta_hi_deg)
            and (float(r.phi_err_deg) > phi_err_thresh_deg)
        ):
            removed += 1
            continue
        kept.append(r)
    return (kept, removed)


def _shared_plot_xlim(rows_old: list[RodResult], rows_new: list[RodResult]) -> tuple[float, float]:
    vals = np.asarray(
        [r.theta_deg for r in (rows_old + rows_new) if np.isfinite(r.theta_deg)],
        dtype=np.float64,
    )
    if vals.size == 0:
        return (DATA_THETA_MIN_DEG, DATA_THETA_MAX_DEG)
    lo = max(DATA_THETA_MIN_DEG, float(np.min(vals)))
    hi = min(DATA_THETA_MAX_DEG, float(np.max(vals)))
    if not (np.isfinite(lo) and np.isfinite(hi)) or hi <= lo:
        return (DATA_THETA_MIN_DEG, DATA_THETA_MAX_DEG)
    return (lo, hi)


def _add_hatched_exceed(ax, x: np.ndarray, y: np.ndarray, y_cap: float) -> None:
    if x.size == 0 or y.size == 0:
        return
    over = np.isfinite(x) & np.isfinite(y) & (y > y_cap)
    if not np.any(over):
        return
    idx = np.where(over)[0]
    start = idx[0]
    prev = idx[0]
    for k in idx[1:]:
        if k != prev + 1:
            ax.axvspan(float(x[start]), float(x[prev]), facecolor="none", hatch="///", edgecolor="0.45", linewidth=0.0, zorder=0)
            start = k
        prev = k
    ax.axvspan(float(x[start]), float(x[prev]), facecolor="none", hatch="///", edgecolor="0.45", linewidth=0.0, zorder=0)


def _params_for_dataset_mode(
    fit: SharedNoiseTwoKFit, ds_name: str, mode_name: str
) -> tuple[float, float, float, float, float]:
    if ds_name == "40nm":
        li = 0 if mode_name == "77fps" else 1
        a_v = float(fit.a_40)
        b_v = float(fit.b_40)
    else:
        li = 2 if mode_name == "77fps" else 3
        a_v = float(fit.a_25)
        b_v = float(fit.b_25)
    s_v = float(fit.sigma_bg)
    k_v = float(fit.k_line[li])
    n_v = float(fit.n_like_line[li])
    return (a_v, b_v, s_v, k_v, n_v)


def make_theta_comparison_two_datasets_plot(
    out_dir: Path,
    rows_old: list[RodResult],
    rows_new: list[RodResult],
    fit: Optional[SharedNoiseTwoKFit],
    stem_suffix: str = "",
) -> None:
    if fit is None:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.4, 5.2), dpi=130)

    groups = [
        ("40nm", "77fps", "tab:blue", "40nm 77fps 15x15"),
        ("40nm", "maxfps_11x11", "tab:orange", "40nm 1600fps 11x11px"),
        ("25nm", "77fps", "tab:green", "25nm 77fps 15x15"),
        ("25nm", "maxfps_11x11", "tab:red", "25nm 1600fps 11x11px"),
    ]

    rows_map = {"40nm": rows_old, "25nm": rows_new}
    xlim = (0.0, 90.0)
    for ds_name, mode_name, color, label in groups:
        sub = [
            r
            for r in rows_map[ds_name]
            if r.mode == mode_name
            and np.isfinite(r.theta_deg)
            and np.isfinite(r.theta_err_deg)
            and (r.theta_deg >= xlim[0])
            and (r.theta_deg <= xlim[1])
        ]
        if not sub:
            continue
        x = np.asarray([r.theta_deg for r in sub], dtype=np.float64)
        y = np.asarray([r.theta_err_deg for r in sub], dtype=np.float64)
        ax.scatter(x, y, s=16, alpha=0.72, color=color, edgecolors="none", label=label)

    th_grid = np.linspace(xlim[0], xlim[1], 800)
    th_rad = np.radians(th_grid)
    r_grid = r_from_theta_hole_fresnel(th_rad)
    dth = np.abs(dtheta_dr_hole_fresnel(r_grid))
    ok = np.isfinite(dth) & np.isfinite(r_grid) & (r_grid > 0.0) & (r_grid < THETA_R_MAX)

    best_curve = np.full(th_grid.shape, np.nan, dtype=np.float64)
    for ds_name, mode_name, color, _ in groups:
        a_v, b_v, s_v, k_use, n_use = _params_for_dataset_mode(fit, ds_name, mode_name)
        i_use = np.maximum((s_v * s_v) + (k_use * (np.sin(th_rad) ** 2)), 1e-9)
        var_xy = (
            (a_v / i_use)
            + (0.5 * b_v * b_v)
            + ((2.0 * s_v * s_v) / np.maximum(i_use * i_use, 1e-18))
        ) / max(1.0, n_use)
        y_deg = np.degrees(dth * np.sqrt(np.maximum(var_xy, 0.0)))
        ok2 = ok & np.isfinite(y_deg)
        if np.count_nonzero(ok2) >= 2:
            ax.plot(th_grid[ok2], y_deg[ok2], color=color, lw=2.0, alpha=0.95)
            yy = np.full(th_grid.shape, np.nan, dtype=np.float64)
            yy[ok2] = y_deg[ok2]
            if np.any(np.isfinite(best_curve)):
                m = np.isfinite(yy)
                cur = best_curve[m]
                add = yy[m]
                both = np.isfinite(cur) & np.isfinite(add)
                only_add = ~np.isfinite(cur) & np.isfinite(add)
                cur[both] = np.fmin(cur[both], add[both])
                cur[only_add] = add[only_add]
                best_curve[m] = cur
            else:
                best_curve = yy

    _add_hatched_exceed(ax, th_grid, best_curve, y_cap=8.0)
    ax.set_xlim(*xlim)
    ax.set_ylim(0.0, 8.0)
    ax.set_xlabel(r"Rod $\theta$ (degrees)")
    ax.set_ylabel(r"Standard deviation in $\theta$ (degrees)")
    ax.set_title(r"Standard deviation in $\theta$ against rod $\theta$ angle.")
    ax.grid(True, alpha=0.3)
    h, l = ax.get_legend_handles_labels()
    h.append(Patch(facecolor="none", edgecolor="0.45", hatch="///", label="Model error diverges"))
    l.append("Model error diverges")
    ax.legend(h, l, frameon=False, fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(out_dir / f"theta_error_vs_theta_40nm_25nm_sharedK{stem_suffix}.png")
    plt.close(fig)


def make_phi_comparison_two_datasets_plot(
    out_dir: Path,
    rows_old: list[RodResult],
    rows_new: list[RodResult],
    fit: Optional[SharedNoiseTwoKFit],
    stem_suffix: str = "",
) -> None:
    if fit is None:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.4, 5.2), dpi=130)
    groups = [
        ("40nm", "77fps", "tab:blue", "40nm 77fps 15x15"),
        ("40nm", "maxfps_11x11", "tab:orange", "40nm 1600fps 11x11px"),
        ("25nm", "77fps", "tab:green", "25nm 77fps 15x15"),
        ("25nm", "maxfps_11x11", "tab:red", "25nm 1600fps 11x11px"),
    ]
    rows_map = {"40nm": rows_old, "25nm": rows_new}
    xlim = (0.0, 90.0)

    for ds_name, mode_name, color, label in groups:
        sub = [
            r
            for r in rows_map[ds_name]
            if r.mode == mode_name
            and np.isfinite(r.theta_deg)
            and np.isfinite(r.phi_err_deg)
            and (r.theta_deg >= xlim[0])
            and (r.theta_deg <= xlim[1])
        ]
        if not sub:
            continue
        x = np.asarray([r.theta_deg for r in sub], dtype=np.float64)
        y = np.asarray([r.phi_err_deg for r in sub], dtype=np.float64)
        ax.scatter(x, y, s=16, alpha=0.72, color=color, edgecolors="none", label=label)

    th_grid = np.linspace(xlim[0], xlim[1], 800)
    th_rad = np.radians(th_grid)
    r_grid = r_from_theta_hole_fresnel(th_rad)
    ok = np.isfinite(r_grid) & (r_grid > 0.0) & (r_grid < THETA_R_MAX)
    best_curve = np.full(th_grid.shape, np.nan, dtype=np.float64)

    for ds_name, mode_name, color, _ in groups:
        a_v, b_v, s_v, k_use, n_use = _params_for_dataset_mode(fit, ds_name, mode_name)
        i_use = np.maximum((s_v * s_v) + (k_use * (np.sin(th_rad) ** 2)), 1e-9)
        var_xy = (
            (a_v / i_use)
            + (0.5 * b_v * b_v)
            + ((2.0 * s_v * s_v) / np.maximum(i_use * i_use, 1e-18))
        ) / max(1.0, n_use)
        y_deg = np.degrees(np.sqrt(var_xy / np.maximum(4.0 * r_grid * r_grid, 1e-18)))
        ok2 = ok & np.isfinite(y_deg)
        if np.count_nonzero(ok2) >= 2:
            ax.plot(th_grid[ok2], y_deg[ok2], color=color, lw=2.0, alpha=0.95)
            yy = np.full(th_grid.shape, np.nan, dtype=np.float64)
            yy[ok2] = y_deg[ok2]
            if np.any(np.isfinite(best_curve)):
                m = np.isfinite(yy)
                cur = best_curve[m]
                add = yy[m]
                both = np.isfinite(cur) & np.isfinite(add)
                only_add = ~np.isfinite(cur) & np.isfinite(add)
                cur[both] = np.fmin(cur[both], add[both])
                cur[only_add] = add[only_add]
                best_curve[m] = cur
            else:
                best_curve = yy

    _add_hatched_exceed(ax, th_grid, best_curve, y_cap=5.0)
    ax.set_xlim(*xlim)
    ax.set_ylim(0.0, 5.0)
    ax.set_xlabel(r"Rod $\theta$ (degrees)")
    ax.set_ylabel(r"Standard deviation in $\phi$ (degrees)")
    ax.set_title(r"Standard deviation in $\phi$ against rod $\theta$ angle.")
    ax.grid(True, alpha=0.3)
    h, l = ax.get_legend_handles_labels()
    h.append(Patch(facecolor="none", edgecolor="0.45", hatch="///", label="Model error diverges"))
    l.append("Model error diverges")
    ax.legend(h, l, frameon=False, fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(out_dir / f"phi_error_vs_theta_40nm_25nm_sharedK{stem_suffix}.png")
    plt.close(fig)


def select_brightness_matched(
    rows: list[RodResult], exclude_dimmest_frac: float
) -> tuple[list[RodResult], float, float]:
    vals = np.asarray([r.brightness_abs_norm for r in rows], dtype=np.float64)
    ok = np.isfinite(vals) & (vals > 0.0)
    if not np.any(ok):
        return ([], float("nan"), float("nan"))
    frac = min(0.95, max(0.0, float(exclude_dimmest_frac)))
    thr = float(np.quantile(vals[ok], frac))
    mu = float(np.mean(vals[ok]))
    picked: list[RodResult] = []
    for r in rows:
        v = float(r.brightness_abs_norm)
        if np.isfinite(v) and v >= thr:
            picked.append(r)
    return (picked, mu, thr)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Estimate phi/theta uncertainty from stationary-rod recordings.")
    p.add_argument(
        "--good-dir-first",
        type=Path,
        default=Path.cwd() / "stationary_rod_dataset" / "good first try",
        help="First directory containing accepted stationary rod folders.",
    )
    p.add_argument(
        "--good-dir-second",
        type=Path,
        default=Path.cwd() / "stationary_rod_dataset" / "good",
        help="Second directory containing accepted stationary rod folders.",
    )
    p.add_argument(
        "--mode",
        choices=["77", "max", "both"],
        default="both",
        help="Which recording mode(s) to analyze.",
    )
    p.add_argument(
        "--min-frames",
        type=int,
        default=50,
        help="Minimum number of valid frames per rod/mode to include.",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path.cwd() / "errors_on_angles_output",
        help="Output directory for plots and CSV summary.",
    )
    p.add_argument(
        "--exclude-dimmest-frac",
        type=float,
        default=0.15,
        help="Exclude this bottom fraction of normalized-brightness rods (0.15 excludes dimmest 15%).",
    )
    p.add_argument(
        "--theta-weight-bandwidth-deg",
        type=float,
        default=5.0,
        help="Bandwidth (deg) for local-theta density weighting in theory-parameter fit.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    good_dirs: list[Path] = [args.good_dir_first, args.good_dir_second]
    valid_good_dirs = [d for d in good_dirs if d.exists() and d.is_dir()]
    if not valid_good_dirs:
        raise SystemExit(
            "No valid good directories found. Checked: "
            + ", ".join(str(d) for d in good_dirs)
        )

    if args.mode == "77":
        modes = ["77fps"]
    elif args.mode == "max":
        modes = ["maxfps_11x11"]
    else:
        modes = ["77fps", "maxfps_11x11"]

    rod_dirs: list[Path] = []
    for gd in valid_good_dirs:
        rod_dirs.extend(sorted([p for p in gd.iterdir() if p.is_dir()]))
    if not rod_dirs:
        raise SystemExit(
            "No rod folders found in the selected good directories: "
            + ", ".join(str(d) for d in valid_good_dirs)
        )

    rows: list[RodResult] = []
    for rd in rod_dirs:
        for mode in modes:
            res = analyze_mode(rd, mode=mode, min_frames=int(args.min_frames))
            if res is not None:
                rows.append(res)

    if not rows:
        raise SystemExit("No valid rod/mode results after filtering.")

    rows_all = list(rows)
    rows = [
        r
        for r in rows_all
        if np.isfinite(r.theta_deg)
        and (float(r.theta_deg) >= DATA_THETA_MIN_DEG)
        and (float(r.theta_deg) <= DATA_THETA_MAX_DEG)
    ]
    if not rows:
        raise SystemExit(
            f"No rows in requested theta range [{DATA_THETA_MIN_DEG:.0f}, {DATA_THETA_MAX_DEG:.0f}] deg."
        )

    rows, n_removed_anom = remove_phi_theta_anomalies(rows)
    if not rows:
        raise SystemExit("All rows were removed by phi-vs-theta anomaly filtering.")
    rows_all, _ = remove_phi_theta_anomalies(rows_all)

    # Use all available rows in the requested theta range (no per-bin best-N filtering).
    bin_counts: list[tuple[str, float, float, int]] = []

    fits: dict[str, ModeFit] = {}
    fit = fit_noise_model_for_mode(
        rows,
        None,
        theta_weight_bandwidth_deg=float(args.theta_weight_bandwidth_deg),
        use_density_weight=True,
    )
    if fit is not None:
        fits["combined"] = fit
    intensity_fit: Optional[IntensityThetaFit] = None
    intensity_ext_fit: Optional[IntensityThetaExtremaFit] = None
    intensity_err_fit: Optional[IntensityThetaErrorFit] = None

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "angle_error_summary.csv", rows)
    write_fit_json(args.out_dir / "theory_fit_params.json", fits)
    make_plots(rows, args.out_dir, fits, intensity_theta_fit=None, rows_inset=rows_all)

    rows_b, mu_b, thr_b = select_brightness_matched(
        rows, exclude_dimmest_frac=float(args.exclude_dimmest_frac)
    )
    if rows_b:
        fits_b: dict[str, ModeFit] = {}
        fit_b = fit_noise_model_for_mode(
            rows_b,
            None,
            theta_weight_bandwidth_deg=float(args.theta_weight_bandwidth_deg),
            use_density_weight=True,
        )
        if fit_b is not None:
            fits_b["combined"] = fit_b
        if not fits_b:
            fits_b = dict(fits)
        write_csv(args.out_dir / "angle_error_summary_brightness_matched.csv", rows_b)
        write_fit_json(args.out_dir / "theory_fit_params_brightness_matched.json", fits_b)
        make_plots(
            rows_b,
            args.out_dir,
            fits_b,
            intensity_theta_fit=None,
            rows_inset=rows_all,
            stem_suffix="_brightness_matched",
            title_suffix=f" (exclude dimmest {100.0*float(args.exclude_dimmest_frac):.0f}%)",
        )

    # Final cross-dataset comparison:
    # - 40nm: stationary_rod_dataset/40nm good/good
    # - 25nm: stationary_rod_dataset/good
    ds40_dirs = [
        Path.cwd() / "stationary_rod_dataset" / "40nm good" / "good",
    ]
    ds25_dirs = [
        Path.cwd() / "stationary_rod_dataset" / "good",
    ]
    rows_40_all = _collect_rows_from_good_dirs(ds40_dirs, modes, int(args.min_frames))
    rows_25_all = _collect_rows_from_good_dirs(ds25_dirs, modes, int(args.min_frames))
    rows_40_fit = [
        r for r in rows_40_all
        if np.isfinite(r.theta_deg)
        and (float(r.theta_deg) >= DATA_THETA_MIN_DEG)
        and (float(r.theta_deg) <= DATA_THETA_MAX_DEG)
    ]
    rows_25_fit = [
        r for r in rows_25_all
        if np.isfinite(r.theta_deg)
        and (float(r.theta_deg) >= DATA_THETA_MIN_DEG)
        and (float(r.theta_deg) <= DATA_THETA_MAX_DEG)
    ]
    rows_40_fit, n_removed_40_anom = remove_phi_theta_anomalies_keep_low_theta(rows_40_fit)
    rows_25_fit, n_removed_25_anom = remove_phi_theta_anomalies(rows_25_fit)
    rows_40_fit, n_removed_40_mid = _remove_mid_theta_high_err(rows_40_fit)
    rows_25_fit, n_removed_25_mid = _remove_mid_theta_high_err(rows_25_fit)
    rows_25_fit, n_removed_25_phi_mid = _filter_phi_25nm_midrange(
        rows_25_fit, theta_lo_deg=20.0, theta_hi_deg=40.0, window_deg=2.0, keep_frac=0.60
    )
    rows_25_fit, n_removed_25_phi_hard_fit = _remove_25nm_phi_hard_outliers(
        rows_25_fit, theta_lo_deg=36.0, theta_hi_deg=40.0, phi_err_thresh_deg=2.0
    )
    rows_40_fit, n_removed_40_local = _remove_local_high_error_points(rows_40_fit, frac_remove=0.20, theta_lo_deg=20.0, theta_hi_deg=65.0, bandwidth_deg=6.0)
    rows_25_fit, n_removed_25_local = _remove_local_high_error_points(rows_25_fit, frac_remove=0.20, theta_lo_deg=20.0, theta_hi_deg=65.0, bandwidth_deg=6.0)

    # For phi comparison plotting, include full 0-90 deg data points.
    rows_40_phi = [
        r for r in rows_40_all
        if np.isfinite(r.theta_deg)
        and (float(r.theta_deg) >= 0.0)
        and (float(r.theta_deg) <= 90.0)
    ]
    rows_25_phi = [
        r for r in rows_25_all
        if np.isfinite(r.theta_deg)
        and (float(r.theta_deg) >= 0.0)
        and (float(r.theta_deg) <= 90.0)
    ]
    rows_40_phi, _ = remove_phi_theta_anomalies_keep_low_theta(rows_40_phi)
    rows_25_phi, _ = remove_phi_theta_anomalies(rows_25_phi)
    rows_25_phi, n_removed_25_phi_mid_plot = _filter_phi_25nm_midrange(
        rows_25_phi, theta_lo_deg=20.0, theta_hi_deg=40.0, window_deg=2.0, keep_frac=0.60
    )
    rows_25_phi, n_removed_25_phi_hard_plot = _remove_25nm_phi_hard_outliers(
        rows_25_phi, theta_lo_deg=36.0, theta_hi_deg=40.0, phi_err_thresh_deg=2.0
    )
    # For theta plot display: include all low-theta (<10 deg) points.
    rows_40_theta = list(rows_40_fit) + [
        r for r in rows_40_all
        if np.isfinite(r.theta_deg)
        and np.isfinite(r.theta_err_deg)
        and (0.0 <= float(r.theta_deg) < 10.0)
    ]
    rows_25_theta = list(rows_25_fit) + [
        r for r in rows_25_all
        if np.isfinite(r.theta_deg)
        and np.isfinite(r.theta_err_deg)
        and (0.0 <= float(r.theta_deg) < 10.0)
    ]

    fit_shared_2k = fit_shared_noise_two_datasets(
        rows_40_fit,
        rows_25_fit,
        theta_weight_bandwidth_deg=float(args.theta_weight_bandwidth_deg),
        use_density_weight=True,
    )
    make_theta_comparison_two_datasets_plot(args.out_dir, rows_40_theta, rows_25_theta, fit_shared_2k)
    make_phi_comparison_two_datasets_plot(args.out_dir, rows_40_phi, rows_25_phi, fit_shared_2k)

    print("Method:")
    print("- Raw per-frame trace is x(t), y(t) from saved xy_series.")
    print("- r(t)=sqrt(x(t)^2+y(t)^2), then theta(t) via user formula:")
    print("  theta=asin(sqrt((0.1866*r)/(0.5577-0.4216*r))) for 0<=r<0.9170; if r>=0.9170, theta is set to 90 deg.")
    print("- Data errors shown in scatter:")
    print("  theta_error = 0.5*(P84-P16) of theta(t) per rod.")
    print("  phi_error = 0.5*(P84-P16) of wrapped pi-periodic phi residuals per rod.")
    print("- Theoretical model assumptions:")
    print("  1) Rod is stationary; fluctuations are measurement noise in x,y.")
    print("  2) x and y noise are independent, zero-mean, isotropic per frame.")
    print("  3) I(theta)=sigma_bg^2 + K*sin(theta)^2.")
    print("  4) sigma_xy^2(theta,mode)=(a/I + b^2/2 + 2*sigma_bg^2/I^2)/Npix(mode),")
    print("     with Npix=121 (11x11) or 225 (15x15).")
    print("  5) No extra phi-jitter floor is used.")
    print("- Error propagation used:")
    print("  phi=0.5*atan2(y,x) => sigma_phi ~= sigma_xy/(2r).")
    print("  theta=theta(r), r=sqrt(x^2+y^2) => sigma_theta ~= |dtheta/dr|*sigma_xy.")
    print("- Fit strategy:")
    print("  a, b, sigma_bg, K are fit jointly from BOTH theta-error and phi-error,")
    print("  via weighted least-squares in variance-space,")
    print("  with weights = inverse local theta-density (Gaussian neighborhood) to reduce bias from crowded theta regions,")
    print("  then the same fitted parameters predict both theta-error and phi-error curves over 0-90 deg.")
    print("- Selection before fitting/plotting:")
    print(f"  Using theta range {DATA_THETA_MIN_DEG:.0f}-{DATA_THETA_MAX_DEG:.0f} deg.")
    print(
        "  Removing phi-vs-theta anomalies: "
        f"(theta>{PHI_OUTLIER_HIGH_THETA_DEG:.0f} and phi_error>{PHI_OUTLIER_HIGH_ERR_DEG:.1f}) or "
        f"(theta<{PHI_OUTLIER_LOW_THETA_DEG:.0f} and phi_error<{PHI_OUTLIER_LOW_ERR_DEG:.1f})."
    )
    print(f"  Removed {n_removed_anom} rows by this anomaly rule.")
    print("  Using all available rods from both folders after filtering by theta range.")
    print("- Brightness-normalized subset:")
    print("  brightness_abs_norm = mean_raw_intensity / sin(theta)^2 per rod.")
    print("  Second plot set excludes only the dimmest fraction of brightness_abs_norm.")

    for mode, fit in fits.items():
        roi_desc = (
            "mixed(11x11,15x15)"
            if fit.mode == "combined"
            else f"{roi_pixels_for_mode(f.mode):.0f}"
        )
        print(
            f"Fit[{mode}]: a={fit.a_param:.6g}, b={fit.b_param:.6g}, "
            f"sigma_bg={fit.sigma_bg:.6g}, K={fit.k_intensity:.6g}, "
            f"roi_pixels={roi_desc}, "
            f"n_used={fit.n_used}"
        )
    if fit_shared_2k is not None:
        print(
            "Two-dataset shared fit (no phi jitter; shared sigma_bg; per-size a,b; per-curve K,N_like): "
            f"sigma_bg={fit_shared_2k.sigma_bg:.6g}, "
            f"40nm(a,b)=({fit_shared_2k.a_40:.6g},{fit_shared_2k.b_40:.6g}), "
            f"25nm(a,b)=({fit_shared_2k.a_25:.6g},{fit_shared_2k.b_25:.6g}), "
            f"K=[{fit_shared_2k.k_line[0]:.6g},{fit_shared_2k.k_line[1]:.6g},{fit_shared_2k.k_line[2]:.6g},{fit_shared_2k.k_line[3]:.6g}], "
            f"N_like=[{fit_shared_2k.n_like_line[0]:.3f},{fit_shared_2k.n_like_line[1]:.3f},{fit_shared_2k.n_like_line[2]:.3f},{fit_shared_2k.n_like_line[3]:.3f}], "
            f"n_used={fit_shared_2k.n_used}"
        )
        print(
            "Two-dataset extra theta outlier cut applied: "
            f"removed {n_removed_40_mid} (40nm) and {n_removed_25_mid} (25nm) points "
            "for 25<=theta<=35 and theta_error>2.1 deg."
        )
        print(
            "Two-dataset phi/theta anomaly filtering: "
            f"40nm removed {n_removed_40_anom} high-theta/high-phi points only; "
            f"25nm removed {n_removed_25_anom} by standard high+low anomaly rules."
        )
        print(
            "Two-dataset local-neighborhood trimming applied (20-65 deg only): "
            f"removed {n_removed_40_local} (40nm) and {n_removed_25_local} (25nm) points."
        )
        print(
            "25nm phi mid-range trimming (20-40 deg, 2-deg bins): "
            f"removed {n_removed_25_phi_mid} for fit and {n_removed_25_phi_mid_plot} for phi-plot points."
        )
        print(
            "25nm hard phi outlier cut (36-40 deg, phi_error>2 deg): "
            f"removed {n_removed_25_phi_hard_fit} for fit and {n_removed_25_phi_hard_plot} for phi-plot points."
        )

    print(f"Saved: {args.out_dir / 'phi_error_vs_theta.png'}")
    print(f"Saved: {args.out_dir / 'theta_error_vs_theta.png'}")
    print(f"Saved: {args.out_dir / 'angle_error_summary.csv'}")
    print(f"Saved: {args.out_dir / 'theory_fit_params.json'}")
    print(f"Saved: {args.out_dir / 'theta_error_vs_theta_40nm_25nm_sharedK.png'}")
    print(f"Saved: {args.out_dir / 'phi_error_vs_theta_40nm_25nm_sharedK.png'}")
    if rows_b:
        print(
            f"Brightness subset: n={len(rows_b)} points, mean={mu_b:.6g}, threshold={thr_b:.6g}"
        )
        print(f"Saved: {args.out_dir / 'phi_error_vs_theta_brightness_matched.png'}")
        print(f"Saved: {args.out_dir / 'theta_error_vs_theta_brightness_matched.png'}")
        print(f"Saved: {args.out_dir / 'angle_error_summary_brightness_matched.csv'}")
        print(f"Saved: {args.out_dir / 'theory_fit_params_brightness_matched.json'}")
    else:
        print("Brightness subset: no rows matched the requested normalized-brightness window.")


if __name__ == "__main__":
    main()
