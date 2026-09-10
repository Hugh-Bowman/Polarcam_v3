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


DATASETS = {
    "0.6ms": Path(r"C:\Polarcam Software\Polarcam_v3\polarcam_live\datasets\40nm stationary buffer sound on\pending"),
    "1.2ms": Path(r"C:\Polarcam Software\Polarcam_v3\polarcam_live\datasets\40nm stationary buffer sound on\1.2ms"),
}
OUTPUT_DIR = Path(r"C:\Polarcam Software\Polarcam_v3\polarcam_live\datasets\40nm stationary buffer sound on\precision_model_theta_phi_40nm_sound_on_background_corrected_scaled_to_inspection_exposure")
BACKGROUND_PROFILES = {
    "0.6ms": Path(r"C:\Polarcam Software\Polarcam_v3\polarcam_live\backgrounds\background_min_20260805-164541_exp0.6ms_gain1.npy"),
    "1.2ms": Path(r"C:\Polarcam Software\Polarcam_v3\polarcam_live\backgrounds\background_min_20260805-170533_exp1.2ms_gain1.npy"),
}

# Fourkas finite-NA water/buffer curve from the GUI angle reconstruction model.
FOURKAS_WATER = {
    "J1": 0.8151146019095186,
    "J2": 0.07979900400468018,
    "J3": 0.12805600171572504,
    "r_max": 0.8216609883293448,
}


@dataclass
class RodDatum:
    exposure_label: str
    rod: str
    path: Path
    n_frames: int
    r_mean: float
    sigma_xy: float
    intensity_mean: float
    theta_deg: float
    dtheta_dr_rad: float
    sigma_theta_deg: float
    sigma_phi_deg: float
    background_corrected: bool = False
    background_profile_path: str = ""
    background_scale: float = 1.0
    valid_for_theta_fit: bool = True
    theta_bin_deg: int = 0
    selected_for_fit: bool = False


def fourkas_r_from_theta(theta_rad: np.ndarray) -> np.ndarray:
    j1 = float(FOURKAS_WATER["J1"])
    j2 = float(FOURKAS_WATER["J2"])
    j3 = float(FOURKAS_WATER["J3"])
    a = j1 - j2
    b = j1 + j2
    s2 = np.sin(theta_rad) ** 2
    c2 = np.cos(theta_rad) ** 2
    return (a * s2) / np.maximum((b * s2) + (2.0 * j3 * c2), 1e-15)


def build_fourkas_curve() -> dict[str, np.ndarray]:
    theta_deg = np.linspace(0.0, 89.999, 20000)
    theta_rad = np.radians(theta_deg)
    r = fourkas_r_from_theta(theta_rad)
    dtheta_dr_rad = np.gradient(theta_rad, r)
    return {"theta_deg": theta_deg, "theta_rad": theta_rad, "r": r, "dtheta_dr_rad": dtheta_dr_rad}


def interp_curve(curve: dict[str, np.ndarray], r_value: float) -> tuple[float, float]:
    r_max = float(FOURKAS_WATER["r_max"])
    r_use = min(max(float(r_value), 0.0), r_max - 1e-9)
    theta = float(np.interp(r_use, curve["r"], curve["theta_deg"]))
    dtheta = float(np.interp(r_use, curve["r"], curve["dtheta_dr_rad"]))
    return theta, dtheta


def load_xy_series(meta: dict, meta_path: Path) -> np.ndarray:
    xy = meta.get("xy_series")
    if not isinstance(xy, list):
        raise ValueError(f"No xy_series in {meta_path}")
    arr = np.asarray(xy, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] < 2:
        raise ValueError(f"Bad xy_series in {meta_path}: {arr.shape}")
    arr = arr[:, :2]
    ok = np.isfinite(arr[:, 0]) & np.isfinite(arr[:, 1])
    return arr[ok]


def strip_phase_marker(arr: np.ndarray, meta: dict) -> np.ndarray:
    try:
        if bool(meta.get("actual", {}).get("phase_marker_appended", False)) and arr.shape[0] > 1:
            return arr[:-1]
    except Exception:
        pass
    try:
        if bool(meta.get("phase_marker_appended", False)) and arr.shape[0] > 1:
            return arr[:-1]
    except Exception:
        pass
    return arr


def raw_square_bounds(frame_shape: tuple[int, int], roi_meta: dict) -> tuple[int, int, int, int]:
    h, w = int(frame_shape[0]), int(frame_shape[1])
    win_raw = int(roi_meta.get("win_raw", roi_meta.get("w", min(h, w))))
    win_raw = max(2, int(win_raw))
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


def raw_square_from_frame(frame: np.ndarray, roi_meta: dict) -> tuple[np.ndarray, int, int]:
    g = np.asarray(frame)
    if g.ndim != 2:
        g = np.asarray(g[..., 0])
    x0, y0, x1, y1 = raw_square_bounds((g.shape[0], g.shape[1]), roi_meta)
    return np.asarray(g[y0:y1, x0:x1], dtype=np.float64), x0, y0


def xy_phi_stats_from_raw_window(raw_win: np.ndarray, origin_x: int = 0, origin_y: int = 0) -> tuple[float, float, float]:
    g = np.asarray(raw_win)
    if g.ndim != 2 or g.shape[0] < 2 or g.shape[1] < 2:
        return 0.0, 0.0, 0.0
    gf = np.asarray(g, dtype=np.float64)
    px = int(origin_x) % 2
    py = int(origin_y) % 2
    i90 = gf[py::2, px::2]
    i45 = gf[py::2, (1 - px) :: 2]
    i135 = gf[(1 - py) :: 2, px::2]
    i0 = gf[(1 - py) :: 2, (1 - px) :: 2]
    h = min(i0.shape[0], i45.shape[0], i135.shape[0], i90.shape[0])
    w = min(i0.shape[1], i45.shape[1], i135.shape[1], i90.shape[1])
    if h <= 0 or w <= 0:
        return 0.0, 0.0, 0.0
    i0 = i0[:h, :w]
    i90 = i90[:h, :w]
    i45 = i45[:h, :w]
    i135 = i135[:h, :w]
    finite = np.isfinite(i0) & np.isfinite(i90) & np.isfinite(i45) & np.isfinite(i135)
    if not np.any(finite):
        return 0.0, 0.0, 0.0
    m0 = float(np.mean(i0[finite]))
    m90 = float(np.mean(i90[finite]))
    m45 = float(np.mean(i45[finite]))
    m135 = float(np.mean(i135[finite]))
    x = (m0 - m90) / (m0 + m90 + 1e-6)
    y = (m45 - m135) / (m45 + m135 + 1e-6)
    return float(x), float(y), float(0.5 * np.arctan2(y, x))


def estimate_intensity_mean(npy_path: Path, meta: dict) -> float:
    arr = np.load(npy_path, allow_pickle=False)
    arr = strip_phase_marker(arr, meta)
    roi_meta = dict(meta.get("actual", {}).get("roi") or meta.get("roi") or {})
    if not roi_meta:
        return float(np.mean(np.asarray(arr, dtype=np.float64)))
    vals = []
    for frame in arr:
        win, _x0, _y0 = raw_square_from_frame(frame, roi_meta)
        if win.size:
            vals.append(float(np.mean(win)))
    if not vals:
        return float("nan")
    return float(np.mean(vals))


def background_crop_for_stack(meta: dict, background: np.ndarray) -> np.ndarray:
    roi_meta = dict(meta.get("actual", {}).get("roi") or meta.get("roi") or {})
    if not roi_meta:
        raise ValueError("No ROI metadata for background crop")
    x = int(round(float(roi_meta.get("x", 0))))
    y = int(round(float(roi_meta.get("y", 0))))
    w = int(round(float(roi_meta.get("w", 0))))
    h = int(round(float(roi_meta.get("h", 0))))
    if w <= 0 or h <= 0:
        raise ValueError(f"Bad ROI dimensions: {roi_meta}")
    if background.ndim == 3:
        background = np.mean(background, axis=0)
    if background.ndim != 2:
        raise ValueError(f"Background must be 2D or stack, got {background.shape}")
    if y < 0 or x < 0 or y + h > background.shape[0] or x + w > background.shape[1]:
        raise ValueError(f"ROI {roi_meta} outside background shape {background.shape}")
    return np.asarray(background[y : y + h, x : x + w], dtype=np.float64)


def compute_xy_intensity_from_stack(
    npy_path: Path,
    meta: dict,
    background: np.ndarray | None = None,
    background_scale: float = 1.0,
) -> tuple[np.ndarray, float]:
    arr = np.load(npy_path, allow_pickle=False)
    arr = strip_phase_marker(arr, meta)
    arr = np.asarray(arr, dtype=np.float64)
    if background is not None:
        bg_crop = background_crop_for_stack(meta, background) * float(background_scale)
        if tuple(bg_crop.shape) != tuple(arr.shape[1:3]):
            raise ValueError(f"Background crop {bg_crop.shape} does not match rod frame {arr.shape[1:3]}")
        arr = np.maximum(arr - bg_crop[None, :, :], 0.0)
    roi_meta = dict(meta.get("actual", {}).get("roi") or meta.get("roi") or {})
    if not roi_meta:
        roi_meta = {"w": arr.shape[2], "h": arr.shape[1], "win_raw": min(arr.shape[1], arr.shape[2])}
    xy = []
    vals = []
    for frame in arr:
        win, x0, y0 = raw_square_from_frame(frame, roi_meta)
        if win.size:
            x, y, _phi = xy_phi_stats_from_raw_window(win, origin_x=x0, origin_y=y0)
            xy.append((x, y))
            vals.append(float(np.mean(win)))
    return np.asarray(xy, dtype=np.float64), float(np.mean(vals)) if vals else float("nan")


def find_rod_dirs(root: Path) -> list[Path]:
    if not root.exists():
        return []
    dirs = []
    for meta_path in root.rglob("capture_maxfps_15x15_meta.json"):
        rod_dir = meta_path.parent
        if (rod_dir / "capture_maxfps_15x15.npy").exists():
            dirs.append(rod_dir)
    return sorted(set(dirs))


def background_exposure_ms(background_path: Path) -> float | None:
    meta_path = background_path.with_suffix(".json")
    if not meta_path.exists():
        return None
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        value = meta.get("requested", {}).get("exp_ms")
        return float(value) if value is not None else None
    except Exception:
        return None


def load_dataset(label: str, root: Path, curve: dict[str, np.ndarray], background_path: Path | None = None) -> list[RodDatum]:
    rows: list[RodDatum] = []
    background = None
    bg_exp_ms = None
    if background_path is not None:
        background = np.load(background_path, allow_pickle=False)
        bg_exp_ms = background_exposure_ms(background_path)
    for rod_dir in find_rod_dirs(root):
        meta_path = rod_dir / "capture_maxfps_15x15_meta.json"
        npy_path = rod_dir / "capture_maxfps_15x15.npy"
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if background is None:
                xy = load_xy_series(meta, meta_path)
                intensity = estimate_intensity_mean(npy_path, meta)
                bg_scale = 1.0
            else:
                rod_exp_ms = meta.get("requested", {}).get("exp_ms")
                try:
                    rod_exp_ms = float(rod_exp_ms)
                except Exception:
                    rod_exp_ms = None
                if rod_exp_ms is not None and bg_exp_ms is not None and bg_exp_ms > 0.0:
                    bg_scale = float(rod_exp_ms / bg_exp_ms)
                else:
                    bg_scale = 1.0
                xy, intensity = compute_xy_intensity_from_stack(
                    npy_path,
                    meta,
                    background,
                    background_scale=bg_scale,
                )
            if xy.shape[0] < 5:
                continue
            x = xy[:, 0]
            y = xy[:, 1]
            r = np.sqrt(x * x + y * y)
            r_mean = float(np.mean(r))
            if not np.isfinite(r_mean) or r_mean <= 0.0:
                continue
            valid_for_theta_fit = r_mean < float(FOURKAS_WATER["r_max"])
            theta_deg, dtheta_dr_rad = interp_curve(curve, r_mean) if valid_for_theta_fit else (float("nan"), float("nan"))
            sigma_xy = float(np.hypot(np.std(x), np.std(y)))
            if not np.isfinite(intensity) or intensity <= 0.0:
                continue
            sigma_theta = float(np.degrees(abs(dtheta_dr_rad) * sigma_xy)) if valid_for_theta_fit else float("nan")
            sigma_phi = float(np.degrees(sigma_xy / max(2.0 * r_mean, 1e-12)))
            rows.append(
                RodDatum(
                    exposure_label=label,
                    rod=rod_dir.name,
                    path=rod_dir,
                    n_frames=int(xy.shape[0]),
                    r_mean=r_mean,
                    sigma_xy=sigma_xy,
                    intensity_mean=float(intensity),
                    theta_deg=float(theta_deg),
                    dtheta_dr_rad=float(dtheta_dr_rad),
                    sigma_theta_deg=sigma_theta,
                    sigma_phi_deg=sigma_phi,
                    background_corrected=background is not None,
                    background_profile_path=str(background_path) if background_path is not None else "",
                    background_scale=float(bg_scale),
                    valid_for_theta_fit=bool(valid_for_theta_fit),
                )
            )
        except Exception as exc:
            print(f"Skipping {rod_dir}: {exc}")
    return rows


def select_best_per_theta_bin(rows: list[RodDatum], bin_width_deg: float = 5.0, top_n: int = 3) -> list[RodDatum]:
    bins: dict[int, list[RodDatum]] = {}
    for row in rows:
        if not row.valid_for_theta_fit or not np.isfinite(row.theta_deg):
            continue
        b = int(math.floor(float(row.theta_deg) / float(bin_width_deg)) * int(bin_width_deg))
        b = max(0, min(85, b))
        row.theta_bin_deg = b
        bins.setdefault(b, []).append(row)
    selected: list[RodDatum] = []
    for b in sorted(bins):
        ranked = sorted(bins[b], key=lambda r: (r.sigma_xy, r.sigma_theta_deg, r.rod))
        for row in ranked[:top_n]:
            row.selected_for_fit = True
            selected.append(row)
    return selected


def fit_intensity(selected: list[RodDatum]) -> tuple[float, float]:
    theta = np.radians(np.asarray([r.theta_deg for r in selected], dtype=np.float64))
    inten = np.asarray([r.intensity_mean for r in selected], dtype=np.float64)
    x = np.column_stack([np.cos(theta) ** 2, np.sin(theta) ** 2])
    beta, *_ = np.linalg.lstsq(x, inten, rcond=None)
    return float(max(beta[0], 1e-12)), float(max(beta[1], 1e-12))


def predict_for(theta_deg: np.ndarray, curve: dict[str, np.ndarray], b_cos2: float, c_sin2: float, a_var: float) -> dict[str, np.ndarray]:
    theta = np.radians(np.asarray(theta_deg, dtype=np.float64))
    r = fourkas_r_from_theta(theta)
    dtheta_dr = np.interp(theta_deg, curve["theta_deg"], curve["dtheta_dr_rad"])
    inten = b_cos2 * (np.cos(theta) ** 2) + c_sin2 * (np.sin(theta) ** 2)
    inten = np.maximum(inten, 1e-12)
    sigma_xy = math.sqrt(max(float(a_var), 0.0)) / inten
    return {
        "r": r,
        "intensity": inten,
        "sigma_xy": sigma_xy,
        "sigma_theta_deg": np.degrees(np.abs(dtheta_dr) * sigma_xy),
        "sigma_phi_deg": np.degrees(sigma_xy / np.maximum(2.0 * r, 1e-12)),
    }


def fit_variance(selected: list[RodDatum], curve: dict[str, np.ndarray], b_cos2: float, c_sin2: float) -> float:
    theta = np.asarray([r.theta_deg for r in selected], dtype=np.float64)
    pred_unit = predict_for(theta, curve, b_cos2, c_sin2, a_var=1.0)
    base = np.concatenate([pred_unit["sigma_theta_deg"], pred_unit["sigma_phi_deg"]])
    target = np.concatenate(
        [
            np.asarray([r.sigma_theta_deg for r in selected], dtype=np.float64),
            np.asarray([r.sigma_phi_deg for r in selected], dtype=np.float64),
        ]
    )
    ok = np.isfinite(base) & np.isfinite(target) & (base > 0)
    scale = float(np.sum(base[ok] * target[ok]) / max(np.sum(base[ok] * base[ok]), 1e-30))
    return float(max(scale * scale, 0.0))


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_fit(label: str, selected: list[RodDatum], curve: dict[str, np.ndarray], b_cos2: float, c_sin2: float, a_var: float, out_dir: Path) -> None:
    theta_pts = np.asarray([r.theta_deg for r in selected], dtype=np.float64)
    theta_grid = np.linspace(max(0.2, float(np.min(theta_pts)) - 2.0), min(89.5, float(np.max(theta_pts)) + 2.0), 500)
    pred = predict_for(theta_grid, curve, b_cos2, c_sin2, a_var)

    for kind, y_attr, y_pred_key, color, fname, ylabel in [
        ("theta", "sigma_theta_deg", "sigma_theta_deg", "#1f77b4", "std_theta_vs_theta_fit.png", "std theta (deg)"),
        ("phi", "sigma_phi_deg", "sigma_phi_deg", "#2ca02c", "std_phi_vs_theta_fit.png", "std phi (deg)"),
    ]:
        fig, ax = plt.subplots(figsize=(7.2, 4.8))
        ax.scatter(
            theta_pts,
            np.asarray([getattr(r, y_attr) for r in selected], dtype=np.float64),
            s=42,
            color=color,
            alpha=0.85,
            edgecolors="none",
            label="selected rods",
        )
        ax.plot(theta_grid, pred[y_pred_key], color="#d62728", lw=2.0, label="fitted propagated model")
        ax.set_xlabel("theta (deg)")
        ax.set_ylabel(ylabel)
        ax.set_title(f"40nm rods {label}: std {kind} vs theta")
        ax.grid(True, alpha=0.22)
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(out_dir / fname, dpi=220)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    intens = np.asarray([r.intensity_mean for r in selected], dtype=np.float64)
    theta_grid = np.linspace(max(0.0, float(np.min(theta_pts)) - 2.0), min(90.0, float(np.max(theta_pts)) + 2.0), 500)
    theta_rad = np.radians(theta_grid)
    i_grid = b_cos2 * np.cos(theta_rad) ** 2 + c_sin2 * np.sin(theta_rad) ** 2
    ax.scatter(theta_pts, intens, s=42, color="#9467bd", alpha=0.85, edgecolors="none", label="selected rods")
    ax.plot(theta_grid, i_grid, color="#111111", lw=2.0, label="fit: b cos^2(theta) + c sin^2(theta)")
    ax.set_xlabel("theta (deg)")
    ax.set_ylabel("mean background-corrected raw-window intensity")
    ax.set_title(f"40nm rods {label}: intensity vs theta, background corrected")
    ax.grid(True, alpha=0.22)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / "intensity_vs_theta_fit.png", dpi=220)
    plt.close(fig)


def rows_to_dicts(rows: list[RodDatum]) -> list[dict]:
    out = []
    for r in rows:
        out.append(
            {
                "exposure": r.exposure_label,
                "rod": r.rod,
                "path": str(r.path),
                "n_frames": r.n_frames,
                "theta_bin_deg": r.theta_bin_deg,
                "selected_for_fit": r.selected_for_fit,
                "background_corrected": r.background_corrected,
                "background_profile_path": r.background_profile_path,
                "background_scale": r.background_scale,
                "valid_for_theta_fit": r.valid_for_theta_fit,
                "theta_deg": r.theta_deg,
                "r_mean": r.r_mean,
                "sigma_xy": r.sigma_xy,
                "sigma_theta_deg": r.sigma_theta_deg,
                "sigma_phi_deg": r.sigma_phi_deg,
                "intensity_mean_background_corrected_raw_square": r.intensity_mean,
            }
        )
    return out


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    curve = build_fourkas_curve()
    summaries = []
    for label, root in DATASETS.items():
        background_path = BACKGROUND_PROFILES.get(label)
        rows = load_dataset(label, root, curve, background_path=background_path)
        selected = select_best_per_theta_bin(rows, bin_width_deg=5.0, top_n=3)
        if len(selected) < 3:
            raise SystemExit(f"Too few selected rods for {label}: {len(selected)}")
        b_cos2, c_sin2 = fit_intensity(selected)
        a_var = fit_variance(selected, curve, b_cos2, c_sin2)
        out_dir = OUTPUT_DIR / label.replace(".", "p")
        out_dir.mkdir(parents=True, exist_ok=True)
        write_csv(out_dir / "all_rods.csv", rows_to_dicts(rows))
        write_csv(out_dir / "selected_rods_used_for_fit.csv", rows_to_dicts(selected))
        plot_fit(label, selected, curve, b_cos2, c_sin2, a_var, out_dir)
        summary = {
            "label": label,
            "source_dir": str(root),
            "output_dir": str(out_dir),
            "theta_curve": "Fourkas finite-NA water/buffer",
            "fourkas_water_params": FOURKAS_WATER,
            "n_rods_loaded": len(rows),
            "n_selected_for_fit": len(selected),
            "selection": "top 3 lowest sigma_xy rods per 5 degree theta bin",
            "background_correction": {
                "rod_inspection_files_saved_background_subtracted": False,
                "after_the_fact_background_subtraction_applied": background_path is not None,
                "background_profile_path": str(background_path) if background_path is not None else None,
                "method": "Subtract matching full-frame ROI crop from each rod frame, scaled by inspection_exp_ms/background_profile_exp_ms when those differ, and clip negative pixels to zero before recomputing X/Y and intensity.",
            },
            "largest_mean_r_all_loaded": max((r.r_mean for r in rows), default=float("nan")),
            "largest_mean_r_fit_eligible": max((r.r_mean for r in rows if r.valid_for_theta_fit), default=float("nan")),
            "intensity_model": "I(theta) = b cos^2(theta) + c sin^2(theta), fitted to selected rod intensities",
            "variance_model": "sigma_xy^2 = a / I(theta)^2, fitted after propagating to sigma_theta and sigma_phi",
            "fit_params": {
                "a_var_xy_over_I2": a_var,
                "b_cos2_intensity": b_cos2,
                "c_sin2_intensity": c_sin2,
            },
        }
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        summaries.append(summary)
        print(
            f"{label}: loaded {len(rows)}, selected {len(selected)}, "
            f"max_r={summary['largest_mean_r_all_loaded']:.6g}, "
            f"a={a_var:.6g}, b_cos2={b_cos2:.6g}, c_sin2={c_sin2:.6g}, out={out_dir}"
        )
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
