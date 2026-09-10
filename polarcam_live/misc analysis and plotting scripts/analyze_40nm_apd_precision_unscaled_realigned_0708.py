from __future__ import annotations

import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from nptdms import TdmsFile
from scipy.signal import butter, sosfiltfilt

import analyze_40nm_bg_subtracted_sound_precision_new as water_curve


DATA_DIR = Path(r"E:\40nm apd precision realigned 0708")
OUT_DIR = DATA_DIR / "analysis_unscaled"

# User-requested unscaled channel interpretation.
CHANNEL_MAP = {
    "I90": "ai0",
    "I45": "ai1",
    "I135": "ai2",
    "I0": "ai3",
}

BANDPASS_LOW_HZ = 1.0
R_SIGMA_LO_PCT = 16.0
R_SIGMA_HI_PCT = 84.0
R_SIGMA_DIVISOR = 2.0
PAIR_SUM_RATIO_MIN = 0.89
PAIR_SUM_RATIO_MAX = 1.10


@dataclass
class ApdRow:
    file: Path
    included: bool
    discard_reason: str
    n_samples: int
    sample_rate_hz: float
    duration_s: float
    pair_sum_ratio_45_135_over_0_90: float
    mean_I0: float
    mean_I45: float
    mean_I90: float
    mean_I135: float
    x_mean: float
    y_mean: float
    r_mean: float
    r_max_observed: float
    r_shift_to_rmax: float
    r_p16_raw: float
    r_p84_raw: float
    r_p16: float
    r_p84: float
    sigma_r_p16_p84: float
    theta_deg: float
    theta_p16_deg: float
    theta_p84_deg: float
    sigma_theta_deg: float
    sigma_theta_deg_unfiltered_percentile: float
    phi_mean_deg: float
    sigma_phi_deg: float
    sigma_phi_deg_unfiltered_circular: float


def natural_key(path: Path) -> list[int | str]:
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", path.stem)]


def short_name(name: str) -> str:
    return name.rsplit("/", 1)[-1].lower()


def read_channels(path: Path) -> tuple[dict[str, np.ndarray], float]:
    tdms = TdmsFile.read(path)
    group = tdms.groups()[0]
    raw = {short_name(ch.name): ch for ch in group.channels()}
    out: dict[str, np.ndarray] = {}
    fs = float("nan")
    min_len = None
    for logical, ai_name in CHANNEL_MAP.items():
        if ai_name not in raw:
            raise ValueError(f"{path.name} missing channel {ai_name}")
        ch = raw[ai_name]
        vals = np.asarray(ch[:], dtype=np.float64)
        out[logical] = vals
        min_len = vals.size if min_len is None else min(min_len, vals.size)
        dt = ch.properties.get("wf_increment")
        if dt:
            fs = 1.0 / float(dt)
    if min_len is None or min_len < 16:
        raise ValueError(f"{path.name} has too few samples")
    return {key: vals[:min_len] for key, vals in out.items()}, fs


def circular_phi_stats(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    phi = 0.5 * np.arctan2(y, x)
    z = np.exp(2.0j * phi)
    z_mean = np.mean(z)
    mean_phi = 0.5 * float(np.angle(z_mean))
    resultant = float(np.clip(np.abs(z_mean), 1e-12, 1.0))
    sigma_phi = 0.5 * math.sqrt(max(0.0, -2.0 * math.log(resultant)))
    return float(np.degrees(mean_phi)), float(np.degrees(sigma_phi))


def highpass_std_deg(series_deg: np.ndarray, fs: float) -> float:
    values = np.asarray(series_deg, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size < 16:
        return float("nan")
    values = values - float(np.mean(values))
    if not np.isfinite(fs) or fs <= 2.0 * BANDPASS_LOW_HZ:
        return float(np.std(values))
    sos = butter(2, BANDPASS_LOW_HZ, btype="highpass", fs=fs, output="sos")
    return float(np.std(sosfiltfilt(sos, values)))


def analyse_file(path: Path, curve: dict[str, np.ndarray]) -> ApdRow:
    ch, fs = read_channels(path)
    i0 = ch["I0"]
    i90 = ch["I90"]
    i45 = ch["I45"]
    i135 = ch["I135"]
    eps = 1e-15
    x = (i0 - i90) / np.maximum(i0 + i90, eps)
    y = (i45 - i135) / np.maximum(i45 + i135, eps)
    ok = np.isfinite(x) & np.isfinite(y)
    x = x[ok]
    y = y[ok]
    i0 = i0[ok]
    i90 = i90[ok]
    i45 = i45[ok]
    i135 = i135[ok]
    if x.size < 16:
        raise ValueError(f"{path.name} has too few finite XY samples")
    r = np.hypot(x, y)
    r_mean = float(np.mean(r))
    r_max_observed = float(np.max(r))
    ratio = float(np.mean(i135 + i45) / max(np.mean(i90 + i0), eps))
    ratio_in_range = PAIR_SUM_RATIO_MIN <= ratio <= PAIR_SUM_RATIO_MAX
    included = bool(r_mean <= 1.0 and ratio_in_range)
    discard_reasons = []
    if r_mean > 1.0:
        discard_reasons.append(f"mean r={r_mean:.6g} exceeds 1")
    if not ratio_in_range:
        discard_reasons.append(
            f"pair-sum ratio={ratio:.6g} outside {PAIR_SUM_RATIO_MIN:.2f}-{PAIR_SUM_RATIO_MAX:.2f}"
        )
    discard_reason = "; ".join(discard_reasons)

    r_model_max = float(water_curve.FOURKAS_WATER["r_max"])
    theta_pair = water_curve.theta_from_r(curve, min(r_mean, r_model_max))
    theta_deg = 90.0 if r_mean >= r_model_max else float(theta_pair[0]) if theta_pair else float("nan")
    r_p16_raw = float(np.percentile(r, R_SIGMA_LO_PCT))
    r_p84_raw = float(np.percentile(r, R_SIGMA_HI_PCT))
    r_shift = max(0.0, r_mean - r_model_max)
    r_for_theta = np.clip(r - r_shift, 0.0, r_model_max)
    r_p16 = float(np.percentile(r_for_theta, R_SIGMA_LO_PCT))
    r_p84 = float(np.percentile(r_for_theta, R_SIGMA_HI_PCT))
    theta_p16 = water_curve.theta_value_from_r(curve, r_p16)
    theta_p84 = water_curve.theta_value_from_r(curve, r_p84)
    sigma_theta_unfiltered = float(abs(theta_p84 - theta_p16) / R_SIGMA_DIVISOR)
    theta_series = water_curve.theta_series_from_r(curve, r_for_theta)
    sigma_theta = highpass_std_deg(theta_series, fs)

    phi_wrapped = 0.5 * np.arctan2(y, x)
    phi_unwrapped = 0.5 * np.unwrap(np.angle(np.exp(2.0j * phi_wrapped)))
    phi_mean, sigma_phi_unfiltered = circular_phi_stats(x, y)
    sigma_phi = highpass_std_deg(np.degrees(phi_unwrapped - float(np.mean(phi_unwrapped))), fs)
    n = int(x.size)
    duration = float(n / fs) if np.isfinite(fs) and fs > 0 else float("nan")
    return ApdRow(
        file=path,
        included=included,
        discard_reason=discard_reason,
        n_samples=n,
        sample_rate_hz=float(fs),
        duration_s=duration,
        pair_sum_ratio_45_135_over_0_90=ratio,
        mean_I0=float(np.mean(i0)),
        mean_I45=float(np.mean(i45)),
        mean_I90=float(np.mean(i90)),
        mean_I135=float(np.mean(i135)),
        x_mean=float(np.mean(x)),
        y_mean=float(np.mean(y)),
        r_mean=r_mean,
        r_max_observed=r_max_observed,
        r_shift_to_rmax=float(r_shift),
        r_p16_raw=r_p16_raw,
        r_p84_raw=r_p84_raw,
        r_p16=r_p16,
        r_p84=r_p84,
        sigma_r_p16_p84=float((r_p84 - r_p16) / R_SIGMA_DIVISOR),
        theta_deg=float(theta_deg),
        theta_p16_deg=float(theta_p16),
        theta_p84_deg=float(theta_p84),
        sigma_theta_deg=sigma_theta,
        sigma_theta_deg_unfiltered_percentile=sigma_theta_unfiltered,
        phi_mean_deg=phi_mean,
        sigma_phi_deg=sigma_phi,
        sigma_phi_deg_unfiltered_circular=sigma_phi_unfiltered,
    )


def write_csv(rows: list[ApdRow], path: Path) -> None:
    fields = list(ApdRow.__dataclass_fields__.keys())
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            d = row.__dict__.copy()
            d["file"] = str(row.file)
            writer.writerow(d)


def add_low_theta_region(ax: plt.Axes) -> None:
    ax.axvspan(
        0.0,
        10.0,
        facecolor="#777777",
        alpha=0.08,
        hatch="///",
        edgecolor="#777777",
        linewidth=0.0,
        label=r"error diverges at low $\theta$",
        zorder=0,
    )


def plot_vs_theta(rows: list[ApdRow], attr: str, ylabel: str, title: str, filename: str) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    add_low_theta_region(ax)
    x = np.asarray([row.theta_deg for row in rows], dtype=np.float64)
    y = np.asarray([getattr(row, attr) for row in rows], dtype=np.float64)
    ax.scatter(x, y, s=50, marker="D", color="#2a7f62", edgecolors="white", linewidths=0.45, alpha=0.86, label="APD rods")
    finite = y[np.isfinite(y)]
    ax.set_xlim(0.0, 90.0)
    ax.set_ylim(0.0, max(float(np.max(finite) * 1.12) if finite.size else 1.0, 1.0))
    ax.set_xlabel(r"$\theta$ (deg)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.24)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT_DIR / filename, dpi=220)
    plt.close(fig)


def plot_phi_vs_phi(rows: list[ApdRow]) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    x = np.asarray([row.phi_mean_deg for row in rows], dtype=np.float64)
    y = np.asarray([row.sigma_phi_deg for row in rows], dtype=np.float64)
    ax.scatter(x, y, s=50, marker="D", color="#2a7f62", edgecolors="white", linewidths=0.45, alpha=0.86)
    finite = y[np.isfinite(y)]
    ax.set_ylim(0.0, max(float(np.max(finite) * 1.12) if finite.size else 1.0, 1.0))
    ax.set_xlabel(r"$\phi$ (deg)")
    ax.set_ylabel(r"$\sigma_\phi$ (deg)")
    ax.set_title(r"40 nm APD unscaled: $\sigma_\phi$ vs $\phi$, 1 Hz-Nyquist bandwidth")
    ax.grid(True, alpha=0.24)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "apd_unscaled_sigma_phi_vs_phi.png", dpi=220)
    plt.close(fig)


def plot_xy(rows: list[ApdRow], discarded: list[ApdRow]) -> None:
    fig, ax = plt.subplots(figsize=(5.4, 5.4))
    if discarded:
        ax.scatter([r.x_mean for r in discarded], [r.y_mean for r in discarded], s=45, marker="x", color="#b23a48", label="discarded")
    ax.scatter([r.x_mean for r in rows], [r.y_mean for r in rows], s=52, marker="D", color="#2a7f62", edgecolors="white", linewidths=0.45, label="included")
    ax.axhline(0, color="#777777", lw=0.8, alpha=0.5)
    ax.axvline(0, color="#777777", lw=0.8, alpha=0.5)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("X = (I0 - I90) / (I0 + I90)")
    ax.set_ylabel("Y = (I45 - I135) / (I45 + I135)")
    ax.set_title("40 nm APD unscaled: mean XY")
    ax.grid(True, alpha=0.24)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "apd_unscaled_mean_xy.png", dpi=220)
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    curve = water_curve.build_curve()
    rows: list[ApdRow] = []
    skipped = []
    for path in sorted(DATA_DIR.glob("*.tdms"), key=natural_key):
        try:
            rows.append(analyse_file(path, curve))
        except Exception as exc:
            skipped.append({"file": str(path), "reason": str(exc)})
            print(f"Skipping {path}: {exc}")
    included = [row for row in rows if row.included]
    discarded = [row for row in rows if not row.included]
    write_csv(rows, OUT_DIR / "apd_unscaled_all_recordings.csv")
    write_csv(included, OUT_DIR / "apd_unscaled_precision_points.csv")
    plot_vs_theta(included, "sigma_theta_deg", r"$\sigma_\theta$ (deg)", r"40 nm APD unscaled: $\sigma_\theta$ vs $\theta$, 1 Hz-Nyquist bandwidth", "apd_unscaled_sigma_theta_vs_theta.png")
    plot_vs_theta(included, "sigma_phi_deg", r"$\sigma_\phi$ (deg)", r"40 nm APD unscaled: $\sigma_\phi$ vs $\theta$, 1 Hz-Nyquist bandwidth", "apd_unscaled_sigma_phi_vs_theta.png")
    plot_phi_vs_phi(included)
    plot_xy(included, discarded)
    ratio_rows = [{"file": row.file.name, "ratio_45_135_over_0_90": row.pair_sum_ratio_45_135_over_0_90, "included": row.included, "r_mean": row.r_mean} for row in rows]
    summary = {
        "source_dir": str(DATA_DIR),
        "output_dir": str(OUT_DIR),
        "n_tdms_found": len(rows) + len(skipped),
        "n_read": len(rows),
        "n_included": len(included),
        "n_discarded": len(discarded),
        "n_discarded_mean_r_gt_1": sum(1 for row in discarded if row.r_mean > 1.0),
        "n_discarded_pair_sum_ratio_outside_range": sum(
            1 for row in discarded
            if not (PAIR_SUM_RATIO_MIN <= row.pair_sum_ratio_45_135_over_0_90 <= PAIR_SUM_RATIO_MAX)
        ),
        "discard_rule": (
            "recording discarded from precision plots if mean r exceeds 1 or if "
            f"pair-sum ratio is outside {PAIR_SUM_RATIO_MIN:.2f}-{PAIR_SUM_RATIO_MAX:.2f}"
        ),
        "pair_sum_ratio_filter": {
            "min": PAIR_SUM_RATIO_MIN,
            "max": PAIR_SUM_RATIO_MAX,
        },
        "channel_scaling": "none",
        "channel_map": CHANNEL_MAP,
        "ratio_definition": "(mean I135 + mean I45) / (mean I90 + mean I0), unscaled channels",
        "ratios": ratio_rows,
        "skipped": skipped,
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Output: {OUT_DIR}")
    print(f"Read {len(rows)} TDMS files, included {len(included)}, discarded {len(discarded)}")
    print("RATIOS (I135+I45):(I90+I0)")
    for row in rows:
        status = "included" if row.included else "discarded"
        print(f"{row.file.name}: {row.pair_sum_ratio_45_135_over_0_90:.6f} ({status}, r_mean={row.r_mean:.6f})")


if __name__ == "__main__":
    main()
