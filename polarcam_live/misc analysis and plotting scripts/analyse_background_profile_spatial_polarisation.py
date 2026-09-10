from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
PROFILE_DIR = (
    ROOT
    / "background characterisation"
    / "16072026 backgrounds"
    / "background profiles used for percentage heatmaps"
)
OUT_DIR = PROFILE_DIR / "spatial_fourier_polarisation_analysis"

FILES = [
    ("uncorrected_background", PROFILE_DIR / "uncorrected_background__frame_stack_20260716-140000_average_frame.npy"),
    (
        "different_coverslip_background_subtracted",
        PROFILE_DIR / "different_coverslip_background_subtracted__frame_stack_20260716-140137_average_frame.npy",
    ),
    (
        "same_point_same_coverslip_best_case_background_subtracted",
        PROFILE_DIR / "same_point_same_coverslip_best_case_background_subtracted__frame_stack_20260716-140047_average_frame.npy",
    ),
]


def split_channels(img: np.ndarray) -> dict[str, np.ndarray]:
    # Full-frame convention used by angle_distribution_analysis.py.
    return {
        "I0": img[0::2, 0::2],
        "I45": img[0::2, 1::2],
        "I135": img[1::2, 0::2],
        "I90": img[1::2, 1::2],
    }


def describe(arr: np.ndarray) -> dict[str, float]:
    a = np.asarray(arr, dtype=np.float64)
    return {
        "mean": float(np.mean(a)),
        "std": float(np.std(a)),
        "p01": float(np.percentile(a, 1)),
        "p50": float(np.percentile(a, 50)),
        "p99": float(np.percentile(a, 99)),
    }


def radial_power_spectrum(img: np.ndarray, pixel_pitch_raw: float = 1.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Remove the DC component and apodize so edge steps do not dominate.
    a = np.asarray(img, dtype=np.float64)
    a = a - np.mean(a)
    wy = np.hanning(a.shape[0])
    wx = np.hanning(a.shape[1])
    a = a * wy[:, None] * wx[None, :]
    fft = np.fft.rfft2(a)
    power = np.abs(fft) ** 2
    fy = np.fft.fftfreq(a.shape[0], d=pixel_pitch_raw)
    fx = np.fft.rfftfreq(a.shape[1], d=pixel_pitch_raw)
    rr = np.sqrt(fy[:, None] ** 2 + fx[None, :] ** 2)
    fmax = float(np.max(rr))
    bins = np.linspace(0.0, fmax, 400)
    which = np.digitize(rr.ravel(), bins) - 1
    valid = (which >= 0) & (which < len(bins) - 1)
    counts = np.bincount(which[valid], minlength=len(bins) - 1)
    sums = np.bincount(which[valid], weights=power.ravel()[valid], minlength=len(bins) - 1)
    radial = sums / np.maximum(counts, 1)
    centers = 0.5 * (bins[:-1] + bins[1:])
    wavelength = np.divide(1.0, centers, out=np.full_like(centers, np.inf), where=centers > 0)
    return centers, wavelength, radial


def dominant_scales(freq: np.ndarray, wavelength: np.ndarray, power: np.ndarray) -> dict[str, float]:
    valid = np.isfinite(wavelength) & (wavelength >= 4.0) & (wavelength <= 1500.0) & (freq > 0)
    if not np.any(valid):
        return {}
    f = freq[valid]
    wl = wavelength[valid]
    p = power[valid]
    p_norm = p / max(float(np.max(p)), 1e-30)
    peak_i = int(np.argmax(p))
    cum = np.cumsum(p)
    total = float(cum[-1])
    def wl_at_frac(frac: float) -> float:
        idx = int(np.searchsorted(cum, frac * total))
        idx = max(0, min(idx, len(wl) - 1))
        return float(wl[idx])
    return {
        "dominant_wavelength_px": float(wl[peak_i]),
        "dominant_frequency_cycles_per_px": float(f[peak_i]),
        "power_weighted_mean_wavelength_px": float(np.sum(wl * p) / max(np.sum(p), 1e-30)),
        "wavelength_at_50pct_cumulative_power_px": wl_at_frac(0.5),
        "wavelength_at_90pct_cumulative_power_px": wl_at_frac(0.9),
        "normalised_peak_power": float(p_norm[peak_i]),
    }


def channel_correlation(channels: dict[str, np.ndarray]) -> np.ndarray:
    names = ["I0", "I45", "I135", "I90"]
    vals = [channels[n].astype(np.float64).ravel() for n in names]
    min_n = min(v.size for v in vals)
    vals = [v[:min_n] for v in vals]
    mat = np.vstack(vals)
    return np.corrcoef(mat)


def anisotropy_fields(ch: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    h = min(v.shape[0] for v in ch.values())
    w = min(v.shape[1] for v in ch.values())
    i0 = ch["I0"][:h, :w].astype(np.float64)
    i90 = ch["I90"][:h, :w].astype(np.float64)
    i45 = ch["I45"][:h, :w].astype(np.float64)
    i135 = ch["I135"][:h, :w].astype(np.float64)
    eps = 1e-9
    x = (i0 - i90) / (i0 + i90 + eps)
    y = (i45 - i135) / (i45 + i135 + eps)
    r = np.sqrt(x * x + y * y)
    return x, y, r


def block_mean_intensity_image(ch: dict[str, np.ndarray]) -> np.ndarray:
    h = min(v.shape[0] for v in ch.values())
    w = min(v.shape[1] for v in ch.values())
    return (
        ch["I0"][:h, :w]
        + ch["I45"][:h, :w]
        + ch["I135"][:h, :w]
        + ch["I90"][:h, :w]
    ).astype(np.float64) / 4.0


def neighbour_difference_stats(img: np.ndarray) -> dict[str, float]:
    a = np.asarray(img, dtype=np.float64)
    dx = np.diff(a, axis=1).ravel()
    dy = np.diff(a, axis=0).ravel()
    diffs = np.concatenate([dx, dy])
    abs_diffs = np.abs(diffs)
    return {
        "neighbour_diff_std": float(np.std(diffs)),
        "neighbour_absdiff_mean": float(np.mean(abs_diffs)),
        "neighbour_absdiff_p50": float(np.percentile(abs_diffs, 50)),
        "neighbour_absdiff_p95": float(np.percentile(abs_diffs, 95)),
        "neighbour_absdiff_p99": float(np.percentile(abs_diffs, 99)),
        "neighbour_absdiff_mean_percent_of_image_mean": float(100.0 * np.mean(abs_diffs) / max(np.mean(a), 1e-12)),
        "neighbour_absdiff_p95_percent_of_image_mean": float(100.0 * np.percentile(abs_diffs, 95) / max(np.mean(a), 1e-12)),
    }


def write_heat_image(path: Path, img: np.ndarray, title: str, cmap: str = "viridis") -> None:
    lo, hi = np.percentile(img, [1, 99])
    plt.figure(figsize=(8, 6))
    plt.imshow(img, cmap=cmap, vmin=lo, vmax=hi)
    plt.title(title)
    plt.colorbar(label="value")
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    channel_rows = []
    corr_rows = []
    spectra_rows = []
    names = ["I0", "I45", "I135", "I90"]

    for label, path in FILES:
        img = np.asarray(np.load(path, mmap_mode="r"), dtype=np.float64)
        ch = split_channels(img)
        x, y, r = anisotropy_fields(ch)
        block_mean = block_mean_intensity_image(ch)
        total = sum(v[: x.shape[0], : x.shape[1]] for v in ch.values())
        total_desc = describe(img)
        x_desc = describe(x)
        y_desc = describe(y)
        r_desc = describe(r)
        summary_rows.append(
            {
                "condition": label,
                "source_file": path.name,
                "shape": img.shape,
                "pixel_mean": total_desc["mean"],
                "pixel_std": total_desc["std"],
                "x_pol_mean": x_desc["mean"],
                "x_pol_std": x_desc["std"],
                "y_pol_mean": y_desc["mean"],
                "y_pol_std": y_desc["std"],
                "r_pol_mean": r_desc["mean"],
                "r_pol_p50": r_desc["p50"],
                "r_pol_p99": r_desc["p99"],
                **{f"block_mean_{k}": v for k, v in describe(block_mean).items()},
                **neighbour_difference_stats(block_mean),
            }
        )
        for cname, arr in ch.items():
            d = describe(arr)
            channel_rows.append({"condition": label, "channel": cname, **d})

        corr = channel_correlation(ch)
        for i, a in enumerate(names):
            for j, b in enumerate(names):
                corr_rows.append({"condition": label, "channel_a": a, "channel_b": b, "correlation": float(corr[i, j])})

        write_heat_image(OUT_DIR / f"{label}_profile.png", img, f"{label}: stored profile")
        write_heat_image(OUT_DIR / f"{label}_x_pol.png", x, f"{label}: X=(I0-I90)/(I0+I90)", cmap="coolwarm")
        write_heat_image(OUT_DIR / f"{label}_y_pol.png", y, f"{label}: Y=(I45-I135)/(I45+I135)", cmap="coolwarm")
        write_heat_image(OUT_DIR / f"{label}_r_pol.png", r, f"{label}: polarisation anisotropy radius")
        write_heat_image(
            OUT_DIR / f"{label}_2x2_block_mean_intensity.png",
            block_mean,
            f"{label}: 2x2 block mean intensity",
        )
        write_heat_image(
            OUT_DIR / f"{label}_2x2_block_mean_neighbour_absdiff.png",
            np.hypot(
                np.pad(np.diff(block_mean, axis=1), ((0, 0), (0, 1)), mode="constant"),
                np.pad(np.diff(block_mean, axis=0), ((0, 1), (0, 0)), mode="constant"),
            ),
            f"{label}: neighbour variation in 2x2 block mean",
        )

        for spectrum_name, spectrum_img in [
            ("profile", img),
            ("total_2x2_intensity", total),
            ("mean_2x2_block_intensity", block_mean),
            ("r_pol", r),
        ]:
            freq, wavelength, power = radial_power_spectrum(spectrum_img)
            scale = dominant_scales(freq, wavelength, power)
            spectra_rows.append({"condition": label, "image": spectrum_name, **scale})
            out_csv = OUT_DIR / f"{label}_{spectrum_name}_radial_power_spectrum.csv"
            with out_csv.open("w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["frequency_cycles_per_raw_px", "wavelength_raw_px", "radial_power"])
                for fv, wv, pv in zip(freq, wavelength, power):
                    writer.writerow([fv, wv, pv])
            valid = np.isfinite(wavelength) & (wavelength >= 2) & (wavelength <= 2000)
            plt.figure(figsize=(7.5, 5))
            plt.loglog(wavelength[valid], power[valid])
            plt.gca().invert_xaxis()
            plt.xlabel("Spatial wavelength (raw pixels)")
            plt.ylabel("Radial power")
            plt.title(f"{label}: {spectrum_name} radial Fourier spectrum")
            plt.tight_layout()
            plt.savefig(OUT_DIR / f"{label}_{spectrum_name}_radial_power_spectrum.png", dpi=220)
            plt.close()

    def write_dicts(filename: str, rows: list[dict]) -> None:
        with (OUT_DIR / filename).open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    write_dicts("profile_polarisation_summary.csv", summary_rows)
    write_dicts("channel_statistics.csv", channel_rows)
    write_dicts("channel_correlation_matrix_long.csv", corr_rows)
    write_dicts("fourier_spatial_scale_summary.csv", spectra_rows)

    lines = [
        "Background profile spatial and polarisation analysis",
        "",
        f"Input folder: {PROFILE_DIR}",
        f"Output folder: {OUT_DIR}",
        "",
        "Channel convention:",
        "- I0=img[0::2,0::2], I45=img[0::2,1::2], I135=img[1::2,0::2], I90=img[1::2,1::2].",
        "- Polarisation fields use X=(I0-I90)/(I0+I90), Y=(I45-I135)/(I45+I135), r=sqrt(X^2+Y^2).",
        "",
        "Interpretation guide:",
        "- If X and Y means/stds are small compared with 1, the background is mostly unpolarised.",
        "- Strong channel correlations mean the four channels share common spatial background structure.",
        "- The Fourier dominant wavelength is the strongest radial spatial period after DC removal and Hann windowing.",
        "",
        "Key outputs:",
        "- profile_polarisation_summary.csv",
        "- channel_statistics.csv",
        "- channel_correlation_matrix_long.csv",
        "- fourier_spatial_scale_summary.csv",
        "- *_2x2_block_mean_intensity.png",
        "- *_2x2_block_mean_neighbour_absdiff.png",
        "- *_radial_power_spectrum.csv/png",
        "- *_x_pol.png, *_y_pol.png, *_r_pol.png",
    ]
    (OUT_DIR / "README_spatial_polarisation_analysis.txt").write_text("\n".join(lines) + "\n")
    print(f"Wrote outputs to {OUT_DIR}")


if __name__ == "__main__":
    main()
