from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
import numpy as np
from scipy.ndimage import uniform_filter

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
DEFAULT_ANALYSIS_DIR = (
    ROOT
    / "background characterisation"
    / "z scan of a rod"
    / "pending"
    / "z_scan_interference_analysis_20260720-140342"
)
CENTRAL_WINDOW = 11
N_REPLICATES = 100_000


def strip_marker(arr: np.ndarray) -> np.ndarray:
    a = np.asarray(arr)
    if a.ndim < 3 or a.shape[0] < 2:
        return a
    marker = np.asarray(a[-1])
    if marker.ndim != 2:
        return a
    nz = np.argwhere(marker != 0)
    if nz.shape[0] != 1:
        return a
    my, mx = int(nz[0][0]), int(nz[0][1])
    if float(marker[my, mx]) == 1.0 and float(np.sum(marker, dtype=np.float64)) == 1.0:
        return np.asarray(a[:-1])
    return a


def central_11_from_stack_mean(stack_path: Path) -> np.ndarray:
    stack = strip_marker(np.load(stack_path, mmap_mode="r"))
    mean_frame = np.mean(np.asarray(stack[:, :14, :14], dtype=np.float64), axis=0)
    h, w = mean_frame.shape
    y0 = (h - CENTRAL_WINDOW) // 2
    x0 = (w - CENTRAL_WINDOW) // 2
    return mean_frame[y0 : y0 + CENTRAL_WINDOW, x0 : x0 + CENTRAL_WINDOW]


def phase_blocks(shape: tuple[int, int], block: int, rng: np.random.Generator) -> np.ndarray:
    h, w = shape
    if block <= 1:
        return rng.uniform(0.0, 2.0 * np.pi, size=shape)
    if block >= max(h, w):
        return np.full(shape, rng.uniform(0.0, 2.0 * np.pi), dtype=np.float64)
    ny = int(np.ceil(h / block))
    nx = int(np.ceil(w / block))
    coarse = rng.uniform(0.0, 2.0 * np.pi, size=(ny, nx))
    return np.repeat(np.repeat(coarse, block, axis=0), block, axis=1)[:h, :w]


def phase_convolved(shape: tuple[int, int], window: int, rng: np.random.Generator) -> np.ndarray:
    phi = rng.uniform(0.0, 2.0 * np.pi, size=shape)
    if window <= 1:
        return phi
    z = np.exp(1j * phi)
    zr = uniform_filter(z.real, size=window, mode="reflect")
    zi = uniform_filter(z.imag, size=window, mode="reflect")
    return np.angle(zr + 1j * zi)


def simulate(R: np.ndarray, B: np.ndarray, phase_window: int, n: int, rng: np.random.Generator) -> np.ndarray:
    # For global z phase delta, interference is:
    # I(delta)=sum(R+B)/N + (2/N) Re[exp(i delta) sum_i sqrt(R_i B_i) exp(i phi_i)].
    # Therefore the slope-corrected peak-to-minimum variation is 4*|S|/N.
    weight = np.sqrt(np.clip(R, 0.0, None) * np.clip(B, 0.0, None))
    norm = float(R.size)
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        phi = phase_convolved(R.shape, phase_window, rng)
        s = np.sum(weight * np.exp(1j * phi))
        out[i] = 4.0 * abs(s) / norm
    return out


def simulate_constant_phase(R: np.ndarray, B: np.ndarray) -> float:
    weight = np.sqrt(np.clip(R, 0.0, None) * np.clip(B, 0.0, None))
    return 4.0 * float(np.sum(weight)) / float(R.size)


def describe(x: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
        "p05": float(np.percentile(x, 5)),
        "p50": float(np.percentile(x, 50)),
        "p95": float(np.percentile(x, 95)),
        "p99": float(np.percentile(x, 99)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis-dir", type=Path, default=DEFAULT_ANALYSIS_DIR)
    parser.add_argument("--replicates", type=int, default=N_REPLICATES)
    args = parser.parse_args()
    analysis_dir = args.analysis_dir
    out_dir = analysis_dir / "pixel_resolved_interference_simulation"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = json.loads((analysis_dir / "z_scan_summary.json").read_text())
    rod_dir = Path(summary["rod_recording"])
    bg_dir = Path(summary["background_recording"])
    rod_total = central_11_from_stack_mean(rod_dir / "capture_maxfps_15x15.npy")
    bg = central_11_from_stack_mean(bg_dir / "capture_maxfps_15x15.npy")
    rod_profile = np.clip(rod_total - bg, 0.0, None)

    rod_signal_mean = float(np.mean(rod_profile))
    bg_mean = float(np.mean(bg))
    measured_summary = json.loads((analysis_dir / "slope_corrected_peak_to_min_amplitude_summary.json").read_text())
    measured_p2min = float(measured_summary["slope_corrected_peak_to_min_intensity"])
    measured_pct = float(measured_summary["slope_corrected_peak_to_min_percent_of_rod_signal"])

    rng = np.random.default_rng(2026072001)
    cases = [(f"convolved_phase_{window}x{window}", window) for window in range(1, 6)]

    rows = []
    all_percent = {}
    for label, block in cases:
        vals = simulate(rod_profile, bg, block, int(args.replicates), rng)
        pct = 100.0 * vals / rod_signal_mean
        all_percent[label] = pct
        d_i = describe(vals)
        d_p = describe(pct)
        rows.append(
            {
                "case": label,
                "phase_model": f"random unit phasor field convolved with {block}x{block} square window",
                "phase_window_px": block,
                "n_replicates": int(args.replicates),
                "mean_intensity_units": d_i["mean"],
                "std_intensity_units": d_i["std"],
                "p05_intensity_units": d_i["p05"],
                "p50_intensity_units": d_i["p50"],
                "p95_intensity_units": d_i["p95"],
                "p99_intensity_units": d_i["p99"],
                "mean_percent_of_rod": d_p["mean"],
                "std_percent_of_rod": d_p["std"],
                "p05_percent_of_rod": d_p["p05"],
                "p50_percent_of_rod": d_p["p50"],
                "p95_percent_of_rod": d_p["p95"],
                "p99_percent_of_rod": d_p["p99"],
            }
        )

    constant_val = simulate_constant_phase(rod_profile, bg)
    constant_pct = 100.0 * constant_val / rod_signal_mean
    d_i = describe(np.asarray([constant_val], dtype=np.float64))
    d_p = describe(np.asarray([constant_pct], dtype=np.float64))
    rows.append(
        {
            "case": "constant_phase_11px_window",
            "phase_model": "single phase over whole 11x11 window; marker only on density plot",
            "phase_window_px": CENTRAL_WINDOW,
            "n_replicates": 1,
            "mean_intensity_units": d_i["mean"],
            "std_intensity_units": d_i["std"],
            "p05_intensity_units": d_i["p05"],
            "p50_intensity_units": d_i["p50"],
            "p95_intensity_units": d_i["p95"],
            "p99_intensity_units": d_i["p99"],
            "mean_percent_of_rod": d_p["mean"],
            "std_percent_of_rod": d_p["std"],
            "p05_percent_of_rod": d_p["p05"],
            "p50_percent_of_rod": d_p["p50"],
            "p95_percent_of_rod": d_p["p95"],
            "p99_percent_of_rod": d_p["p99"],
        }
    )

    with (out_dir / "pixel_resolved_interference_simulation_summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    np.save(out_dir / "rod_profile_r_squared_11x11.npy", rod_profile.astype(np.float32), allow_pickle=False)
    np.save(out_dir / "background_profile_b_squared_11x11.npy", bg.astype(np.float32), allow_pickle=False)
    np.save(out_dir / "interference_peak_to_min_percent_replicates.npy", all_percent, allow_pickle=True)

    fig, axes = plt.subplots(1, 2, figsize=(8, 3.8))
    im0 = axes[0].imshow(rod_profile, cmap="inferno")
    axes[0].set_title("Rod r^2 profile")
    plt.colorbar(im0, ax=axes[0], fraction=0.046)
    im1 = axes[1].imshow(bg, cmap="viridis")
    axes[1].set_title("Background b^2 profile")
    plt.colorbar(im1, ax=axes[1], fraction=0.046)
    fig.tight_layout()
    fig.savefig(out_dir / "rod_and_background_profiles_11x11.png", dpi=240)
    plt.close(fig)

    labels = [r["case"] for r in rows]
    means = np.asarray([r["mean_percent_of_rod"] for r in rows], dtype=float)
    stds = np.asarray([r["std_percent_of_rod"] for r in rows], dtype=float)
    p05 = np.asarray([r["p05_percent_of_rod"] for r in rows], dtype=float)
    p95 = np.asarray([r["p95_percent_of_rod"] for r in rows], dtype=float)
    y = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(9, 5.2))
    ax.barh(y, means, xerr=stds, color="tab:blue", alpha=0.75, label="simulation mean +/- sigma")
    ax.errorbar(means, y, xerr=[means - p05, p95 - means], fmt="none", ecolor="black", alpha=0.55, capsize=2, label="5-95%")
    ax.axvline(measured_pct, color="tab:red", lw=2.0, label="measured slope-corrected peak-to-min")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Slope-corrected peak-to-minimum variation (% of rod signal)")
    ax.set_title("Pixel-resolved interference simulation using measured rod/background profiles")
    ax.grid(axis="x", alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "pixel_resolved_expected_vs_measured_percent.png", dpi=240)
    plt.close(fig)

    plt.figure(figsize=(9, 5.2))
    marker_cases = {"constant_phase_11px_window"}
    density_values = [pct for label, pct in all_percent.items() if label not in marker_cases]
    marker_values = {"constant phase": constant_pct}
    density_concat = np.concatenate(density_values)
    x_max = max(float(np.percentile(density_concat, 99.8)), measured_pct) * 1.12
    bins = np.linspace(0, x_max, 220)
    for label, pct in all_percent.items():
        if label in marker_cases:
            continue
        plt.hist(pct, bins=bins, density=True, histtype="step", linewidth=1.5, label=label)
    plt.xlabel("Slope-corrected peak-to-minimum variation (% of rod signal)")
    plt.ylabel("Probability density")
    plt.title("Distribution over random background phase realisations")
    ymin, ymax = plt.ylim()
    plt.scatter(
        [measured_pct],
        [0.0],
        marker="^",
        s=90,
        color="tab:red",
        edgecolor="black",
        linewidth=0.6,
        zorder=5,
        clip_on=False,
        label="measured",
    )
    for label, value in marker_values.items():
        if value <= x_max:
            marker_x = value
            marker_label = label
        else:
            marker_x = x_max
            marker_label = f"{label}: {value:.1f}% off scale"
        plt.scatter(
            [marker_x],
            [0.0],
            marker="s",
            s=72,
            color="tab:purple",
            edgecolor="black",
            linewidth=0.6,
            zorder=5,
            clip_on=False,
            label=marker_label,
        )
    plt.ylim(ymin, ymax)
    plt.xlim(0, x_max)
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(out_dir / "pixel_resolved_phase_replicate_distributions.png", dpi=240)
    plt.close()

    readme = [
        "Pixel-resolved z-interference simulation",
        "",
        f"Analysis folder: {analysis_dir}",
        f"Rod recording: {rod_dir.name}",
        f"Background recording: {bg_dir.name}",
        "",
        "Intensity maps used:",
        "- background b^2 is the time-average central 11x11 raw-pixel profile from the background recording.",
        "- rod r^2 is the time-average central 11x11 raw-pixel rod recording minus the background profile, clipped at zero.",
        "- Interference is computed pixel-by-pixel as 2 sqrt(r_i^2 b_i^2) cos(phi_i + global_z_phase).",
        "- For each random phase map, the z oscillation peak-to-minimum is 4*abs(sum_i sqrt(R_i B_i) exp(i phi_i))/N_pixels.",
        "- Results are reported as percentage of mean rod signal over the 11x11 window.",
        "",
        f"Mean rod r^2 = {rod_signal_mean:.6g}",
        f"Mean background b^2 = {bg_mean:.6g}",
        f"Measured slope-corrected peak-to-minimum = {measured_p2min:.6g} intensity units = {measured_pct:.3g}%",
        f"Monte Carlo replicates per phase condition = {int(args.replicates)}",
    ]
    (out_dir / "README_pixel_resolved_interference_simulation.txt").write_text("\n".join(readme) + "\n")
    print(f"Wrote outputs to {out_dir}")
    print(json.dumps({"rod_mean": rod_signal_mean, "background_mean": bg_mean, "measured_percent": measured_pct, "rows": rows}, indent=2))


if __name__ == "__main__":
    main()
