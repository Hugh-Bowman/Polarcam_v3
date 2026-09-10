from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SOURCE_DIR = Path(
    r"C:\Polarcam Software\Polarcam_v3\polarcam_live\datasets\40nm stationary buffer sound on\precision_error_model_water_fourkas_p16_p84_theta_percentile"
)
TARGET_DIR = Path(r"C:\Polarcam Software\Polarcam_v3\polarcam_live\datasets\40nm stuck rod data")

SUBSETS = [
    ("0p6ms_theta_fit_rods.csv", "06ms theta rods", "0.6 ms", "theta"),
    ("1p2ms_theta_fit_rods.csv", "12ms theta rods", "1.2 ms", "theta"),
    ("0p6ms_phi_fit_rods.csv", "06ms phi rods", "0.6 ms", "phi"),
    ("1p2ms_phi_fit_rods.csv", "12ms phi rods", "1.2 ms", "phi"),
]

PLOT_STYLE = {
    "0.6 ms": {"color": "#1f77b4", "marker": "o", "label": "0.6 ms exposure"},
    "1.2 ms": {"color": "#ff7f0e", "marker": "s", "label": "1.2 ms exposure"},
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def reset_subset_folder(path: Path) -> None:
    target = path.resolve()
    root = TARGET_DIR.resolve()
    if root not in target.parents:
        raise RuntimeError(f"Refusing to clear path outside target root: {target}")
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def make_plot_ready_rows(rows: list[dict[str, str]], exposure: str, kind: str, subset_dir: Path) -> list[dict[str, object]]:
    out = []
    for row in rows:
        source_path = Path(row["path"])
        local_path = subset_dir / source_path.name
        out.append(
            {
                "exposure_ms": 0.6 if exposure == "0.6 ms" else 1.2,
                "exposure_label": exposure,
                "plot_kind": kind,
                "rod": row["rod"],
                "source_path": str(source_path),
                "local_path": str(local_path),
                "theta_deg_current_curve": row["theta_deg"],
                "r_mean": row["r_mean"],
                "r_p16": row["r_p16"],
                "r_p84": row["r_p84"],
                "sigma_theta_deg_current_curve": row["sigma_theta_deg"],
                "sigma_phi_deg": row["sigma_phi_deg"],
                "r_clipped_to_90_current_curve": row["r_clipped_to_90"],
                "n_frames": row["n_frames"],
                "background_subtracted": row["background_subtracted"],
                "background_profile_path": row["background_profile_path"],
                "intensity_mean_raw_square": row["intensity_mean_raw_square"],
            }
        )
    return out


def copy_rod_dirs(rows: list[dict[str, str]], subset_dir: Path) -> None:
    for row in rows:
        source = Path(row["path"])
        destination = subset_dir / source.name
        if destination.exists():
            shutil.rmtree(destination)
        shutil.copytree(source, destination)


def add_low_theta_region(ax: plt.Axes, label: bool = True) -> None:
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


def plot_combined(all_points: list[dict[str, object]], kind: str) -> None:
    if kind == "theta":
        y_key = "sigma_theta_deg_current_curve"
        ylabel = r"$\sigma_\theta$ (deg)"
        title = r"40 nm stuck rods: $\sigma_\theta$ vs $\theta$"
        filename = "combined_sigma_theta_vs_theta.png"
    elif kind == "phi":
        y_key = "sigma_phi_deg"
        ylabel = r"$\sigma_\phi$ (deg)"
        title = r"40 nm stuck rods: $\sigma_\phi$ vs $\theta$"
        filename = "combined_sigma_phi_vs_theta.png"
    else:
        raise ValueError(f"Unknown kind: {kind}")

    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    add_low_theta_region(ax)
    y_values = []
    for exposure in ["0.6 ms", "1.2 ms"]:
        rows = [r for r in all_points if r["plot_kind"] == kind and r["exposure_label"] == exposure]
        style = PLOT_STYLE[exposure]
        theta = np.asarray([float(r["theta_deg_current_curve"]) for r in rows], dtype=np.float64)
        y = np.asarray([float(r[y_key]) for r in rows], dtype=np.float64)
        y_values.append(y)
        ax.scatter(
            theta,
            y,
            s=44,
            marker=style["marker"],
            color=style["color"],
            edgecolors="white",
            linewidths=0.45,
            alpha=0.82,
            label=style["label"],
            zorder=2,
        )
    y_all = np.concatenate(y_values) if y_values else np.asarray([])
    y_all = y_all[np.isfinite(y_all)]
    y_cap = float(np.max(y_all) * 1.12) if y_all.size else 1.0
    ax.set_xlim(0.0, 90.0)
    ax.set_ylim(0.0, max(y_cap, 1.0))
    ax.set_xlabel(r"$\theta$ (deg)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.24)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(TARGET_DIR / filename, dpi=220)
    plt.close(fig)


def main() -> None:
    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    all_plot_rows: list[dict[str, object]] = []
    summary = {
        "source_analysis_dir": str(SOURCE_DIR),
        "target_dir": str(TARGET_DIR),
        "theta_r_curve_used_for_current_theta_columns": {
            "label": "Fourkas finite-NA water/buffer",
            "J1": 0.65235,
            "J2": 0.03744,
            "J3": 0.10765,
            "r_max": 0.8914452224590091,
        },
        "note": "CSV files include r_mean, r_p16, and r_p84 so theta and sigma_theta can be recalculated using a different theta(r) curve later.",
        "subsets": [],
    }

    for csv_name, subset_name, exposure, kind in SUBSETS:
        source_csv = SOURCE_DIR / csv_name
        subset_dir = TARGET_DIR / subset_name
        reset_subset_folder(subset_dir)
        source_rows = read_csv(source_csv)
        copy_rod_dirs(source_rows, subset_dir)
        plot_rows = make_plot_ready_rows(source_rows, exposure, kind, subset_dir)
        easy_csv = TARGET_DIR / f"{subset_name.replace(' ', '_')}_plot_points.csv"
        write_csv(easy_csv, plot_rows)
        all_plot_rows.extend(plot_rows)
        summary["subsets"].append(
            {
                "subset": subset_name,
                "exposure": exposure,
                "plot_kind": kind,
                "n_rods": len(source_rows),
                "source_csv": str(source_csv),
                "plot_points_csv": str(easy_csv),
                "rod_data_folder": str(subset_dir),
            }
        )

    write_csv(TARGET_DIR / "all_plot_points.csv", all_plot_rows)
    plot_combined(all_plot_rows, "theta")
    plot_combined(all_plot_rows, "phi")
    (TARGET_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (TARGET_DIR / "README.txt").write_text(
        "\n".join(
            [
                "40 nm stuck rod data package",
                "",
                "Folders contain the rod recording directories used in the current plots.",
                "CSV files contain one row per plotted point.",
                "To remap theta with a new theta(r) curve, use r_mean for the point location and r_p16/r_p84 for sigma_theta.",
                "Current sigma_theta column was computed as abs(theta(r_p84)-theta(r_p16))/2 using the listed Fourkas water/buffer curve.",
                "Phi uncertainty is sigma_phi_deg from the XY series and does not require theta(r) remapping.",
            ]
        ),
        encoding="utf-8",
    )
    print(f"Output: {TARGET_DIR}")


if __name__ == "__main__":
    main()
