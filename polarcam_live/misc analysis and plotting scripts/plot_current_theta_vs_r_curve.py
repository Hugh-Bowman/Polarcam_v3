from __future__ import annotations

import csv

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import analyze_40nm_bg_subtracted_sound_precision_new as base


def main() -> None:
    out_dir = base.OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    curve = base.build_curve()

    r = np.asarray(curve["r"], dtype=np.float64)
    theta = np.asarray(curve["theta_deg"], dtype=np.float64)
    order = np.argsort(r)
    r_sorted = r[order]
    theta_sorted = theta[order]

    csv_path = out_dir / "theta_vs_r_curve_used_water_fourkas.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["r", "theta_deg"])
        writer.writerows(zip(r_sorted, theta_sorted))

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.plot(r_sorted, theta_sorted, color="#1f77b4", lw=2.0)
    ax.axvline(float(base.FOURKAS_WATER["r_max"]), color="#666666", ls="--", lw=1.0, alpha=0.7)
    ax.text(
        float(base.FOURKAS_WATER["r_max"]),
        4,
        f"r_max={float(base.FOURKAS_WATER['r_max']):.4f}",
        ha="right",
        va="bottom",
        fontsize=9,
        color="#444444",
    )
    ax.set_xlabel("r")
    ax.set_ylabel("theta (deg)")
    ax.set_title("Theta(r) conversion curve: Fourkas water/buffer")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 92.0)
    ax.grid(True, alpha=0.24)
    fig.tight_layout()
    png_path = out_dir / "theta_vs_r_curve_used_water_fourkas.png"
    fig.savefig(png_path, dpi=220)
    plt.close(fig)

    print(f"Plot: {png_path}")
    print(f"CSV: {csv_path}")
    print(f"J1={base.FOURKAS_WATER['J1']}")
    print(f"J2={base.FOURKAS_WATER['J2']}")
    print(f"J3={base.FOURKAS_WATER['J3']}")
    print(f"r_max={base.FOURKAS_WATER['r_max']}")


if __name__ == "__main__":
    main()
