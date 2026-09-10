from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


OUT_DIR = Path(__file__).resolve().parent
ABC_CSV = OUT_DIR / "theta_r_abc_values.csv"
CURVE_CSV = OUT_DIR / "theta_r_curve_values.csv"
PLOT_PNG = OUT_DIR / "theta_r_curves_water1p33_glycerol1p47_naout1p3_nain0p39.png"

MM_TO_IN = 1.0 / 25.4
FIG_W_MM = 90.0
FIG_H_MM = 58.0
FONT_PT = 7


def style_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
            "font.size": FONT_PT,
            "axes.titlesize": FONT_PT,
            "axes.labelsize": FONT_PT,
            "xtick.labelsize": FONT_PT,
            "ytick.labelsize": FONT_PT,
            "legend.fontsize": FONT_PT,
        }
    )


def read_abc() -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    with ABC_CSV.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            out[row["model"]] = {
                "n_medium": float(row["n_medium"]),
                "A": float(row["A"]),
                "B": float(row["B"]),
                "C": float(row["C"]),
                "r_max": float(row["r_max"]),
            }
    return out


def read_curves() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    grouped_r: dict[str, list[float]] = defaultdict(list)
    grouped_theta: dict[str, list[float]] = defaultdict(list)
    with CURVE_CSV.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            grouped_r[row["model"]].append(float(row["r"]))
            grouped_theta[row["model"]].append(float(row["theta_deg"]))
    return {
        model: (
            np.asarray(grouped_r[model], dtype=np.float64),
            np.asarray(grouped_theta[model], dtype=np.float64),
        )
        for model in grouped_r
    }


def main() -> None:
    style_matplotlib()
    abc = read_abc()
    curves = read_curves()

    fig, ax = plt.subplots(figsize=(FIG_W_MM * MM_TO_IN, FIG_H_MM * MM_TO_IN))
    colors = {"water": "#1f77b4", "glycerol": "#d95f02"}

    for model in ("water", "glycerol"):
        r, theta_deg = curves[model]
        params = abc[model]
        ax.plot(r, theta_deg, color=colors[model], lw=1.8, label=f"{model}, n = {params['n_medium']:.2f}")
        ax.axvline(params["r_max"], color=colors[model], ls="--", lw=1.0, alpha=0.9)
        ax.text(
            params["r_max"] + 0.004,
            2.0,
            rf"$r_{{\max}} = {params['r_max']:.3f}$",
            color=colors[model],
            rotation=90,
            ha="left",
            va="bottom",
        )

    ax.set_xlabel(r"$r = \sqrt{X^2 + Y^2}$")
    ax.set_ylabel(r"$\theta$ (deg)")
    ax.set_title(r"Rod $\theta$ from anisotropy radius")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 92.0)
    ax.grid(True, alpha=0.28)
    ax.legend(loc="upper left", frameon=False, handlelength=2.2)
    fig.tight_layout(pad=0.7)
    fig.savefig(PLOT_PNG, dpi=300)
    plt.close(fig)

    print(f"Plot: {PLOT_PNG}")


if __name__ == "__main__":
    main()
