from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


OUTPUT_DIR = Path(__file__).resolve().parent / "theta_r_curve_parameters"
MM_TO_IN = 1.0 / 25.4
FIG_W_MM = 90.0
FIG_H_MM = 58.0

MODELS: dict[str, dict[str, float | str]] = {
    "water": {
        "label": "water",
        "A": 0.894504,
        "B": 0.944236,
        "C": 0.128466,
    },
    "glycerol": {
        "label": "glycerol",
        "A": 0.464980,
        "B": 0.736949,
        "C": 0.297224,
    },
}


def theta_from_r(r: np.ndarray, A: float, B: float, C: float) -> np.ndarray:
    # Implement the user-specified form explicitly as:
    # theta(r) = asin(sqrt((A*r) / (B - C*r)))
    denom = B - (C * r)
    val = np.zeros_like(r, dtype=np.float64)
    valid = denom > 0.0
    val[valid] = (A * r[valid]) / denom[valid]
    val = np.clip(val, 0.0, 1.0)
    return np.arcsin(np.sqrt(val))


def r_max_from_params(A: float, B: float, C: float) -> float:
    # Maximum radius occurs when the asin argument reaches 1:
    # (A*r) / (B - C*r) = 1  ->  r_max = B / (A + C)
    return B / (A + C)


def style_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.size": 7,
            "axes.titlesize": 7,
            "axes.labelsize": 7,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
        }
    )


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    style_matplotlib()

    r = np.linspace(0.0, 1.0, 1001, dtype=np.float64)
    rows: list[list[float | str]] = []
    payload: dict[str, object] = {
        "description": "Theta(r) curves using theta = asin(sqrt((A*r)/(B-C*r))).",
        "formula": "theta(r) = asin(sqrt((A*r)/(B-C*r)))",
        "models": {},
    }

    fig, ax = plt.subplots(figsize=(FIG_W_MM * MM_TO_IN, FIG_H_MM * MM_TO_IN))
    colors = {
        "water": "#1f77b4",
        "glycerol": "#d95f02",
    }
    legend_labels = {
        "water": "water, n = 1.3",
        "glycerol": "glycerol, n = 1.5",
    }

    for key, cfg in MODELS.items():
        label = str(cfg["label"])
        A = float(cfg["A"])
        B = float(cfg["B"])
        C = float(cfg["C"])
        r_max = r_max_from_params(A, B, C)
        theta_deg = np.degrees(theta_from_r(r, A, B, C))

        rows.extend(
            [label, float(rr), float(tt), A, B, C, r_max]
            for rr, tt in zip(r, theta_deg)
        )
        payload["models"][label] = {
            "A": A,
            "B": B,
            "C": C,
            "r_max": r_max,
        }

        color = colors.get(key, None)
        ax.plot(r, theta_deg, lw=1.8, color=color, label=legend_labels.get(key, label))
        ax.axvline(r_max, color=color, ls="--", lw=1.0, alpha=0.9)
        ax.text(
            r_max + 0.004,
            2.0,
            rf"$r_{{\max}} = {r_max:.3f}$",
            color=color,
            rotation=90,
            ha="left",
            va="bottom",
        )

    csv_path = OUTPUT_DIR / "theta_r_curves_water_glycerol_new_abc.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["model", "r", "theta_deg", "A", "B", "C", "r_max"])
        writer.writerows(rows)

    json_path = OUTPUT_DIR / "theta_r_curve_parameters_water_glycerol_new_abc.json"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    ax.set_xlabel(r"$r = \sqrt{X^2 + Y^2}$")
    ax.set_ylabel("θ (deg)")
    ax.set_title("Rod θ from Anisoptopry radius")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 92.0)
    ax.grid(True, alpha=0.28)
    ax.legend(loc="upper left", frameon=False, handlelength=2.2)
    fig.tight_layout(pad=0.7)

    png_path = OUTPUT_DIR / "theta_r_curves_water_glycerol_new_abc.png"
    fig.savefig(png_path, dpi=300)
    fig.savefig(OUTPUT_DIR / "theta_r_curves_water_vs_glycerol99p5.png", dpi=300)
    plt.close(fig)

    print(f"Plot: {png_path}")
    print(f"CSV: {csv_path}")
    print(f"JSON: {json_path}")
    for key, cfg in MODELS.items():
        A = float(cfg["A"])
        B = float(cfg["B"])
        C = float(cfg["C"])
        r_max = r_max_from_params(A, B, C)
        print(f"{key}: A={A:.6f} B={B:.6f} C={C:.6f} r_max={r_max:.6f}")


if __name__ == "__main__":
    main()
