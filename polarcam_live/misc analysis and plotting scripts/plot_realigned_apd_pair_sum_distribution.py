from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ANALYSIS_DIR = Path(r"E:\40nm apd precision realigned 0708\analysis_unscaled")
CSV_PATH = ANALYSIS_DIR / "apd_unscaled_all_recordings.csv"


def natural_key(text: str) -> list[int | str]:
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", text)]


def main() -> None:
    rows = []
    with CSV_PATH.open("r", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            rows.append(
                {
                    "file": Path(row["file"]).name,
                    "ratio": float(row["pair_sum_ratio_45_135_over_0_90"]),
                    "r_mean": float(row["r_mean"]),
                    "included": row["included"].lower() == "true",
                }
            )
    rows = sorted(rows, key=lambda row: natural_key(row["file"]))
    ratios = np.asarray([row["ratio"] for row in rows], dtype=np.float64)
    r_mean = np.asarray([row["r_mean"] for row in rows], dtype=np.float64)
    order = np.arange(1, len(rows) + 1)

    near_masks = {
        "within_5pct_of_1": np.abs(ratios - 1.0) <= 0.05,
        "within_10pct_of_1": np.abs(ratios - 1.0) <= 0.10,
        "within_15pct_of_1": np.abs(ratios - 1.0) <= 0.15,
    }
    summary = {
        "n_recordings": len(rows),
        "ratio_mean": float(np.mean(ratios)),
        "ratio_std": float(np.std(ratios, ddof=1)),
        "ratio_min": float(np.min(ratios)),
        "ratio_max": float(np.max(ratios)),
        "counts": {key: int(np.count_nonzero(mask)) for key, mask in near_masks.items()},
        "recordings_within_10pct": [rows[i]["file"] for i in np.where(near_masks["within_10pct_of_1"])[0]],
        "recordings_outside_15pct": [rows[i]["file"] for i in np.where(~near_masks["within_15pct_of_1"])[0]],
    }

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.6))
    ax = axes[0, 0]
    bins = np.linspace(max(0.75, np.min(ratios) - 0.03), min(1.6, np.max(ratios) + 0.03), 18)
    ax.hist(ratios, bins=bins, color="#2a7f62", alpha=0.78, edgecolor="white")
    ax.axvline(1.0, color="#111111", lw=1.2, label="balanced = 1")
    ax.axvspan(0.95, 1.05, color="#999999", alpha=0.16, label="within 5%")
    ax.axvspan(0.90, 1.10, color="#999999", alpha=0.08, label="within 10%")
    ax.set_xlabel("(I135 + I45) / (I90 + I0)")
    ax.set_ylabel("recordings")
    ax.set_title("Pair-sum ratio distribution")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(True, alpha=0.2)

    ax = axes[0, 1]
    colors = np.where(np.abs(ratios - 1.0) <= 0.10, "#2a7f62", "#b23a48")
    ax.scatter(order, ratios, c=colors, s=46, edgecolors="white", linewidths=0.45)
    ax.axhline(1.0, color="#111111", lw=1.0)
    ax.axhspan(0.90, 1.10, color="#999999", alpha=0.10)
    ax.set_xlabel("recording order")
    ax.set_ylabel("(I135 + I45) / (I90 + I0)")
    ax.set_title("Pair-sum ratio vs recording order")
    ax.grid(True, alpha=0.2)
    for idx, row in enumerate(rows, start=1):
        if abs(row["ratio"] - 1.0) > 0.15:
            ax.annotate(row["file"].replace(".tdms", ""), (idx, row["ratio"]), xytext=(3, 4), textcoords="offset points", fontsize=7)

    ax = axes[1, 0]
    ax.scatter(r_mean, ratios, c=colors, s=46, edgecolors="white", linewidths=0.45)
    ax.axhline(1.0, color="#111111", lw=1.0)
    ax.axhspan(0.90, 1.10, color="#999999", alpha=0.10)
    ax.set_xlabel("mean r")
    ax.set_ylabel("(I135 + I45) / (I90 + I0)")
    ax.set_title("Pair-sum ratio vs mean r")
    ax.grid(True, alpha=0.2)

    ax = axes[1, 1]
    sorted_idx = np.argsort(ratios)
    ax.scatter(np.arange(1, len(rows) + 1), ratios[sorted_idx], c=colors[sorted_idx], s=46, edgecolors="white", linewidths=0.45)
    ax.axhline(1.0, color="#111111", lw=1.0)
    ax.axhspan(0.90, 1.10, color="#999999", alpha=0.10)
    ax.set_xlabel("ranked recording")
    ax.set_ylabel("(I135 + I45) / (I90 + I0)")
    ax.set_title("Ranked pair-sum ratios")
    ax.grid(True, alpha=0.2)

    fig.tight_layout()
    fig.savefig(ANALYSIS_DIR / "apd_unscaled_pair_sum_ratio_distribution.png", dpi=220)
    plt.close(fig)

    (ANALYSIS_DIR / "pair_sum_ratio_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Plot: {ANALYSIS_DIR / 'apd_unscaled_pair_sum_ratio_distribution.png'}")


if __name__ == "__main__":
    main()
