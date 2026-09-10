from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


CURVES = {
    "25x65nm": {
        "curve_csv": Path("theta_r_curves_for_analysis") / "theta_r_curve_25x65nm_recording_bootstrap.csv",
        "good_dir": Path("stationary rods 25nm 02072026") / "filtered_intensity_p98_50_to_400_sigma_theta_pruned" / "good",
        "fill": "#2ca02c",
        "line": "#1b7f3a",
    },
    "40x65nm": {
        "curve_csv": Path("theta_r_curves_for_analysis") / "theta_r_curve_40x65nm_recording_bootstrap.csv",
        "good_dir": Path("stationary rod data 01072026") / "good",
        "fill": "#d62728",
        "line": "#a51c30",
    },
}

OUTPUT_DIR = Path("outputs") / "40nm_vs_25nm_std_comparison_refit_new_curves"


def _load_curve(path: Path) -> dict[str, np.ndarray]:
    rows = list(csv.DictReader(path.open("r", encoding="utf-8", newline="")))
    return {
        "r": np.asarray([float(row["r"]) for row in rows], dtype=np.float64),
        "center": np.asarray([float(row["theta_deg_center"]) for row in rows], dtype=np.float64),
        "lo": np.asarray([float(row["theta_deg_lo_1sigma"]) for row in rows], dtype=np.float64),
        "hi": np.asarray([float(row["theta_deg_hi_1sigma"]) for row in rows], dtype=np.float64),
        "std": np.asarray([float(row["theta_deg_std"]) for row in rows], dtype=np.float64),
    }


def _load_stationary_mean_r(good_dir: Path) -> np.ndarray:
    vals = []
    for rod_dir in sorted([p for p in good_dir.iterdir() if p.is_dir()]):
        meta_path = rod_dir / "capture_maxfps_15x15_meta.json"
        if not meta_path.exists():
            continue
        payload = json.loads(meta_path.read_text(encoding="utf-8"))
        xy = np.asarray(payload.get("xy_series", []), dtype=np.float64)
        if xy.ndim != 2 or xy.shape[0] == 0 or xy.shape[1] < 2:
            continue
        xy = xy[:, :2]
        valid = np.isfinite(xy[:, 0]) & np.isfinite(xy[:, 1])
        xy = xy[valid]
        if xy.size == 0:
            continue
        r = np.sqrt((xy[:, 0] * xy[:, 0]) + (xy[:, 1] * xy[:, 1]))
        vals.append(float(np.mean(r)))
    arr = np.sort(np.asarray(vals, dtype=np.float64))
    if arr.size == 0:
        raise ValueError(f"No stationary rods found in {good_dir}")
    return arr


def _second_highest(arr: np.ndarray) -> tuple[float, float]:
    raw_max = float(arr[-1])
    if arr.size == 1:
        return raw_max, raw_max
    return float(arr[-2]), raw_max


def _stretch_curve(curve: dict[str, np.ndarray], target_r_max: float) -> tuple[dict[str, np.ndarray], float]:
    src_r_max = float(np.max(curve["r"]))
    scale = float(target_r_max / max(src_r_max, 1e-12))
    return {
        "r": curve["r"] * scale,
        "center": curve["center"].copy(),
        "lo": curve["lo"].copy(),
        "hi": curve["hi"].copy(),
        "std": curve["std"].copy(),
    }, scale


def main() -> None:
    out_dir = Path.cwd() / OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {}

    fig, ax = plt.subplots(figsize=(7.4, 4.9))
    for label, cfg in CURVES.items():
        curve = _load_curve(Path.cwd() / cfg["curve_csv"])
        stationary_r = _load_stationary_mean_r(Path.cwd() / cfg["good_dir"])
        target_r_max, raw_r_max = _second_highest(stationary_r)
        stretched, scale = _stretch_curve(curve, target_r_max)

        ax.fill_between(
            stretched["r"],
            stretched["lo"],
            stretched["hi"],
            color=cfg["fill"],
            alpha=0.22,
            label=f"{label} water 1 sigma",
        )
        ax.plot(stretched["r"], stretched["center"], color=cfg["line"], lw=2.2, label=f"{label} water theta(r)")

        payload[label] = {
            "curve_csv": str((Path.cwd() / cfg["curve_csv"]).resolve()),
            "good_dir": str((Path.cwd() / cfg["good_dir"]).resolve()),
            "source_curve_r_max": float(np.max(curve["r"])),
            "stationary_raw_r_max": raw_r_max,
            "stationary_second_highest_r_max": target_r_max,
            "stretch_scale": scale,
        }

    ax.set_xlabel("r")
    ax.set_ylabel("theta (deg)")
    ax.set_title("Theta(r) with recording-bootstrap error, stretched for water stationary rods")
    ax.grid(True, alpha=0.22)
    ax.legend(loc="best", frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / "theta_vs_r_25nm_40nm_water_stretched_second_highest_rmax.png", dpi=220)
    plt.close(fig)

    (out_dir / "theta_vs_r_25nm_40nm_water_stretched_second_highest_rmax_summary.json").write_text(
        json.dumps(
            {
                "output_dir": str(out_dir.resolve()),
                "method": "Take the saved glycerol-derived theta(r) curves with recording-bootstrap uncertainty, and stretch each in r so its maximum r matches the second-highest observed rod mean r in the corresponding stationary-water dataset.",
                "datasets": payload,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
