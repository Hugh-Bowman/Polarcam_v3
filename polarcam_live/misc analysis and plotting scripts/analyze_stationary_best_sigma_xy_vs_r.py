from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    import matplotlib.pyplot as plt
except Exception as exc:  # pragma: no cover
    raise RuntimeError(f"matplotlib is required to run this script: {exc}")


DATASET_DIR = Path("stationary rod data 01072026")
PENDING_DIRNAME = "pending"
GOOD_DIRNAME = "good"
OUTPUT_SUBDIR = Path("plots") / "best_sigma_xy_vs_r"
N_R_BINS = 10
TOP_PER_BIN = 3
EXTRA_PER_BIN = 4


@dataclass
class RodStats:
    rod: str
    source: str
    n_frames: int
    r_mean: float
    r_std: float
    sigma_x: float
    sigma_y: float
    sigma_xy: float
    motion_max_axis_range: float
    path: Path


def _load_xy_series(meta_path: Path) -> np.ndarray:
    payload = json.loads(meta_path.read_text(encoding="utf-8"))
    xy_series = payload.get("xy_series")
    if not isinstance(xy_series, list):
        raise ValueError(f"No xy_series in {meta_path}")
    arr = np.asarray(xy_series, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] < 2:
        raise ValueError(f"Bad xy_series shape in {meta_path}: {arr.shape}")
    arr = arr[:, :2]
    valid = np.isfinite(arr[:, 0]) & np.isfinite(arr[:, 1])
    return arr[valid]


def _rod_stats_from_dir(rod_dir: Path, source: str) -> RodStats | None:
    meta_path = rod_dir / "capture_maxfps_15x15_meta.json"
    if not meta_path.exists():
        return None
    arr = _load_xy_series(meta_path)
    if arr.size == 0:
        return None
    x = arr[:, 0]
    y = arr[:, 1]
    r = np.sqrt((x * x) + (y * y))
    sigma_x = float(np.std(x))
    sigma_y = float(np.std(y))
    sigma_xy = float(np.sqrt((sigma_x * sigma_x) + (sigma_y * sigma_y)))
    motion_max_axis_range = float(max(np.max(x) - np.min(x), np.max(y) - np.min(y)))
    return RodStats(
        rod=rod_dir.name,
        source=source,
        n_frames=int(arr.shape[0]),
        r_mean=float(np.mean(r)),
        r_std=float(np.std(r)),
        sigma_x=sigma_x,
        sigma_y=sigma_y,
        sigma_xy=sigma_xy,
        motion_max_axis_range=motion_max_axis_range,
        path=rod_dir,
    )


def _load_all_rods(dataset_root: Path) -> list[RodStats]:
    rods: list[RodStats] = []
    for source_name in (PENDING_DIRNAME, GOOD_DIRNAME):
        source_dir = dataset_root / source_name
        if not source_dir.is_dir():
            continue
        for rod_dir in sorted([p for p in source_dir.iterdir() if p.is_dir()]):
            item = _rod_stats_from_dir(rod_dir, source=source_name)
            if item is not None:
                rods.append(item)
    return rods


def _select_best_per_r_bin(
    rods: list[RodStats],
    n_bins: int,
    top_per_bin: int,
    extra_per_bin: int,
) -> tuple[list[tuple[RodStats, str]], list[dict]]:
    if not rods:
        return [], []
    r_vals = np.asarray([r.r_mean for r in rods], dtype=np.float64)
    r_min = float(np.min(r_vals))
    r_max = float(np.max(r_vals))
    if not np.isfinite(r_min) or not np.isfinite(r_max):
        return [], []
    if r_max <= r_min:
        bins = np.array([r_min, r_max + 1e-6], dtype=np.float64)
    else:
        bins = np.linspace(r_min, r_max, int(n_bins) + 1, dtype=np.float64)
    selected: list[tuple[RodStats, str]] = []
    summary: list[dict] = []
    for i in range(len(bins) - 1):
        lo = float(bins[i])
        hi = float(bins[i + 1])
        if i == len(bins) - 2:
            in_bin = [r for r in rods if lo <= r.r_mean <= hi]
        else:
            in_bin = [r for r in rods if lo <= r.r_mean < hi]
        ranked = sorted(in_bin, key=lambda r: (r.sigma_xy, r.motion_max_axis_range, r.rod))
        chosen_best = ranked[: int(top_per_bin)]
        chosen_extra = ranked[int(top_per_bin) : int(top_per_bin) + int(extra_per_bin)]
        selected.extend((r, "best") for r in chosen_best)
        selected.extend((r, "extra") for r in chosen_extra)
        summary.append(
            {
                "bin_index": int(i),
                "r_lo": lo,
                "r_hi": hi,
                "n_candidates": int(len(in_bin)),
                "n_best": int(len(chosen_best)),
                "n_extra": int(len(chosen_extra)),
                "best_rods": [r.rod for r in chosen_best],
                "extra_rods": [r.rod for r in chosen_extra],
            }
        )
    return selected, summary


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    dataset_root = Path.cwd() / DATASET_DIR
    out_dir = dataset_root / OUTPUT_SUBDIR
    out_dir.mkdir(parents=True, exist_ok=True)

    rods = _load_all_rods(dataset_root)
    if not rods:
        raise SystemExit("No rods found in pending/good folders.")

    selected, bin_summary = _select_best_per_r_bin(
        rods,
        n_bins=N_R_BINS,
        top_per_bin=TOP_PER_BIN,
        extra_per_bin=EXTRA_PER_BIN,
    )
    if not selected:
        raise SystemExit("No rods selected for sigma_xy vs r plot.")

    rows = []
    for item, tier in selected:
        rows.append(
            {
                "rod": item.rod,
                "source": item.source,
                "tier": tier,
                "n_frames": item.n_frames,
                "r_mean": item.r_mean,
                "r_std": item.r_std,
                "sigma_x": item.sigma_x,
                "sigma_y": item.sigma_y,
                "sigma_xy": item.sigma_xy,
                "motion_max_axis_range": item.motion_max_axis_range,
                "path": str(item.path),
            }
        )
    _write_csv(
        out_dir / "best_candidates_sigma_xy_vs_r.csv",
        rows,
        [
            "rod",
            "source",
            "tier",
            "n_frames",
            "r_mean",
            "r_std",
            "sigma_x",
            "sigma_y",
            "sigma_xy",
            "motion_max_axis_range",
            "path",
        ],
    )

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    tiers = [("best", "#1f77b4", f"best ({sum(1 for _, t in selected if t == 'best')})"), ("extra", "#ff7f0e", f"extra ({sum(1 for _, t in selected if t == 'extra')})")]
    for tier, color, label in tiers:
        sub = [r for r, t in selected if t == tier]
        if not sub:
            continue
        ax.scatter(
            [r.r_mean for r in sub],
            [r.sigma_xy for r in sub],
            s=38,
            alpha=0.9,
            color=color,
            label=label,
        )
    ax.set_xlabel("Mean r")
    ax.set_ylabel("sigma_xy")
    ax.set_title(f"Best {TOP_PER_BIN} + extra {EXTRA_PER_BIN} rods per r bin by smallest sigma_xy")
    ax.grid(True, alpha=0.22)
    ax.legend(loc="best", frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / "best_candidates_sigma_xy_vs_r.png", dpi=220)
    plt.close(fig)

    summary = {
        "dataset_root": str(dataset_root),
        "output_dir": str(out_dir),
        "pending_count": int(sum(1 for r in rods if r.source == "pending")),
        "good_count": int(sum(1 for r in rods if r.source == "good")),
        "total_count": int(len(rods)),
        "n_r_bins": int(N_R_BINS),
        "top_per_bin": int(TOP_PER_BIN),
        "extra_per_bin": int(EXTRA_PER_BIN),
        "selected_count": int(len(selected)),
        "best_count": int(sum(1 for _, t in selected if t == "best")),
        "extra_count": int(sum(1 for _, t in selected if t == "extra")),
        "bin_summary": bin_summary,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
