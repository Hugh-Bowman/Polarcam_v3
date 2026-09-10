from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


DATASET_ROOT = Path("datasets") / "tumbling 25nm glycerol"
SUBDIRS = ("pending", "good", "bad")
OUTPUT_DIR = DATASET_ROOT / "plots" / "balanced_phi_xy_points"
PHI_BIN_DEG = 5.0
RANGE_THRESHOLD = 1.0
RNG_SEED = 12345


@dataclass
class RecordingXY:
    source: str
    rod: str
    x: np.ndarray
    y: np.ndarray
    range_x: float
    range_y: float


def _phi_deg_from_xy(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    phi = 0.5 * np.arctan2(y, x)
    return np.degrees(np.mod(phi, np.pi))


def _load_filtered_recordings(root: Path) -> list[RecordingXY]:
    out: list[RecordingXY] = []
    for source in SUBDIRS:
        src_dir = root / source
        if not src_dir.exists():
            continue
        for rod_dir in sorted([p for p in src_dir.iterdir() if p.is_dir()]):
            meta_path = rod_dir / "capture_maxfps_15x15_meta.json"
            if not meta_path.exists():
                continue
            payload = json.loads(meta_path.read_text(encoding="utf-8"))
            metrics = dict(payload.get("xy_metrics") or {})
            range_x = metrics.get("range_x")
            range_y = metrics.get("range_y")
            if range_x is None or range_y is None:
                continue
            if not (float(range_x) > RANGE_THRESHOLD and float(range_y) > RANGE_THRESHOLD):
                continue
            xy_series = payload.get("xy_series")
            if not isinstance(xy_series, list) or not xy_series:
                continue
            xy = np.asarray(xy_series, dtype=np.float64)
            if xy.ndim != 2 or xy.shape[1] < 2:
                continue
            xy = xy[:, :2]
            valid = np.isfinite(xy[:, 0]) & np.isfinite(xy[:, 1])
            xy = xy[valid]
            if xy.size == 0:
                continue
            out.append(
                RecordingXY(
                    source=source,
                    rod=rod_dir.name,
                    x=np.asarray(xy[:, 0], dtype=np.float64),
                    y=np.asarray(xy[:, 1], dtype=np.float64),
                    range_x=float(range_x),
                    range_y=float(range_y),
                )
            )
    return out


def _pool_points(recordings: list[RecordingXY]) -> tuple[np.ndarray, np.ndarray]:
    x = np.concatenate([r.x for r in recordings])
    y = np.concatenate([r.y for r in recordings])
    return x, y


def _sample_balanced_points(
    x: np.ndarray,
    y: np.ndarray,
    phi_bin_deg: float,
    rng_seed: int,
) -> tuple[np.ndarray, np.ndarray, dict]:
    phi_deg = _phi_deg_from_xy(x, y)
    edges = np.arange(0.0, 180.0 + phi_bin_deg, phi_bin_deg, dtype=np.float64)
    bin_ids = np.digitize(phi_deg, edges, right=False) - 1
    bin_ids = np.clip(bin_ids, 0, len(edges) - 2)

    idx_by_bin: list[np.ndarray] = []
    counts: list[int] = []
    for i in range(len(edges) - 1):
        idx = np.flatnonzero(bin_ids == i)
        idx_by_bin.append(idx)
        counts.append(int(idx.size))

    min_count = int(min(counts))
    n_per_bin = max(1, min_count // 2)
    rng = np.random.default_rng(rng_seed)

    sampled_idx: list[np.ndarray] = []
    rows: list[dict] = []
    for i, idx in enumerate(idx_by_bin):
        choose = rng.choice(idx, size=n_per_bin, replace=False)
        sampled_idx.append(np.asarray(choose, dtype=np.int64))
        rows.append(
            {
                "phi_bin_start_deg": float(edges[i]),
                "phi_bin_end_deg": float(edges[i + 1]),
                "available_points": int(idx.size),
                "sampled_points": int(n_per_bin),
            }
        )

    sampled_idx_arr = np.concatenate(sampled_idx)
    return x[sampled_idx_arr], y[sampled_idx_arr], {
        "phi_bin_edges_deg": edges.tolist(),
        "least_populated_bin_count": int(min_count),
        "sampled_per_bin": int(n_per_bin),
        "bin_rows": rows,
    }


def _write_points_csv(path: Path, x: np.ndarray, y: np.ndarray) -> None:
    phi_deg = _phi_deg_from_xy(x, y)
    r = np.sqrt((x * x) + (y * y))
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["x", "y", "r", "phi_deg"])
        writer.writeheader()
        for xv, yv, rv, pv in zip(x, y, r, phi_deg):
            writer.writerow(
                {
                    "x": float(xv),
                    "y": float(yv),
                    "r": float(rv),
                    "phi_deg": float(pv),
                }
            )


def _write_bin_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["phi_bin_start_deg", "phi_bin_end_deg", "available_points", "sampled_points"],
        )
        writer.writeheader()
        writer.writerows(rows)


def _plot_phi_distribution(phi_deg: np.ndarray, out_path: Path) -> None:
    bins = np.arange(0.0, 185.0, 5.0, dtype=np.float64)
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.hist(phi_deg, bins=bins, color="#1f77b4", alpha=0.88, edgecolor="white")
    ax.set_xlabel("Phi (deg)")
    ax.set_ylabel("Point count")
    ax.set_title("Balanced phi distribution from pooled tumbling 25nm xy points")
    ax.set_xlim(0.0, 180.0)
    ax.grid(True, axis="y", alpha=0.22)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def _plot_r_distribution(r: np.ndarray, out_path: Path) -> None:
    bins = np.linspace(float(np.min(r)), float(np.max(r)), 50)
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.hist(r, bins=bins, color="#2ca02c", alpha=0.88, edgecolor="white", density=True)
    ax.set_xlabel("r")
    ax.set_ylabel("Density")
    ax.set_title("r density from balanced phi-bin sampled tumbling 25nm points")
    ax.grid(True, axis="y", alpha=0.22)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def main() -> None:
    root = Path.cwd() / DATASET_ROOT
    out_dir = Path.cwd() / OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    recordings = _load_filtered_recordings(root)
    if not recordings:
        raise SystemExit("No recordings passed the range filter.")

    x_all, y_all = _pool_points(recordings)
    x_sel, y_sel, sample_info = _sample_balanced_points(
        x=x_all,
        y=y_all,
        phi_bin_deg=float(PHI_BIN_DEG),
        rng_seed=int(RNG_SEED),
    )
    phi_sel_deg = _phi_deg_from_xy(x_sel, y_sel)
    r_sel = np.sqrt((x_sel * x_sel) + (y_sel * y_sel))

    _write_points_csv(out_dir / "balanced_sampled_xy_points.csv", x_sel, y_sel)
    _write_bin_csv(out_dir / "phi_bin_sampling_counts.csv", list(sample_info["bin_rows"]))
    _plot_phi_distribution(phi_sel_deg, out_dir / "phi_distribution_balanced_sampled_points.png")
    _plot_r_distribution(r_sel, out_dir / "r_density_balanced_sampled_points.png")

    summary = {
        "dataset_root": str(root.resolve()),
        "output_dir": str(out_dir.resolve()),
        "range_threshold": float(RANGE_THRESHOLD),
        "phi_bin_deg": float(PHI_BIN_DEG),
        "rng_seed": int(RNG_SEED),
        "n_recordings_passing_filter": int(len(recordings)),
        "recordings_by_source": {
            src: int(sum(1 for r in recordings if r.source == src)) for src in SUBDIRS
        },
        "n_pooled_points": int(x_all.size),
        "least_populated_bin_count": int(sample_info["least_populated_bin_count"]),
        "sampled_per_bin": int(sample_info["sampled_per_bin"]),
        "n_selected_points": int(x_sel.size),
        "r_range_selected": [float(np.min(r_sel)), float(np.max(r_sel))],
        "method": "Use only recordings with range_x > 1 and range_y > 1, pool all xy points, bin pooled points by phi in 5 degree bins, randomly sample half of the least populated bin size from each bin.",
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
