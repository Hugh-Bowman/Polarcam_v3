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
OUTPUT_DIR = DATASET_ROOT / "plots" / "all_rods_mean_phi_distribution"
R_MIN = 0.5


@dataclass
class RodPhi:
    source: str
    rod: str
    mean_phi_deg: float
    phi_std_deg: float
    n_frames: int


def _wrap_phi_pi(phi_rad: np.ndarray) -> np.ndarray:
    return np.mod(phi_rad, np.pi)


def _circular_mean_phi_deg(phi_rad: np.ndarray) -> float:
    # Phi is pi-periodic, so compute the mean on doubled angles.
    z = np.exp(1j * 2.0 * phi_rad)
    mean_angle = 0.5 * np.angle(np.mean(z))
    return float(np.degrees(mean_angle) % 180.0)


def _periodic_phi_std_deg(phi_rad: np.ndarray, mean_phi_deg: float) -> float:
    mean_phi_rad = np.radians(mean_phi_deg)
    # Smallest signed difference on a pi-periodic domain.
    delta = ((phi_rad - mean_phi_rad + (0.5 * np.pi)) % np.pi) - (0.5 * np.pi)
    return float(np.degrees(np.std(delta)))


def _load_rods(root: Path) -> list[RodPhi]:
    rods: list[RodPhi] = []
    for source in SUBDIRS:
        src_dir = root / source
        if not src_dir.exists():
            continue
        for rod_dir in sorted([p for p in src_dir.iterdir() if p.is_dir()]):
            meta_path = rod_dir / "capture_maxfps_15x15_meta.json"
            if not meta_path.exists():
                continue
            payload = json.loads(meta_path.read_text(encoding="utf-8"))
            phi_series = payload.get("phi_series")
            xy_series = payload.get("xy_series")
            if not isinstance(phi_series, list) or not phi_series:
                continue
            if not isinstance(xy_series, list) or not xy_series:
                continue
            phi = np.asarray(phi_series, dtype=np.float64)
            xy = np.asarray(xy_series, dtype=np.float64)
            if xy.ndim != 2 or xy.shape[1] < 2:
                continue
            n = min(int(phi.shape[0]), int(xy.shape[0]))
            if n <= 0:
                continue
            phi = phi[:n]
            xy = xy[:n, :2]
            r = np.sqrt((xy[:, 0] * xy[:, 0]) + (xy[:, 1] * xy[:, 1]))
            valid = np.isfinite(phi) & np.isfinite(r) & (r > R_MIN)
            phi = phi[valid]
            if phi.size == 0:
                continue
            phi = _wrap_phi_pi(phi)
            mean_phi_deg = _circular_mean_phi_deg(phi)
            rods.append(
                RodPhi(
                    source=source,
                    rod=rod_dir.name,
                    mean_phi_deg=mean_phi_deg,
                    phi_std_deg=_periodic_phi_std_deg(phi, mean_phi_deg),
                    n_frames=int(phi.size),
                )
            )
    return rods


def _load_all_phi_points(root: Path) -> tuple[np.ndarray, dict[str, int]]:
    vals: list[np.ndarray] = []
    counts_by_source = {src: 0 for src in SUBDIRS}
    for source in SUBDIRS:
        src_dir = root / source
        if not src_dir.exists():
            continue
        for rod_dir in sorted([p for p in src_dir.iterdir() if p.is_dir()]):
            meta_path = rod_dir / "capture_maxfps_15x15_meta.json"
            if not meta_path.exists():
                continue
            payload = json.loads(meta_path.read_text(encoding="utf-8"))
            phi_series = payload.get("phi_series")
            xy_series = payload.get("xy_series")
            if not isinstance(phi_series, list) or not phi_series:
                continue
            if not isinstance(xy_series, list) or not xy_series:
                continue
            phi = np.asarray(phi_series, dtype=np.float64)
            xy = np.asarray(xy_series, dtype=np.float64)
            if xy.ndim != 2 or xy.shape[1] < 2:
                continue
            n = min(int(phi.shape[0]), int(xy.shape[0]))
            if n <= 0:
                continue
            phi = phi[:n]
            xy = xy[:n, :2]
            r = np.sqrt((xy[:, 0] * xy[:, 0]) + (xy[:, 1] * xy[:, 1]))
            valid = np.isfinite(phi) & np.isfinite(r) & (r > R_MIN)
            phi = phi[valid]
            if phi.size == 0:
                continue
            phi = np.degrees(_wrap_phi_pi(phi))
            vals.append(phi)
            counts_by_source[source] += int(phi.size)
    if not vals:
        return np.asarray([], dtype=np.float64), counts_by_source
    return np.concatenate(vals), counts_by_source


def _write_csv(rods: list[RodPhi], out_dir: Path) -> None:
    rows = [
        {
            "source": r.source,
            "rod": r.rod,
            "mean_phi_deg": r.mean_phi_deg,
            "phi_std_deg": r.phi_std_deg,
            "n_frames": r.n_frames,
        }
        for r in rods
    ]
    with (out_dir / "mean_phi_by_rod.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _plot_hist(rods: list[RodPhi], out_dir: Path) -> None:
    vals = np.asarray([r.mean_phi_deg for r in rods], dtype=np.float64)
    bins = np.linspace(0.0, 180.0, 19)
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.hist(vals, bins=bins, color="#1f77b4", alpha=0.88, edgecolor="white")
    ax.set_xlabel("Mean phi (deg)")
    ax.set_ylabel("Rod count")
    ax.set_title("Mean phi distribution, all tumbling 25nm glycerol rods")
    ax.set_xlim(0.0, 180.0)
    ax.grid(True, axis="y", alpha=0.22)
    fig.tight_layout()
    fig.savefig(out_dir / "mean_phi_distribution_all_rods.png", dpi=220)
    plt.close(fig)


def _plot_point_hist(phi_deg: np.ndarray, out_dir: Path) -> None:
    bins = np.linspace(0.0, 180.0, 37)
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.hist(phi_deg, bins=bins, color="#1f77b4", alpha=0.88, edgecolor="white")
    ax.set_xlabel("Phi (deg)")
    ax.set_ylabel("Point count")
    ax.set_title("Phi distribution, all tumbling 25nm glycerol points with r > 0.5")
    ax.set_xlim(0.0, 180.0)
    ax.grid(True, axis="y", alpha=0.22)
    fig.tight_layout()
    fig.savefig(out_dir / "phi_distribution_all_points_r_gt_0p5.png", dpi=220)
    plt.close(fig)


def main() -> None:
    root = Path.cwd() / DATASET_ROOT
    out_dir = Path.cwd() / OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    rods = _load_rods(root)
    if not rods:
        raise SystemExit("No usable rods found in pending/good/bad.")
    phi_points_deg, point_counts_by_source = _load_all_phi_points(root)
    if phi_points_deg.size == 0:
        raise SystemExit("No usable phi points found with r > 0.5.")
    _write_csv(rods, out_dir)
    _plot_hist(rods, out_dir)
    _plot_point_hist(phi_points_deg, out_dir)
    summary = {
        "dataset_root": str(root.resolve()),
        "output_dir": str(out_dir.resolve()),
        "n_rods": int(len(rods)),
        "counts_by_source": {src: int(sum(1 for r in rods if r.source == src)) for src in SUBDIRS},
        "n_phi_points_r_gt_0p5": int(phi_points_deg.size),
        "phi_point_counts_by_source": point_counts_by_source,
        "phi_deg_range": [float(np.min(phi_points_deg)), float(np.max(phi_points_deg))],
        "mean_phi_deg_range": [
            float(min(r.mean_phi_deg for r in rods)),
            float(max(r.mean_phi_deg for r in rods)),
        ],
        "r_min": float(R_MIN),
        "method": "Phi point distribution uses all individual frames with r > 0.5 from xy_series across pending/good/bad. The per-rod mean-phi table and plot are also retained.",
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
