"""
detection_alg.py

Standalone, editable spot-detection + polarization XY trace extraction for Polarcam stacks.

Assumptions:
- Input stack is a NumPy .npy file with shape (T, H, W), dtype uint16 (12-bit data in 0..4095).
- Polarization mosaic layout (matches your app):
  (row%2,col%2): (0,0)->90°, (0,1)->45°, (1,0)->135°, (1,1)->0°

What it does:
1) Builds an 8-bit "view" of the first frame using a LUT (floor/cap/gamma).
2) Detects bright "spots" on that 8-bit view via percentile threshold + morphology + connected components.
3) For each spot mask, computes normalized XY traces over time:
      x(t) = (I0 - I90) / (I0 + I90 + eps)
      y(t) = (I45 - I135) / (I45 + I135 + eps)

Outputs (optional):
- overlay PNG (frame 0 + detected spot outlines + indices)
- per-spot CSVs of x,y per frame
- per-spot diagnostic PNGs (crop + channel pixels + XY scatter)

Dependencies:
- numpy
- scipy (ndimage)
- matplotlib (only if you request plots)

This is designed to be a "playground" module: tweak detection parameters, swap algorithms,
add filtering, etc.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
from scipy import ndimage as ndi


# -----------------------------
# Configuration
# -----------------------------
@dataclass
class LutConfig:
    floor: int = 1200
    cap: int = 4095
    gamma: float = 0.6


@dataclass
class SpotDetectConfig:
    thr_percentile: float = 99.7
    min_area: int = 8
    max_area: int = 5000
    grow_px: int = 1
    open_structure: int = 3  # opening kernel size (square)
    connectivity: int = 1    # ndi.label connectivity


@dataclass
class OutputConfig:
    write_overlay: bool = True
    write_per_spot: bool = True
    write_debug_pngs: bool = True
    outdir: Optional[Path] = None


# -----------------------------
# LUT (12-bit -> 8-bit)
# -----------------------------
def build_highlight_lut(floor: int, cap: int, gamma: float) -> np.ndarray:
    cap = max(floor + 1, min(int(cap), 4095))
    floor = max(0, min(int(floor), 4094))
    gamma = float(max(0.05, min(gamma, 10.0)))

    x = np.arange(4096, dtype=np.float32)
    t = (x - floor) / float(cap - floor)
    t = np.clip(t, 0.0, 1.0)
    y = np.power(t, gamma) * 255.0
    return np.clip(np.rint(y), 0, 255).astype(np.uint8)


# -----------------------------
# Polarization layout masks
# -----------------------------
def channel_parity_masks(h: int, w: int):
    """
    Returns boolean masks (H,W):
      m0, m45, m90, m135
    """
    r = (np.arange(h)[:, None] & 1)
    c = (np.arange(w)[None, :] & 1)
    m0 = (r == 1) & (c == 1)     # 0°
    m45 = (r == 0) & (c == 1)    # 45°
    m90 = (r == 0) & (c == 0)    # 90°
    m135 = (r == 1) & (c == 0)   # 135°
    return m0, m45, m90, m135


# -----------------------------
# Spot detection
# -----------------------------
def find_spots_on_view(
    view8: np.ndarray,
    cfg: SpotDetectConfig,
) -> list[np.ndarray]:
    """
    Detect spots on an 8-bit image.

    Returns:
        list of boolean masks (H,W), one per detected spot.
    """
    if view8.ndim != 2:
        raise ValueError(f"view8 must be 2D (H,W), got shape={view8.shape}")

    thr = float(np.percentile(view8, cfg.thr_percentile))
    mask = view8 >= thr

    k = int(max(1, cfg.open_structure))
    structure = np.ones((k, k), dtype=bool)
    mask = ndi.binary_opening(mask, structure=structure)

    lbl, n = ndi.label(mask, structure=ndi.generate_binary_structure(2, cfg.connectivity))
    if n == 0:
        return []

    masks: list[np.ndarray] = []
    for lab in range(1, n + 1):
        comp = (lbl == lab)
        area = int(comp.sum())
        if area < cfg.min_area or area > cfg.max_area:
            continue
        if cfg.grow_px > 0:
            g = int(cfg.grow_px)
            comp = ndi.binary_dilation(comp, structure=np.ones((2 * g + 1, 2 * g + 1), dtype=bool))
        masks.append(comp.astype(bool, copy=False))

    return masks


# -----------------------------
# XY trace computation
# -----------------------------
def compute_xy_traces_normalized(
    stack: np.ndarray,
    spot_masks: Sequence[np.ndarray],
    eps: float = 1e-9,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    stack: (T,H,W) uint16
    spot_masks: list of (H,W) bool masks

    Returns:
        list of (x_t, y_t) arrays per spot, each length T.
        If a spot doesn't contain pixels for all 4 channels, its traces are NaNs.
    """
    if stack.ndim != 3:
        raise ValueError(f"stack must be (T,H,W), got shape={stack.shape}")
    T, H, W = stack.shape

    m0, m45, m90, m135 = channel_parity_masks(H, W)
    traces: list[tuple[np.ndarray, np.ndarray]] = []

    for mask in spot_masks:
        if mask.shape != (H, W):
            raise ValueError(f"spot mask shape {mask.shape} does not match stack frame shape {(H,W)}")

        mask = mask.astype(bool, copy=False)
        m0_s = m0 & mask
        m45_s = m45 & mask
        m90_s = m90 & mask
        m135_s = m135 & mask

        # Channel safety: if any channel absent, return NaNs
        if not (m0_s.any() and m45_s.any() and m90_s.any() and m135_s.any()):
            traces.append((np.full(T, np.nan), np.full(T, np.nan)))
            continue

        x = np.empty(T, dtype=np.float64)
        y = np.empty(T, dtype=np.float64)

        # Loop over time; mean per channel within the spot mask
        for t in range(T):
            f = stack[t]
            c0 = float(f[m0_s].mean())
            c45 = float(f[m45_s].mean())
            c90 = float(f[m90_s].mean())
            c135 = float(f[m135_s].mean())

            x[t] = (c0 - c90) / (c0 + c90 + eps)
            y[t] = (c45 - c135) / (c45 + c135 + eps)

        traces.append((x, y))

    return traces


# -----------------------------
# Diagnostics / Output helpers
# -----------------------------
def _bbox_from_mask(mask: np.ndarray, pad: int, H: int, W: int):
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        return 0, H, 0, W
    y0 = max(0, int(ys.min()) - pad)
    y1 = min(H, int(ys.max()) + pad + 1)
    x0 = max(0, int(xs.min()) - pad)
    x1 = min(W, int(xs.max()) + pad + 1)
    return y0, y1, x0, x1


def save_overlay_png(view8: np.ndarray, spot_masks: Sequence[np.ndarray], out_path: Path) -> None:
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 6), dpi=120)
    plt.imshow(view8, cmap="gray", vmin=0, vmax=255)

    for i, m in enumerate(spot_masks):
        edge = m ^ ndi.binary_erosion(m, structure=np.ones((3, 3), dtype=bool))
        ys, xs = np.nonzero(edge)
        if ys.size:
            plt.scatter(xs, ys, s=2, alpha=0.9)

        cy, cx = ndi.center_of_mass(m)
        if np.isfinite(cx) and np.isfinite(cy):
            plt.text(cx, cy, f"{i}", color="yellow", fontsize=9, ha="center", va="center")

    plt.title("Frame 0 with detected spots")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def save_per_spot_outputs(
    traces: Sequence[tuple[np.ndarray, np.ndarray]],
    view8: np.ndarray,
    spot_masks: Sequence[np.ndarray],
    outdir: Path,
    write_debug_pngs: bool = True,
) -> None:
    """
    Writes:
      - xy_sN.csv per spot (x,y per frame)
      - xy_sN.png per spot (optional): crop + channel pixels + XY scatter
    """
    outdir.mkdir(parents=True, exist_ok=True)

    H, W = view8.shape
    m0, m45, m90, m135 = channel_parity_masks(H, W)

    # Channel colors for debug plots
    ch_colors = {
        "0°": "#e41a1c",
        "45°": "#377eb8",
        "90°": "#4daf4a",
        "135°": "#ff7f00",
    }

    for i, (xs, ys) in enumerate(traces):
        mask = spot_masks[i].astype(bool, copy=False)

        # CSV
        csv_path = outdir / f"xy_s{i}.csv"
        arr = np.column_stack([xs.astype(float), ys.astype(float)])
        np.savetxt(csv_path, arr, delimiter=",", header="x,y", comments="", fmt="%.6f")

        if not write_debug_pngs:
            continue

        import matplotlib.pyplot as plt

        y0, y1, x0, x1 = _bbox_from_mask(mask, pad=4, H=H, W=W)
        crop = view8[y0:y1, x0:x1]

        m0_s = (m0 & mask)[y0:y1, x0:x1]
        m45_s = (m45 & mask)[y0:y1, x0:x1]
        m90_s = (m90 & mask)[y0:y1, x0:x1]
        m135_s = (m135 & mask)[y0:y1, x0:x1]
        edge = (mask ^ ndi.binary_erosion(mask, structure=np.ones((3, 3), dtype=bool)))[y0:y1, x0:x1]

        fig = plt.figure(figsize=(10, 4.8), dpi=130)
        gs = fig.add_gridspec(1, 2, width_ratios=[1.1, 1.0], wspace=0.28)

        ax0 = fig.add_subplot(gs[0, 0])
        ax0.imshow(crop, cmap="gray", vmin=0, vmax=255)

        ey, ex = np.nonzero(edge)
        if ey.size:
            ax0.scatter(ex, ey, s=6, c="yellow", alpha=0.9, label="mask edge")

        for m_sel, label, color in [
            (m0_s, "0°", ch_colors["0°"]),
            (m45_s, "45°", ch_colors["45°"]),
            (m90_s, "90°", ch_colors["90°"]),
            (m135_s, "135°", ch_colors["135°"]),
        ]:
            yy, xx = np.nonzero(m_sel)
            if yy.size:
                ax0.scatter(xx, yy, s=8, c=color, alpha=0.85, label=label)

        ax0.set_title(f"Spot s{i} — pixels used per channel")
        ax0.set_xlim([0, crop.shape[1]])
        ax0.set_ylim([crop.shape[0], 0])
        ax0.set_xticks([])
        ax0.set_yticks([])
        ax0.legend(loc="upper right", frameon=True, fontsize=8)

        ax1 = fig.add_subplot(gs[0, 1])
        m_valid = np.isfinite(xs) & np.isfinite(ys)
        if m_valid.any():
            ax1.scatter(xs[m_valid], ys[m_valid], s=14, alpha=0.9)

        ax1.axhline(0, lw=0.8, alpha=0.6, color="k")
        ax1.axvline(0, lw=0.8, alpha=0.6, color="k")
        ax1.set_xlim(-1, 1)
        ax1.set_ylim(-1, 1)
        ax1.set_aspect("equal", adjustable="box")
        ax1.grid(True, ls=":", alpha=0.4)
        ax1.set_xlabel("(I0 − I90) / (I0 + I90)")
        ax1.set_ylabel("(I45 − I135) / (I45 + I135)")
        ax1.set_title(f"Spot s{i} — XY scatter")

        fig.tight_layout()
        png_path = outdir / f"xy_s{i}.png"
        fig.savefig(png_path)
        plt.close(fig)


# -----------------------------
# End-to-end runner
# -----------------------------
@dataclass
class DetectionResult:
    view8: np.ndarray
    spot_masks: list[np.ndarray]
    traces: list[tuple[np.ndarray, np.ndarray]]


def run_detection(
    stack: np.ndarray,
    lut_cfg: LutConfig,
    det_cfg: SpotDetectConfig,
) -> DetectionResult:
    """
    Runs LUT->view8, spot detection on frame0, then XY traces for each spot.
    """
    if stack.ndim != 3:
        raise ValueError(f"Expected (T,H,W) stack; got shape={stack.shape}")
    if stack.dtype != np.uint16:
        stack = stack.astype(np.uint16, copy=False)

    lut = build_highlight_lut(lut_cfg.floor, lut_cfg.cap, lut_cfg.gamma)
    view8 = lut[stack[0]]  # first frame

    masks = find_spots_on_view(view8, det_cfg)
    traces = compute_xy_traces_normalized(stack, masks)

    return DetectionResult(view8=view8, spot_masks=masks, traces=traces)


# -----------------------------
# CLI
# -----------------------------
def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Standalone spot detection + XY trace extraction for Polarcam stacks.")
    ap.add_argument("stack_path", type=Path, help="Path to stack_*.npy with shape (T,H,W), uint16")

    ap.add_argument("--floor", type=int, default=1200, help="LUT floor (12-bit DN)")
    ap.add_argument("--cap", type=int, default=4095, help="LUT cap (12-bit DN)")
    ap.add_argument("--gamma", type=float, default=0.6, help="LUT gamma")

    ap.add_argument("--thr_pct", type=float, default=99.7, help="Threshold percentile for spot finding")
    ap.add_argument("--min_area", type=int, default=8, help="Min spot area (px)")
    ap.add_argument("--max_area", type=int, default=5000, help="Max spot area (px)")
    ap.add_argument("--grow", type=int, default=1, help="Grow each spot mask by N pixels")
    ap.add_argument("--open_k", type=int, default=3, help="Binary opening kernel size (square)")
    ap.add_argument("--conn", type=int, default=1, help="Connectivity for labeling (1 or 2)")

    ap.add_argument("--outdir", type=Path, default=None, help="Output directory (default: alongside stack)")
    ap.add_argument("--no_overlay", action="store_true", help="Do not write overlay PNG")
    ap.add_argument("--no_per_spot", action="store_true", help="Do not write per-spot outputs")
    ap.add_argument("--no_debug_pngs", action="store_true", help="Do not write per-spot debug PNGs (still writes CSVs)")
    return ap.parse_args()


def main() -> None:
    args = _parse_args()

    stack_path: Path = args.stack_path
    if not stack_path.exists():
        raise SystemExit(f"Stack not found: {stack_path}")

    outdir = args.outdir or (stack_path.parent / f"spot_playground_{time.strftime('%Y%m%d-%H%M%S')}")
    outdir.mkdir(parents=True, exist_ok=True)

    # Load stack (memmap keeps it lightweight)
    stack = np.load(stack_path, mmap_mode="r")
    if stack.ndim != 3:
        raise SystemExit(f"Expected (T,H,W) array; got shape={stack.shape}")

    T, H, W = stack.shape
    print(f"[info] stack: T={T}, H={H}, W={W}, dtype={stack.dtype}")

    lut_cfg = LutConfig(floor=args.floor, cap=args.cap, gamma=args.gamma)
    det_cfg = SpotDetectConfig(
        thr_percentile=args.thr_pct,
        min_area=args.min_area,
        max_area=args.max_area,
        grow_px=args.grow,
        open_structure=args.open_k,
        connectivity=args.conn,
    )

    res = run_detection(stack, lut_cfg, det_cfg)
    print(f"[info] detected {len(res.spot_masks)} spot(s)")

    if not args.no_overlay:
        overlay_path = outdir / "frame0_spots.png"
        save_overlay_png(res.view8, res.spot_masks, overlay_path)
        print(f"[ok] overlay -> {overlay_path}")

    if not args.no_per_spot:
        per_spot_dir = outdir / "per_spot"
        save_per_spot_outputs(
            res.traces,
            res.view8,
            res.spot_masks,
            per_spot_dir,
            write_debug_pngs=(not args.no_debug_pngs),
        )
        print(f"[ok] per-spot outputs -> {per_spot_dir}")

    print(f"[done] outputs in: {outdir}")


if __name__ == "__main__":
    main()
