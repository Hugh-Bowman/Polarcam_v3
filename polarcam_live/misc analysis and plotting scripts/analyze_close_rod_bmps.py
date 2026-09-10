from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage as ndi
from scipy.signal import find_peaks


LEFT_CROP_WIDTH = 100
PEAK_SIGMA = 1.2
PEAK_MIN_DIST = 6
REFINE_RADIUS = 4
PROFILE_MARGIN_PX = 8.0
PROFILE_STEP_PX = 0.25
PROFILE_HALF_WIDTH_PX = 0.7
TIGHT_CROP_PAD_PX = 8
UM_PER_PIXEL = 0.071


def _load_gray(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(path)
    if img.ndim == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def _pick_two_peaks(gray_crop: np.ndarray) -> list[tuple[float, float]]:
    smoothed = ndi.gaussian_filter(gray_crop.astype(np.float32), sigma=PEAK_SIGMA)
    local_max = ndi.maximum_filter(
        smoothed,
        size=(2 * PEAK_MIN_DIST + 1, 2 * PEAK_MIN_DIST + 1),
        mode="nearest",
    )
    ys, xs = np.where(smoothed == local_max)
    values = smoothed[ys, xs]
    order = np.argsort(values)[::-1]

    peaks: list[tuple[float, float]] = []
    for idx in order:
        x = float(xs[idx])
        y = float(ys[idx])
        too_close = any((x - px) ** 2 + (y - py) ** 2 < PEAK_MIN_DIST**2 for px, py in peaks)
        if too_close:
            continue
        peaks.append((x, y))
        if len(peaks) == 2:
            break
    if len(peaks) != 2:
        raise RuntimeError("Could not find two rod peaks.")
    return sorted(peaks, key=lambda pt: pt[0])


def _refine_center(gray_crop: np.ndarray, seed_xy: tuple[float, float]) -> tuple[float, float]:
    sx, sy = seed_xy
    x0 = max(0, int(np.floor(sx)) - REFINE_RADIUS)
    x1 = min(gray_crop.shape[1], int(np.floor(sx)) + REFINE_RADIUS + 1)
    y0 = max(0, int(np.floor(sy)) - REFINE_RADIUS)
    y1 = min(gray_crop.shape[0], int(np.floor(sy)) + REFINE_RADIUS + 1)
    patch = gray_crop[y0:y1, x0:x1].astype(np.float64)
    if patch.size == 0:
        return sx, sy

    baseline = float(np.percentile(patch, 20.0))
    weights = np.clip(patch - baseline, 0.0, None)
    if float(weights.sum()) <= 0.0:
        return sx, sy

    yy, xx = np.mgrid[y0:y1, x0:x1]
    cx = float((weights * xx).sum() / weights.sum())
    cy = float((weights * yy).sum() / weights.sum())
    return cx, cy


def _sample_line_sum(
    gray_crop: np.ndarray,
    start_xy: np.ndarray,
    end_xy: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    vec = end_xy - start_xy
    length = float(np.hypot(vec[0], vec[1]))
    if length <= 0.0:
        raise RuntimeError("Rod centres are identical.")

    unit = vec / length
    perp = np.array([-unit[1], unit[0]], dtype=np.float64)
    distances = np.arange(0.0, length + PROFILE_STEP_PX, PROFILE_STEP_PX, dtype=np.float64)
    center_points = start_xy[None, :] + distances[:, None] * unit[None, :]

    offsets = (-PROFILE_HALF_WIDTH_PX, 0.0, PROFILE_HALF_WIDTH_PX)
    profile = np.zeros(distances.shape, dtype=np.float64)
    for offset in offsets:
        points = center_points + offset * perp[None, :]
        rows = points[:, 1]
        cols = points[:, 0]
        profile += ndi.map_coordinates(
            gray_crop.astype(np.float64),
            [rows, cols],
            order=1,
            mode="nearest",
        )
    profile /= float(len(offsets))
    return distances, profile, center_points


def _tight_crop_bounds(
    image_shape: tuple[int, int],
    p0_xy: np.ndarray,
    p1_xy: np.ndarray,
    centres: list[tuple[float, float]],
) -> tuple[int, int, int, int]:
    xs = [p0_xy[0], p1_xy[0], *(c[0] for c in centres)]
    ys = [p0_xy[1], p1_xy[1], *(c[1] for c in centres)]
    x0 = max(0, int(np.floor(min(xs))) - TIGHT_CROP_PAD_PX)
    x1 = min(image_shape[1], int(np.ceil(max(xs))) + TIGHT_CROP_PAD_PX + 1)
    y0 = max(0, int(np.floor(min(ys))) - TIGHT_CROP_PAD_PX)
    y1 = min(image_shape[0], int(np.ceil(max(ys))) + TIGHT_CROP_PAD_PX + 1)
    return x0, x1, y0, y1


def _find_profile_peak_positions(distances_px: np.ndarray, profile: np.ndarray) -> np.ndarray:
    peaks, props = find_peaks(profile, prominence=max(5.0, 0.08 * float(np.max(profile))))
    if peaks.size >= 2:
        order = np.argsort(props["prominences"])[::-1][:2]
        chosen = np.sort(peaks[order])
        return distances_px[chosen]

    fallback = np.array(
        [PROFILE_MARGIN_PX, float(distances_px[-1] - PROFILE_MARGIN_PX)],
        dtype=np.float64,
    )
    return fallback


def _build_outputs_for_image(path: Path, out_dir: Path) -> dict[str, float | str]:
    gray = _load_gray(path)
    gray_crop = gray[:, :LEFT_CROP_WIDTH]

    peaks = _pick_two_peaks(gray_crop)
    centres = [_refine_center(gray_crop, peak) for peak in peaks]
    centres = sorted(centres, key=lambda pt: pt[0])
    c1 = np.array(centres[0], dtype=np.float64)
    c2 = np.array(centres[1], dtype=np.float64)

    join_vec = c2 - c1
    separation = float(np.hypot(join_vec[0], join_vec[1]))
    unit = join_vec / separation
    start_xy = c1 - PROFILE_MARGIN_PX * unit
    end_xy = c2 + PROFILE_MARGIN_PX * unit

    distances, profile, _ = _sample_line_sum(gray_crop, start_xy, end_xy)
    center_marks = np.array([PROFILE_MARGIN_PX, PROFILE_MARGIN_PX + separation], dtype=np.float64)
    peak_marks = _find_profile_peak_positions(distances, profile)
    peak_sep_um = float(abs(peak_marks[1] - peak_marks[0]) * UM_PER_PIXEL)
    x0, x1, y0, y1 = _tight_crop_bounds(gray_crop.shape, start_xy, end_xy, centres)
    gray_tight = gray_crop[y0:y1, x0:x1]

    stem = path.stem
    overlay_path = out_dir / f"{stem}_annotated.png"
    profile_path = out_dir / f"{stem}_profile.png"
    summary_path = out_dir / f"{stem}_summary.png"

    fig, ax = plt.subplots(figsize=(7.0, 3.5), dpi=180)
    ax.imshow(gray_tight, cmap="gray", interpolation="nearest")
    ax.plot([start_xy[0] - x0, end_xy[0] - x0], [start_xy[1] - y0, end_xy[1] - y0], color="cyan", linewidth=1.6)
    ax.scatter(
        [c1[0] - x0, c2[0] - x0],
        [c1[1] - y0, c2[1] - y0],
        s=42,
        c=["lime", "orange"],
        edgecolors="black",
        linewidths=0.7,
    )
    ax.set_title(path.name)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(overlay_path, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.0, 3.5), dpi=180)
    distances_um = distances * UM_PER_PIXEL
    peak_marks_um = peak_marks * UM_PER_PIXEL
    ax.plot(distances_um, profile, color="tab:blue", linewidth=1.6)
    ax.axvline(center_marks[0] * UM_PER_PIXEL, color="lime", linestyle="--", linewidth=1.0)
    ax.axvline(center_marks[1] * UM_PER_PIXEL, color="orange", linestyle="--", linewidth=1.0)
    y_annot = float(np.max(profile) * 0.90)
    ax.annotate(
        "",
        xy=(peak_marks_um[1], y_annot),
        xytext=(peak_marks_um[0], y_annot),
        arrowprops=dict(arrowstyle="<->", color="crimson", linewidth=1.2),
    )
    ax.text(
        float(np.mean(peak_marks_um)),
        y_annot + 0.03 * float(np.max(profile)),
        f"{peak_sep_um:.2f} μm",
        color="crimson",
        ha="center",
        va="bottom",
        fontsize=10,
        fontweight="bold",
    )
    ax.set_title(f"{path.name} line-sum profile")
    ax.set_xlabel("Distance along centre line (μm)")
    ax.set_ylabel("Summed intensity")
    fig.tight_layout()
    fig.savefig(profile_path, bbox_inches="tight")
    plt.close(fig)

    fig, (ax_img, ax_prof) = plt.subplots(
        1,
        2,
        figsize=(11.5, 4.2),
        dpi=180,
        gridspec_kw={"width_ratios": [1.0, 1.15]},
    )
    ax_img.imshow(gray_tight, cmap="gray", interpolation="nearest")
    ax_img.plot([start_xy[0] - x0, end_xy[0] - x0], [start_xy[1] - y0, end_xy[1] - y0], color="cyan", linewidth=1.7)
    ax_img.scatter(
        [c1[0] - x0, c2[0] - x0],
        [c1[1] - y0, c2[1] - y0],
        s=48,
        c=["lime", "orange"],
        edgecolors="black",
        linewidths=0.8,
    )
    ax_img.set_title("Tight crop with centres")
    ax_img.axis("off")
    bar_w_px = min(40.0, max(12.0, float(gray_tight.shape[1]) * 0.55))
    bar_x0 = 2.0
    bar_x1 = bar_x0 + bar_w_px
    bar_y = float(gray_tight.shape[0] - 3)
    ax_img.plot([bar_x0, bar_x1], [bar_y, bar_y], color="white", linewidth=2.0, solid_capstyle="butt")
    ax_img.text(0.5 * (bar_x0 + bar_x1), bar_y - 1.5, f"width {bar_w_px:.0f} px", color="white", ha="center", va="bottom", fontsize=9)

    ax_prof.plot(distances_um, profile, color="tab:blue", linewidth=1.7)
    ax_prof.axvline(center_marks[0] * UM_PER_PIXEL, color="lime", linestyle="--", linewidth=1.0, label="Rod 1 centre")
    ax_prof.axvline(center_marks[1] * UM_PER_PIXEL, color="orange", linestyle="--", linewidth=1.0, label="Rod 2 centre")
    ax_prof.annotate(
        "",
        xy=(peak_marks_um[1], y_annot),
        xytext=(peak_marks_um[0], y_annot),
        arrowprops=dict(arrowstyle="<->", color="crimson", linewidth=1.2),
    )
    ax_prof.text(
        float(np.mean(peak_marks_um)),
        y_annot + 0.03 * float(np.max(profile)),
        f"{peak_sep_um:.2f} μm",
        color="crimson",
        ha="center",
        va="bottom",
        fontsize=10,
        fontweight="bold",
    )
    ax_prof.set_title("Diagonal-aware line-sum intensity")
    ax_prof.set_xlabel("Distance along centre line (μm)")
    ax_prof.set_ylabel("Summed intensity")
    ax_prof.legend(loc="upper right", frameon=False)

    fig.suptitle(path.name)
    fig.tight_layout()
    fig.savefig(summary_path, bbox_inches="tight")
    plt.close(fig)

    return {
        "file": path.name,
        "center1_x": float(c1[0]),
        "center1_y": float(c1[1]),
        "center2_x": float(c2[0]),
        "center2_y": float(c2[1]),
        "separation_px": separation,
        "peak_separation_um": peak_sep_um,
        "annotated_png": overlay_path.name,
        "profile_png": profile_path.name,
        "summary_png": summary_path.name,
    }


def _save_combined_2x4(rows: list[dict[str, float | str]], input_dir: Path, out_dir: Path) -> None:
    if len(rows) != 4:
        return

    plt.rcParams.update(
        {
            "font.size": 20,
            "axes.titlesize": 26,
            "axes.labelsize": 26,
            "xtick.labelsize": 20,
            "ytick.labelsize": 20,
        }
    )

    prepared: list[dict[str, object]] = []
    max_w = 0
    max_h = 0
    for row in rows:
        file_name = str(row["file"])
        path = input_dir / file_name
        gray = _load_gray(path)
        gray_crop = gray[:, :LEFT_CROP_WIDTH]
        c1 = np.array([float(row["center1_x"]), float(row["center1_y"])], dtype=np.float64)
        c2 = np.array([float(row["center2_x"]), float(row["center2_y"])], dtype=np.float64)
        vec = c2 - c1
        unit = vec / float(np.hypot(vec[0], vec[1]))
        start_xy = c1 - PROFILE_MARGIN_PX * unit
        end_xy = c2 + PROFILE_MARGIN_PX * unit
        x0, x1, y0, y1 = _tight_crop_bounds(gray_crop.shape, start_xy, end_xy, [(float(c1[0]), float(c1[1])), (float(c2[0]), float(c2[1]))])
        gray_tight = gray_crop[y0:y1, x0:x1]
        max_h = max(max_h, int(gray_tight.shape[0]))
        max_w = max(max_w, int(gray_tight.shape[1]))

        distances_px, profile, _ = _sample_line_sum(gray_crop, start_xy, end_xy)
        profile = profile / max(1e-12, float(np.max(profile)))
        distances_um = distances_px * UM_PER_PIXEL
        peak_marks_px = _find_profile_peak_positions(distances_px, profile)
        peak_marks_um = peak_marks_px * UM_PER_PIXEL
        peak_sep_um = float(abs(peak_marks_um[1] - peak_marks_um[0]))
        y_annot = 0.88

        prepared.append(
            {
                "path": path,
                "gray_tight": gray_tight,
                "c1": c1,
                "c2": c2,
                "start_xy": start_xy,
                "end_xy": end_xy,
                "x0": x0,
                "y0": y0,
                "distances_um": distances_um,
                "profile": profile,
                "peak_marks_um": peak_marks_um,
                "peak_sep_um": peak_sep_um,
                "y_annot": y_annot,
            }
        )

    fig, axes = plt.subplots(
        4,
        2,
        figsize=(11.5, 18.0),
        dpi=220,
        gridspec_kw={"width_ratios": [1.0, 1.7]},
    )
    for idx, item in enumerate(prepared):
        path = item["path"]
        gray_tight = item["gray_tight"]
        c1 = item["c1"]
        c2 = item["c2"]
        start_xy = item["start_xy"]
        end_xy = item["end_xy"]
        x0 = int(item["x0"])
        y0 = int(item["y0"])
        distances_um = item["distances_um"]
        profile = item["profile"]
        peak_marks_um = item["peak_marks_um"]
        peak_sep_um = float(item["peak_sep_um"])
        y_annot = float(item["y_annot"])

        ax_img = axes[idx, 0]
        ax_prof = axes[idx, 1]

        canvas = np.zeros((max_h, max_w), dtype=gray_tight.dtype)
        y_off = (max_h - gray_tight.shape[0]) // 2
        x_off = (max_w - gray_tight.shape[1]) // 2
        canvas[y_off:y_off + gray_tight.shape[0], x_off:x_off + gray_tight.shape[1]] = gray_tight

        ax_img.imshow(canvas, cmap="gray", interpolation="nearest")
        ax_img.plot(
            [start_xy[0] - x0 + x_off, end_xy[0] - x0 + x_off],
            [start_xy[1] - y0 + y_off, end_xy[1] - y0 + y_off],
            color="cyan",
            linewidth=2.0,
        )
        ax_img.scatter(
            [c1[0] - x0 + x_off, c2[0] - x0 + x_off],
            [c1[1] - y0 + y_off, c2[1] - y0 + y_off],
            s=50,
            c=["lime", "orange"],
            edgecolors="black",
            linewidths=0.8,
        )
        if idx == 0:
            ax_img.set_title("2 rods raw image", pad=14)
        ax_img.axis("off")
        if idx == 3:
            bar_w_px = 20.0
            bar_x0 = 3.0
            bar_x1 = bar_x0 + bar_w_px
            bar_y = float(max_h - 4)
            ax_img.plot([bar_x0, bar_x1], [bar_y, bar_y], color="white", linewidth=2.4, solid_capstyle="butt")
            ax_img.text(0.5 * (bar_x0 + bar_x1), bar_y - 1.8, "20 px", color="white", ha="center", va="bottom", fontsize=22)

        ax_prof.plot(distances_um, profile, color="tab:blue", linewidth=2.2)
        ax_prof.axvline(peak_marks_um[0], color="orange", linestyle="--", linewidth=1.4)
        ax_prof.axvline(peak_marks_um[1], color="orange", linestyle="--", linewidth=1.4)
        ax_prof.annotate(
            "",
            xy=(peak_marks_um[1], y_annot),
            xytext=(peak_marks_um[0], y_annot),
            arrowprops=dict(arrowstyle="<->", color="crimson", linewidth=1.6),
        )
        ax_prof.text(
            float(np.mean(peak_marks_um)),
            y_annot + 0.05 * float(np.max(profile)),
            f"{peak_sep_um:.2f} μm",
            color="crimson",
            ha="center",
            va="bottom",
            fontsize=22,
            fontweight="bold",
        )
        ax_prof.set_ylim(0.0, 1.08)
        if idx == 0:
            ax_prof.set_title("Intensity along line of centres", pad=14)
        if idx == 3:
            ax_prof.set_xlabel("Distance along centre line (μm)")
        else:
            ax_prof.set_xlabel("")
            ax_prof.tick_params(labelbottom=False)
        ax_prof.set_ylabel("Intensity (AU)")

    fig.tight_layout()
    fig.savefig(out_dir / "combined_2x4_close_rods.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Find two close rods in BMPs, mark the centres and joining line, and extract a diagonal-aware intensity profile."
    )
    parser.add_argument("input_dir", type=Path, help="Folder containing the BMP images.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional output folder. Default: <input_dir>/two_rod_cross_sections",
    )
    args = parser.parse_args()

    input_dir = args.input_dir.resolve()
    output_dir = (args.output_dir or (input_dir / "two_rod_cross_sections")).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    bmp_paths = sorted(input_dir.glob("*.bmp")) + sorted(input_dir.glob("*.BMP"))
    seen: set[Path] = set()
    bmp_paths = [p for p in bmp_paths if not (p in seen or seen.add(p))]
    if not bmp_paths:
        raise SystemExit(f"No BMP files found in {input_dir}")

    rows: list[dict[str, float | str]] = []
    for path in bmp_paths:
        rows.append(_build_outputs_for_image(path, output_dir))

    csv_path = output_dir / "centres_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "file",
                "center1_x",
                "center1_y",
                "center2_x",
                "center2_y",
                "separation_px",
                "peak_separation_um",
                "annotated_png",
                "profile_png",
                "summary_png",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    _save_combined_2x4(rows, input_dir, output_dir)

    print(f"Saved outputs to {output_dir}")


if __name__ == "__main__":
    main()
