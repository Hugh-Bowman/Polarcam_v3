from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def _norm_corr(a: np.ndarray, b: np.ndarray) -> float:
    av = a.ravel().astype(np.float64)
    bv = b.ravel().astype(np.float64)
    av -= av.mean()
    bv -= bv.mean()
    an = np.linalg.norm(av)
    bn = np.linalg.norm(bv)
    if an == 0.0 or bn == 0.0:
        return 0.0
    return float(np.dot(av, bv) / (an * bn))


def _load_objective_image(path: Path) -> np.ndarray:
    obj = np.load(path, mmap_mode="r")
    if obj.ndim == 2:
        return obj.astype(np.float32)
    if obj.ndim == 3:
        return obj.astype(np.float32).mean(axis=0)
    raise ValueError(f"Unsupported covered objective shape: {obj.shape}")


def _as_stack(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 2:
        return arr[None, :, :]
    if arr.ndim == 3:
        return arr
    raise ValueError(f"Input must be 2D or 3D npy; got shape {arr.shape}")


def _subtract_first_n_mean(stack: np.ndarray, n_first: int) -> np.ndarray:
    n = stack.shape[0]
    n_use = min(max(1, int(n_first)), n)
    m = stack[:n_use].mean(axis=0)
    return stack - m[None, :, :]


def _build_target(input_path: Path, obj_img: np.ndarray, n_mean: int) -> np.ndarray:
    arr = np.load(input_path, mmap_mode="r")
    stack = _as_stack(arr).astype(np.float32)
    if stack.shape[1:] != obj_img.shape:
        raise ValueError(f"Input frame shape {stack.shape[1:]} != objective shape {obj_img.shape}")
    bgsub = stack - obj_img[None, :, :]
    centered = _subtract_first_n_mean(bgsub, n_first=n_mean)
    if centered.shape[0] == 1:
        return centered[0]
    return centered.mean(axis=0)


def _reference_centered(ref_stack: np.ndarray, obj_img: np.ndarray, n_frames: int, n_mean: int) -> np.ndarray:
    n = min(int(n_frames), ref_stack.shape[0])
    part = ref_stack[:n].astype(np.float32)
    if part.shape[1:] != obj_img.shape:
        raise ValueError(f"Reference frame shape {part.shape[1:]} != objective shape {obj_img.shape}")
    bgsub = part - obj_img[None, :, :]
    return _subtract_first_n_mean(bgsub, n_first=n_mean)


def _plot_corr(corr: np.ndarray, out_png: Path, title: str) -> tuple[int, float]:
    idx = int(np.argmax(corr))
    val = float(corr[idx])
    x = np.arange(corr.shape[0])
    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    ax.plot(x, corr, lw=1.6, color="C0")
    ax.scatter([idx], [val], color="crimson", zorder=3)
    ax.annotate(
        f"peak: frame {idx}, r={val:.6f}",
        xy=(idx, val),
        xytext=(8, 8),
        textcoords="offset points",
        color="crimson",
    )
    ax.set_xlabel("Frame index")
    ax.set_ylabel("Correlation to processed input target")
    ax.set_title(title)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_png, dpi=170)
    plt.close(fig)
    return idx, val


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Match a processed input image/stack against reference z-scroll frames by correlation."
        )
    )
    parser.add_argument("--input", type=Path, required=True, help="Input .npy (single frame 2D or stack 3D).")
    parser.add_argument(
        "--reference",
        type=Path,
        default=Path(r"recordings/2026-05-19/widefield_frames/2ms faster z.npy"),
        help="Reference z-scroll stack .npy (default: 2ms faster z).",
    )
    parser.add_argument(
        "--covered-objective",
        type=Path,
        default=Path(r"recordings/2026-05-13/widefield_frames/best covered objective.npy"),
        help="Covered objective profile .npy (2D or 3D stack).",
    )
    parser.add_argument(
        "--mean-first",
        type=int,
        default=45,
        help="Number of first frames used to compute mean image for mean subtraction.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(r"recordings/2026-05-19/widefield_frames/analysis_match_to_2ms_faster_z"),
        help="Output directory.",
    )
    args = parser.parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    obj_img = _load_objective_image(args.covered_objective)
    target = _build_target(args.input, obj_img=obj_img, n_mean=args.mean_first)
    ref = np.load(args.reference, mmap_mode="r")
    if ref.ndim != 3:
        raise ValueError(f"Reference must be a 3D stack; got shape {ref.shape}")

    report_lines = []
    report_lines.append(f"input={args.input}")
    report_lines.append(f"reference={args.reference}")
    report_lines.append(f"covered_objective={args.covered_objective}")
    report_lines.append(f"mean_first={int(args.mean_first)}")
    report_lines.append(f"target_shape={target.shape}")

    for n_ref in (50, 100):
        ref_centered = _reference_centered(ref, obj_img=obj_img, n_frames=n_ref, n_mean=args.mean_first)
        corr = np.array([_norm_corr(target, ref_centered[i]) for i in range(ref_centered.shape[0])], dtype=np.float64)
        np.savez(
            out_dir / f"corr_to_reference_first{n_ref}_data.npz",
            correlations=corr.astype(np.float32),
            best_idx=np.array(int(np.argmax(corr)), dtype=np.int32),
            best_corr=np.array(float(np.max(corr)), dtype=np.float64),
        )
        peak_idx, peak_val = _plot_corr(
            corr,
            out_png=out_dir / f"corr_to_reference_first{n_ref}.png",
            title=f"Correlation to reference frames 0..{n_ref - 1}",
        )
        report_lines.append(f"first{n_ref}: best_frame={peak_idx}, best_corr={peak_val:.6f}")

    (out_dir / "summary.txt").write_text("\n".join(report_lines), encoding="utf-8")
    print("\n".join(report_lines))
    print(f"saved={out_dir}")


if __name__ == "__main__":
    main()
