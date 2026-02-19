import argparse
from pathlib import Path

import numpy as np


DEFAULT_VIDEO_PATH = r"C:\IDS recordings\Sparse_stationary_rods_various_exposure\pol_after_pbs2.avi"
DEFAULT_OUT_NPY = r"C:\IDS recordings\Sparse_stationary_rods_various_exposure\gradient_rods2_frames.npy"


def _to_gray(frame: np.ndarray) -> np.ndarray:
    """
    Convert a video frame to grayscale float64.

    Supports:
    - (H,W) already grayscale
    - (H,W,3) or (H,W,4) color; uses channel mean (keeps things dependency-free)
    """
    if frame.ndim == 2:
        g = frame
    elif frame.ndim == 3:
        g = frame[..., :3].mean(axis=2)
    else:
        raise ValueError(f"Unexpected frame shape: {frame.shape}")
    return np.asarray(g, dtype=np.float64)


def iter_frames_cv2(video_path: Path):
    import cv2  # optional dependency; preferred if installed

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    try:
        # Some codec failures don't throw; they just return ok=False immediately.
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError(f"cv2 could not decode any frames from: {video_path}")
        yield frame
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            yield frame
    finally:
        cap.release()


def iter_frames_imageio(video_path: Path):
    import imageio.v2 as imageio  # optional dependency; requires ffmpeg support

    rdr = imageio.get_reader(str(video_path))
    try:
        for frame in rdr:
            yield frame
    finally:
        try:
            rdr.close()
        except Exception:
            pass


def iter_frames(video_path: Path):
    # Prefer OpenCV if available; fall back to imageio.
    try:
        yield from iter_frames_cv2(video_path)
        return
    except ModuleNotFoundError:
        pass
    except Exception:
        # If cv2 exists but fails to open/read, try imageio as a fallback.
        pass

    yield from iter_frames_imageio(video_path)


def video_to_npy_and_mean(
    video_path: Path,
    out_npy: Path,
    *,
    grayscale: bool = True,
    progress_every: int = 250,
) -> tuple[Path, np.ndarray, int]:
    """
    Load an AVI/video, write all frames to an .npy stack, and compute the per-pixel mean image.

    Returns:
      (out_npy, mean_image_float64, n_frames)
    """
    frames: list[np.ndarray] = []
    sum_img: np.ndarray | None = None
    n = 0

    for frame in iter_frames(video_path):
        if grayscale:
            g = _to_gray(frame)
            # Store as uint8 when possible (saves disk), but keep mean math in float64.
            if g.dtype != np.uint8:
                # If original is uint8, _to_gray returns float64. Convert back for storage.
                g_store = np.clip(np.rint(g), 0, 255).astype(np.uint8)
            else:
                g_store = g
            frames.append(g_store)
            if sum_img is None:
                sum_img = np.zeros_like(g, dtype=np.float64)
            if g.shape != sum_img.shape:
                raise RuntimeError(f"Frame shape changed: {g.shape} vs {sum_img.shape}")
            sum_img += g
        else:
            g = np.asarray(frame, dtype=np.float64)
            frames.append(np.asarray(frame))
            if sum_img is None:
                sum_img = np.zeros_like(g, dtype=np.float64)
            if g.shape != sum_img.shape:
                raise RuntimeError(f"Frame shape changed: {g.shape} vs {sum_img.shape}")
            sum_img += g

        n += 1
        if progress_every and (n % int(progress_every) == 0):
            print(f"Read frames: {n}")

    if sum_img is None or n == 0:
        raise RuntimeError("No frames read from video (decoder failed or empty file).")

    out_npy.parent.mkdir(parents=True, exist_ok=True)
    stack = np.stack(frames, axis=0)
    np.save(out_npy, stack)
    mean_img = sum_img / float(n)
    return out_npy, mean_img, n


def mean_2x2_positions(mean_img: np.ndarray) -> np.ndarray:
    h, w = mean_img.shape
    h2 = h - (h % 2)
    w2 = w - (w % 2)
    m = mean_img[:h2, :w2]
    out = np.zeros((2, 2), dtype=np.float64)
    out[0, 0] = m[0::2, 0::2].mean()
    out[0, 1] = m[0::2, 1::2].mean()
    out[1, 0] = m[1::2, 0::2].mean()
    out[1, 1] = m[1::2, 1::2].mean()
    return out


def compute_xy_from_mosaic(m2: np.ndarray, pattern: str) -> tuple[float, float]:
    """
    Compute anisotropy-like X,Y from 2x2 mosaic means.

    Default pattern ("ids") matches the mapping used elsewhere in this repo when
    ROI offsets are even (phase_x=phase_y=0):
      [ I90  I45 ]
      [ I135 I0  ]
    """
    p = pattern.strip().lower()
    if p in ("ids", "default"):
        I90 = float(m2[0, 0])
        I45 = float(m2[0, 1])
        I135 = float(m2[1, 0])
        I0 = float(m2[1, 1])
    else:
        raise ValueError(f"Unknown pattern: {pattern!r} (supported: ids)")

    eps = 1e-12
    X = (I0 - I90) / (I0 + I90 + eps)
    Y = (I45 - I135) / (I45 + I135 + eps)
    return float(X), float(Y)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Load an AVI/video, convert all frames to an .npy stack, compute the per-pixel mean "
            "image across all frames, then compute 2x2 mosaic-position means and anisotropy X,Y."
        )
    )
    ap.add_argument(
        "video",
        nargs="?",
        default=DEFAULT_VIDEO_PATH,
        help="Input AVI/video path",
    )
    ap.add_argument(
        "--out-npy",
        default=DEFAULT_OUT_NPY,
        help="Output .npy stack path (all frames).",
    )
    ap.add_argument(
        "--save-mean-npy",
        default=None,
        help="Optional path to save mean image as .npy",
    )
    ap.add_argument(
        "--pattern",
        default="ids",
        help="2x2 angle mapping pattern (default: ids)",
    )
    args = ap.parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        raise SystemExit(f"File not found: {video_path}")

    out_npy = Path(args.out_npy)
    out_npy, mean_img, n = video_to_npy_and_mean(video_path, out_npy)
    m2 = mean_2x2_positions(mean_img)
    X, Y = compute_xy_from_mosaic(m2, args.pattern)

    if args.save_mean_npy:
        np.save(Path(args.save_mean_npy), mean_img)

    print(f"Frames: {n}")
    print(f"Saved frames stack: {out_npy}")
    print(f"Mean image shape: {mean_img.shape[0]} x {mean_img.shape[1]}")
    print("2x2 mosaic position means:")
    print(f"[{m2[0,0]:.6f} {m2[0,1]:.6f}]")
    print(f"[{m2[1,0]:.6f} {m2[1,1]:.6f}]")
    print(f"X,Y = {X:.8f}, {Y:.8f}")


if __name__ == "__main__":
    main()
