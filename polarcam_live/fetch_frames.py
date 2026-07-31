from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from time import sleep, strftime

import numpy as np
from PySide6.QtWidgets import QApplication

from Controlling.controller.controller import Controller


log = logging.getLogger(__name__)
BACKGROUND_PROFILE_FILENAME = "background_profile.npy"
LEGACY_BACKGROUND_PROFILE_FILENAMES = ("background_profile_gaussian_sigma20.npy",)


def _as_saved_frame(frame: np.ndarray) -> np.ndarray:
    """Preserve native camera depth for saved recordings."""
    arr = np.asarray(frame)
    if arr.dtype in (np.uint8, np.uint16):
        return np.array(arr, copy=True)
    if np.issubdtype(arr.dtype, np.integer):
        return np.asarray(arr, dtype=np.uint16)
    return np.asarray(np.clip(arr, 0, 65535), dtype=np.uint16)


def _find_background_profile_path() -> Path | None:
    candidates = []
    for name in (BACKGROUND_PROFILE_FILENAME, *LEGACY_BACKGROUND_PROFILE_FILENAMES):
        candidates.append(Path.cwd() / name)
        candidates.append(Path(__file__).resolve().parent / name)
    for path in candidates:
        try:
            if path.exists():
                return path
        except Exception:
            continue
    return None


def _load_background_profile() -> tuple[np.ndarray | None, Path | None]:
    path = _find_background_profile_path()
    if path is None:
        return None, None
    try:
        prof = np.asarray(np.load(path, allow_pickle=False), dtype=np.float32)
    except Exception:
        return None, path
    if prof.ndim != 2:
        return None, path
    return prof, path


def _crop_background_profile(
    profile: np.ndarray,
    frame_shape: tuple[int, int],
    actual_roi: dict | None,
    requested_roi: tuple[int, int, int, int] | None,
) -> np.ndarray | None:
    fh, fw = int(frame_shape[0]), int(frame_shape[1])
    if tuple(profile.shape) == (fh, fw):
        return profile
    x0 = y0 = None
    if isinstance(actual_roi, dict):
        x0 = actual_roi.get("OffsetX", actual_roi.get("x"))
        y0 = actual_roi.get("OffsetY", actual_roi.get("y"))
    if (x0 is None or y0 is None) and requested_roi is not None:
        try:
            x0 = requested_roi[0]
            y0 = requested_roi[1]
        except Exception:
            x0 = None
            y0 = None
    if x0 is None or y0 is None:
        return None
    try:
        ix = int(round(float(x0)))
        iy = int(round(float(y0)))
    except Exception:
        return None
    if ix < 0 or iy < 0:
        return None
    if (iy + fh) > int(profile.shape[0]) or (ix + fw) > int(profile.shape[1]):
        return None
    return np.asarray(profile[iy : iy + fh, ix : ix + fw], dtype=np.float32)


def _subtract_background(
    frame: np.ndarray,
    profile: np.ndarray | None,
    actual_roi: dict | None,
    requested_roi: tuple[int, int, int, int] | None,
) -> np.ndarray:
    arr = np.asarray(frame)
    if profile is None or arr.ndim != 2:
        return np.array(arr, copy=True)
    bg = _crop_background_profile(profile, (int(arr.shape[0]), int(arr.shape[1])), actual_roi, requested_roi)
    if bg is None:
        return np.array(arr, copy=True)
    work = arr.astype(np.float32, copy=False) - bg
    np.maximum(work, 0.0, out=work)
    if arr.dtype == np.uint8:
        return np.asarray(np.clip(np.rint(work), 0.0, 255.0), dtype=np.uint8)
    if arr.dtype == np.uint16:
        return np.asarray(np.clip(np.rint(work), 0.0, 65535.0), dtype=np.uint16)
    if np.issubdtype(arr.dtype, np.integer):
        info = np.iinfo(arr.dtype)
        return np.asarray(np.clip(np.rint(work), float(info.min), float(info.max)), dtype=arr.dtype)
    return work.astype(arr.dtype, copy=False)


def _phase_marker_frame(
    frame_shape: tuple[int, int],
    actual_roi: dict | None,
    requested_roi: tuple[int, int, int, int] | None,
    dtype: np.dtype,
) -> np.ndarray | None:
    if len(frame_shape) != 2:
        return None
    h, w = int(frame_shape[0]), int(frame_shape[1])
    if h <= 0 or w <= 0:
        return None
    x0 = None
    y0 = None
    if isinstance(actual_roi, dict):
        x0 = actual_roi.get("OffsetX", actual_roi.get("x"))
        y0 = actual_roi.get("OffsetY", actual_roi.get("y"))
    if (x0 is None or y0 is None) and requested_roi is not None:
        try:
            x0 = requested_roi[0]
            y0 = requested_roi[1]
        except Exception:
            x0 = None
            y0 = None
    if x0 is None or y0 is None:
        return None
    try:
        px = int(round(float(x0))) % 2
        py = int(round(float(y0))) % 2
    except Exception:
        return None
    marker = np.zeros((h, w), dtype=dtype)
    marker[min(py, h - 1), min(px, w - 1)] = 1
    return marker


def fetch_frames(
    out_dir: Path,
    n_frames: int = 100,
    stop_after: int | None = 50,
    fps: float | None = None,
    exp_ms: float | None = None,
    gain: float | None = None,
    gain_analog: float | None = None,
    gain_digital: float | None = None,
    roi: tuple[int, int, int, int] | None = None,
    stop_flag: Path | None = None,
    out_path: Path | None = None,
    preview_path: Path | None = None,
    preview_every: int | None = None,
    subtract_background: bool = False,
) -> tuple[Path, float | None, int, dict | None]:
    app = QApplication.instance() or QApplication([])
    controller = Controller()
    background_profile = None
    background_profile_path = None
    if subtract_background:
        background_profile, background_profile_path = _load_background_profile()

    collected: list[np.ndarray] = []
    done = False
    actual_fps: float | None = None
    actual_roi: dict | None = None
    actual_timing: dict | None = None
    actual_gains: dict | None = None
    first_frame_seen = False
    max_raw_value: int | None = None
    capture_error: str | None = None
    dropped_first_frame = False

    def _timing_matches(snapshot: dict | None) -> bool:
        if not isinstance(snapshot, dict):
            return False
        if exp_ms is not None:
            exp_us = snapshot.get("exposure_us")
            if exp_us is None or abs(float(exp_us) - (1000.0 * float(exp_ms))) > 1.0:
                return False
        if fps is not None:
            snap_fps = snapshot.get("fps")
            if snap_fps is None:
                snap_fps = snapshot.get("resulting_fps")
            if snap_fps is None:
                return False
            snap_fps = float(snap_fps)
            req_fps = float(fps)
            tol = 2.0
            if abs(snap_fps - req_fps) <= tol:
                return True
            fps_max = snapshot.get("fps_max")
            if fps_max is not None:
                try:
                    fps_max_f = float(fps_max)
                except Exception:
                    fps_max_f = None
                if fps_max_f is not None:
                    # Accept the camera snapping to its hardware ceiling when the
                    # requested FPS exceeds what the current readout mode can do.
                    if req_fps >= (fps_max_f - tol) and abs(snap_fps - fps_max_f) <= tol:
                        return True
            return False
        return True

    preview_every_n = int(preview_every) if preview_every is not None else None
    if preview_every_n is not None and preview_every_n < 1:
        preview_every_n = 1

    def _write_preview(frame_native: np.ndarray) -> None:
        if preview_path is None:
            return
        try:
            tmp = preview_path.with_suffix(preview_path.suffix + ".tmp")
            with tmp.open("wb") as f:
                np.save(f, frame_native)
            tmp.replace(preview_path)
        except Exception:
            return

    def _on_frame(arr_obj: object) -> None:
        nonlocal done, first_frame_seen, max_raw_value, capture_error, dropped_first_frame
        if done:
            return
        frame_native = np.asarray(arr_obj, copy=True)
        if not dropped_first_frame:
            dropped_first_frame = True
            first_frame_seen = True
            return
        try:
            cur_max = int(np.max(frame_native))
            max_raw_value = cur_max if max_raw_value is None else max(max_raw_value, cur_max)
        except Exception:
            pass
        frame_native = _subtract_background(
            frame_native,
            profile=background_profile,
            actual_roi=actual_roi,
            requested_roi=roi,
        )
        frame_saved = _as_saved_frame(frame_native)
        collected.append(frame_saved)
        if preview_every_n is not None and (len(collected) % preview_every_n) == 0:
            _write_preview(frame_saved)
        first_frame_seen = True
        if stop_after is not None and len(collected) >= stop_after:
            done = True

    def _on_timing(payload: object) -> None:
        nonlocal actual_fps, actual_timing
        try:
            d = dict(payload or {})
            actual_timing = d
            rf = d.get("resulting_fps")
            if rf is None:
                rf = d.get("fps")
            if rf is not None:
                actual_fps = float(rf)
        except Exception:
            return

    def _on_roi(payload: object) -> None:
        nonlocal actual_roi
        try:
            actual_roi = dict(payload or {})
        except Exception:
            return

    def _on_gains(payload: object) -> None:
        nonlocal actual_gains
        try:
            actual_gains = dict(payload or {})
        except Exception:
            return

    try:
        controller.open()
        controller.cam.roi.connect(_on_roi)
        controller.cam.gains.connect(_on_gains)
        controller.cam.timing.connect(_on_timing)
        if roi is None:
            controller.full_sensor()
        else:
            x, y, w, h = roi
            controller.set_roi(float(w), float(h), float(x), float(y))
        try:
            controller.refresh_roi()
        except Exception:
            pass
        use_gain_analog = gain_analog if gain_analog is not None else gain
        use_gain_digital = gain_digital
        controller.set_timing(fps, exp_ms)
        if use_gain_analog is not None or use_gain_digital is not None:
            try:
                ga = float(use_gain_analog) if use_gain_analog is not None else None
                gd = float(use_gain_digital) if use_gain_digital is not None else None
                controller.set_gains(ga, gd)
            except Exception:
                pass
        controller.cam.frame.connect(_on_frame)
        try:
            controller.refresh_timing()
        except Exception:
            pass
        try:
            controller.refresh_gains()
        except Exception:
            pass
        # Give the controller a short chance to publish timing/gain snapshots, but
        # do not refuse capture if the camera reports a snapped/clamped value.
        # The saved JSON still records the actual timing so analysis can inspect it.
        t_cfg = __import__("time").time()
        while ((__import__("time").time() - t_cfg) < 0.75):
            app.processEvents()
            sleep(0.005)
            if actual_timing is not None or (exp_ms is None and fps is None):
                break
            try:
                controller.refresh_timing()
            except Exception:
                pass
            try:
                controller.refresh_gains()
            except Exception:
                pass
        controller.start()
        try:
            controller.refresh_timing()
        except Exception:
            pass
        try:
            controller.refresh_gains()
        except Exception:
            pass

        t0 = __import__("time").time()
        while actual_roi is None and (__import__("time").time() - t0) < 0.5:
            app.processEvents()
            sleep(0.01)

        # Wait briefly for first frame, retry start once if needed.
        t_first = __import__("time").time()
        while not first_frame_seen and (__import__("time").time() - t_first) < 0.75:
            app.processEvents()
            sleep(0.002)
        if not first_frame_seen:
            try:
                controller.stop()
            except Exception:
                pass
            controller.start()
            t_retry = __import__("time").time()
            while not first_frame_seen and (__import__("time").time() - t_retry) < 1.25:
                app.processEvents()
                sleep(0.002)

        while not done:
            app.processEvents()
            sleep(0.002)
            if stop_flag is not None and stop_flag.exists():
                done = True
    finally:
        try:
            controller.cam.frame.disconnect(_on_frame)
        except Exception:
            pass
        try:
            controller.cam.timing.disconnect(_on_timing)
        except Exception:
            pass
        try:
            controller.cam.gains.disconnect(_on_gains)
        except Exception:
            pass
        try:
            controller.cam.roi.disconnect(_on_roi)
        except Exception:
            pass
        try:
            controller.stop()
        except Exception:
            pass
        try:
            controller.close()
        except Exception:
            pass

    if not collected:
        if capture_error:
            raise RuntimeError(capture_error)
        raise RuntimeError("No frames captured.")
    if capture_error:
        raise RuntimeError(capture_error)

    out_dir.mkdir(parents=True, exist_ok=True)
    ts = strftime("%Y%m%d-%H%M%S")
    if out_path is None:
        out_path = out_dir / f"frame_stack_{ts}.npy"
    stack = np.stack(collected[:n_frames], axis=0)
    frame_count_saved = int(stack.shape[0]) if stack.ndim >= 1 else 0
    marker = _phase_marker_frame(
        frame_shape=(int(stack.shape[1]), int(stack.shape[2])) if stack.ndim >= 3 else (),
        actual_roi=actual_roi,
        requested_roi=roi,
        dtype=stack.dtype,
    )
    if marker is not None and stack.ndim == 3:
        stack = np.concatenate([stack, marker[np.newaxis, ...]], axis=0)
    np.save(out_path, stack)
    return out_path, actual_fps, int(frame_count_saved), {
        "roi": actual_roi,
        "timing": actual_timing,
        "gains": actual_gains,
        "max_raw_value": max_raw_value,
        "max_saved_value": (int(np.max(stack)) if stack.size else None),
        "phase_marker_appended": bool(marker is not None),
        "background_profile_path": (str(background_profile_path) if background_profile_path is not None else None),
        "background_subtracted": bool(subtract_background and (background_profile is not None)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch a short burst of frames.")
    parser.add_argument("--out-dir", default="frame_runs", help="Output directory")
    parser.add_argument("--n-frames", type=int, default=100, help="Max frames in stack")
    parser.add_argument("--stop-after", type=int, default=None, help="Stop acquisition after N frames")
    parser.add_argument("--fps", type=float, default=None, help="Requested frame rate")
    parser.add_argument("--exp-ms", type=float, default=None, help="Exposure time (ms)")
    parser.add_argument("--gain", type=float, default=None, help="Requested analog gain (legacy)")
    parser.add_argument("--gain-analog", type=float, default=None, help="Requested analog gain")
    parser.add_argument("--gain-digital", type=float, default=None, help="Requested digital gain")
    parser.add_argument("--roi", type=int, nargs=4, default=None, metavar=("X", "Y", "W", "H"))
    parser.add_argument("--stop-flag", default=None, help="Path to a stop-flag file")
    parser.add_argument("--out-path", default=None, help="Exact output .npy path")
    parser.add_argument("--preview-path", default=None, help="Path to write a single-frame preview .npy")
    parser.add_argument("--preview-every", type=int, default=None, help="Write preview every N frames")
    parser.add_argument("--subtract-background", action="store_true", help="Subtract stored background profile")
    parser.add_argument("--json", action="store_true", help="Emit JSON with path and actual_fps")
    args = parser.parse_args()

    out_path, actual_fps, count, actual_info = fetch_frames(
        Path(args.out_dir),
        n_frames=max(1, int(args.n_frames)),
        stop_after=(max(1, int(args.stop_after)) if args.stop_after is not None else None),
        fps=args.fps,
        exp_ms=args.exp_ms,
        gain=args.gain,
        gain_analog=args.gain_analog,
        gain_digital=args.gain_digital,
        roi=tuple(args.roi) if args.roi else None,
        stop_flag=Path(args.stop_flag) if args.stop_flag else None,
        out_path=Path(args.out_path) if args.out_path else None,
        preview_path=Path(args.preview_path) if args.preview_path else None,
        preview_every=args.preview_every,
        subtract_background=bool(args.subtract_background),
    )
    if args.json:
        print(json.dumps({
            "path": str(out_path),
            "actual_fps": actual_fps,
            "count": count,
            "roi": (actual_info or {}).get("roi"),
            "timing": (actual_info or {}).get("timing"),
            "gains": (actual_info or {}).get("gains"),
            "max_raw_value": (actual_info or {}).get("max_raw_value"),
            "max_saved_value": (actual_info or {}).get("max_saved_value"),
            "phase_marker_appended": (actual_info or {}).get("phase_marker_appended"),
            "background_profile_path": (actual_info or {}).get("background_profile_path"),
            "background_subtracted": (actual_info or {}).get("background_subtracted"),
        }))
    else:
        print(str(out_path))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
