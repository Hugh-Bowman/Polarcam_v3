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


def _as_uint8(frame16: np.ndarray) -> np.ndarray:
    """Convert uint16 Mono12-style data to uint8 grayscale."""
    if frame16.dtype != np.uint16:
        frame16 = frame16.astype(np.uint16, copy=False)
    # Mono12 is 0..4095; shift by 4 to map to 0..255.
    return (frame16 >> 4).astype(np.uint8, copy=False)


def fetch_frames(
    out_dir: Path,
    n_frames: int = 100,
    stop_after: int | None = 50,
    fps: float | None = None,
    exp_ms: float | None = None,
    roi: tuple[int, int, int, int] | None = None,
    stop_flag: Path | None = None,
    out_path: Path | None = None,
) -> tuple[Path, float | None, int]:
    app = QApplication.instance() or QApplication([])
    controller = Controller()

    collected: list[np.ndarray] = []
    done = False
    actual_fps: float | None = None

    def _on_frame(arr_obj: object) -> None:
        nonlocal done
        if done:
            return
        frame16 = np.asarray(arr_obj, dtype=np.uint16, copy=True)
        frame8 = _as_uint8(frame16)
        collected.append(frame8)
        if stop_after is not None and len(collected) >= stop_after:
            done = True

    def _on_timing(payload: object) -> None:
        nonlocal actual_fps
        try:
            d = dict(payload or {})
            rf = d.get("resulting_fps")
            if rf is None:
                rf = d.get("fps")
            if rf is not None:
                actual_fps = float(rf)
        except Exception:
            return

    try:
        controller.open()
        if roi is None:
            controller.full_sensor()
        else:
            x, y, w, h = roi
            controller.set_roi(float(w), float(h), float(x), float(y))
        controller.set_timing(fps, exp_ms)
        controller.start()

        controller.cam.frame.connect(_on_frame)
        controller.cam.timing.connect(_on_timing)
        try:
            controller.refresh_timing()
        except Exception:
            pass

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
            controller.stop()
        except Exception:
            pass
        try:
            controller.close()
        except Exception:
            pass

    if not collected:
        raise RuntimeError("No frames captured.")

    out_dir.mkdir(parents=True, exist_ok=True)
    ts = strftime("%Y%m%d-%H%M%S")
    if out_path is None:
        out_path = out_dir / f"frame_stack_{ts}.npy"
    stack = np.stack(collected[:n_frames], axis=0)
    np.save(out_path, stack)
    return out_path, actual_fps, int(stack.shape[0])


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch a short burst of frames.")
    parser.add_argument("--out-dir", default="frame_runs", help="Output directory")
    parser.add_argument("--n-frames", type=int, default=100, help="Max frames in stack")
    parser.add_argument("--stop-after", type=int, default=None, help="Stop acquisition after N frames")
    parser.add_argument("--fps", type=float, default=None, help="Requested frame rate")
    parser.add_argument("--exp-ms", type=float, default=None, help="Exposure time (ms)")
    parser.add_argument("--roi", type=int, nargs=4, default=None, metavar=("X", "Y", "W", "H"))
    parser.add_argument("--stop-flag", default=None, help="Path to a stop-flag file")
    parser.add_argument("--out-path", default=None, help="Exact output .npy path")
    parser.add_argument("--json", action="store_true", help="Emit JSON with path and actual_fps")
    args = parser.parse_args()

    out_path, actual_fps, count = fetch_frames(
        Path(args.out_dir),
        n_frames=max(1, int(args.n_frames)),
        stop_after=(max(1, int(args.stop_after)) if args.stop_after is not None else None),
        fps=args.fps,
        exp_ms=args.exp_ms,
        roi=tuple(args.roi) if args.roi else None,
        stop_flag=Path(args.stop_flag) if args.stop_flag else None,
        out_path=Path(args.out_path) if args.out_path else None,
    )
    if args.json:
        print(json.dumps({"path": str(out_path), "actual_fps": actual_fps, "count": count}))
    else:
        print(str(out_path))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
