from __future__ import annotations

from typing import Iterator, Optional
import numpy as np
from frame_source import FrameSource


# OpenCV import lives here (so pol_reconstruction stays cv2-free)
import cv2


class AviSource(FrameSource):
    """
    FrameSource implementation for AVI files.

    - Reads frames via cv2.VideoCapture
    - Converts color -> grayscale (optional)
    - Ensures frames are 2D uint8 arrays by default
    - Checks for shape consistency
    """

    def __init__(self, path: str, *, force_gray: bool = True):
        self.path = path
        self.force_gray = force_gray

        cap = cv2.VideoCapture(self.path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Could not open video: {self.path}")

        ok, frame = cap.read()
        cap.release()
        if not ok or frame is None:
            raise ValueError(f"Could not read first frame from: {self.path}")

        frame2d = self._to_2d_uint8(frame)
        self.shape = (int(frame2d.shape[0]), int(frame2d.shape[1]))
        self.dtype = np.dtype(np.uint8)

    def _to_2d_uint8(self, frame: np.ndarray) -> np.ndarray:
        # frame is usually HxWx3 (BGR) or HxW (grayscale)
        if frame.ndim == 3:
            if self.force_gray:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                frame = frame[:, :, 0]  # take one channel

        if frame.ndim != 2:
            raise ValueError(f"Expected 2D frame after conversion, got shape {frame.shape}")

        if frame.dtype != np.uint8:
            frame = frame.astype(np.uint8, copy=False)

        return frame

    def frames(self) -> Iterator[np.ndarray]:
        cap = cv2.VideoCapture(self.path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Could not open video: {self.path}")

        try:
            while True:
                ok, frame = cap.read()
                if not ok or frame is None:
                    break

                frame2d = self._to_2d_uint8(frame)

                if frame2d.shape != self.shape:
                    raise ValueError(
                        f"Frame shape changed. Expected {self.shape}, got {frame2d.shape}."
                    )

                yield frame2d

        finally:
            cap.release()
