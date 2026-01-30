from __future__ import annotations

"""
Minimal Qt-facing camera interface used by IDSCamera.

This defines the signals that the GUI and controller expect to exist on any
camera backend. IDSCamera implements the actual behavior in ids_backend.py.
"""

from PySide6.QtCore import QObject, Signal


class ICamera(QObject):
    # Lifecycle + data
    opened = Signal(str)
    started = Signal()
    stopped = Signal()
    closed = Signal()
    frame = Signal(object)  # numpy ndarray (H, W) uint16
    error = Signal(str)

    # State snapshots
    roi = Signal(dict)      # {Width, Height, OffsetX, OffsetY}
    timing = Signal(dict)   # {fps, resulting_fps, exposure_us, ...}
    gains = Signal(dict)    # {'analog':{...}, 'digital':{...}}

    # Auto-desaturation feedback
    desaturated = Signal(dict)
    auto_desat_started = Signal()
    auto_desat_finished = Signal()

    def __init__(self) -> None:
        super().__init__()
