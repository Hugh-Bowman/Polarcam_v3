from __future__ import annotations

import argparse
import atexit
import ctypes
import random
import signal
import threading
import time
from ctypes import wintypes


user32 = ctypes.WinDLL("user32", use_last_error=True)
STOP_EVENT = threading.Event()


class POINT(ctypes.Structure):
    _fields_ = [("x", wintypes.LONG), ("y", wintypes.LONG)]


def get_cursor_pos() -> tuple[int, int]:
    pt = POINT()
    if not user32.GetCursorPos(ctypes.byref(pt)):
        raise ctypes.WinError(ctypes.get_last_error())
    return int(pt.x), int(pt.y)


def set_cursor_pos(x: int, y: int) -> None:
    if not user32.SetCursorPos(int(x), int(y)):
        raise ctypes.WinError(ctypes.get_last_error())


def jiggle_mouse(delta: int) -> None:
    x, y = get_cursor_pos()
    set_cursor_pos(x + delta, y)
    STOP_EVENT.wait(0.05)
    set_cursor_pos(x, y)


def request_stop(*_args: object) -> None:
    STOP_EVENT.set()


def main() -> None:
    parser = argparse.ArgumentParser(description="Jiggle the mouse slightly to keep the PC awake.")
    parser.add_argument("--min-interval", type=float, default=1.0, help="Minimum seconds between jiggles. Default: 1")
    parser.add_argument("--max-interval", type=float, default=20.0, help="Maximum seconds between jiggles. Default: 20")
    parser.add_argument("--min-delta", type=int, default=1, help="Minimum pixels to move the mouse. Default: 1")
    parser.add_argument("--max-delta", type=int, default=4, help="Maximum pixels to move the mouse. Default: 4")
    args = parser.parse_args()

    min_interval = max(0.1, float(args.min_interval))
    max_interval = max(min_interval, float(args.max_interval))
    min_delta = max(1, abs(int(args.min_delta)))
    max_delta = max(min_delta, abs(int(args.max_delta)))

    atexit.register(request_stop)
    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    print(
        "Mouse jiggle running: "
        f"interval={min_interval:g}-{max_interval:g}s "
        f"delta={min_delta}-{max_delta}px. Press Ctrl+C to stop."
    )
    try:
        while not STOP_EVENT.is_set():
            interval = random.uniform(min_interval, max_interval)
            if STOP_EVENT.wait(interval):
                break
            delta = random.randint(min_delta, max_delta)
            if random.choice((True, False)):
                delta = -delta
            jiggle_mouse(delta)
    except KeyboardInterrupt:
        request_stop()
    finally:
        print("Stopped.")


if __name__ == "__main__":
    main()
