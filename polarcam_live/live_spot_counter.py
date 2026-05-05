from __future__ import annotations

import queue
import time
import tkinter as tk
from tkinter import ttk, messagebox

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageTk


class LiveSpotCounterApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Live Spot Counter")

        self._live_running = False
        self._live_controller = None
        self._live_app = None
        self._live_queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=2)
        self._live_after_id = None
        self._live_img_ref = None
        self._live_last_frame = None

        self._overlay_centers: list[tuple[float, float]] = []
        self._overlay_until = 0.0

        self._exp_ms_var = tk.StringVar(value="0.05")
        self._gain_var = tk.StringVar(value="20")
        self._thr_var = tk.StringVar(value="180")
        self._min_area_var = tk.StringVar(value="6")
        self._max_area_var = tk.StringVar(value="300")

        self._status_var = tk.StringVar(value="Live feed stopped")
        self._last_count_var = tk.StringVar(value="Last count: -")
        self._fields_var = tk.StringVar(value="Fields counted: 0")
        self._total_var = tk.StringVar(value="Total spots: 0")
        self._mean_var = tk.StringVar(value="Mean spots/field: 0.00")

        self._fields_counted = 0
        self._total_spots = 0

        self._img_label = None
        self._start_btn = None
        self._stop_btn = None
        self._build_ui()

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def _build_ui(self) -> None:
        top = ttk.Frame(self.root, padding=8)
        top.pack(side=tk.TOP, fill=tk.X)

        self._start_btn = ttk.Button(top, text="Start live feed", command=self._start_live_feed)
        self._start_btn.pack(side=tk.LEFT)
        self._stop_btn = ttk.Button(top, text="Stop live feed", command=self._stop_live_feed)
        self._stop_btn.pack(side=tk.LEFT, padx=(6, 0))
        self._stop_btn.state(["disabled"])

        ttk.Label(top, text="Exposure ms").pack(side=tk.LEFT, padx=(16, 4))
        ttk.Entry(top, textvariable=self._exp_ms_var, width=7).pack(side=tk.LEFT)
        ttk.Label(top, text="Gain").pack(side=tk.LEFT, padx=(8, 4))
        ttk.Entry(top, textvariable=self._gain_var, width=7).pack(side=tk.LEFT)
        ttk.Button(top, text="Apply", command=self._apply_live_settings).pack(side=tk.LEFT, padx=(6, 0))

        params = ttk.Frame(self.root, padding=(8, 0, 8, 8))
        params.pack(side=tk.TOP, fill=tk.X)

        ttk.Label(params, text="Intensity threshold (0-255)").pack(side=tk.LEFT)
        ttk.Entry(params, textvariable=self._thr_var, width=6).pack(side=tk.LEFT, padx=(4, 10))
        ttk.Label(params, text="Min area").pack(side=tk.LEFT)
        ttk.Entry(params, textvariable=self._min_area_var, width=6).pack(side=tk.LEFT, padx=(4, 10))
        ttk.Label(params, text="Max area").pack(side=tk.LEFT)
        ttk.Entry(params, textvariable=self._max_area_var, width=6).pack(side=tk.LEFT, padx=(4, 12))

        ttk.Button(params, text="Count", command=self._count_current_field).pack(side=tk.LEFT)
        ttk.Button(params, text="Reset totals", command=self._reset_totals).pack(side=tk.LEFT, padx=(6, 0))
        ttk.Label(params, textvariable=self._status_var).pack(side=tk.RIGHT)

        stats = ttk.Frame(self.root, padding=(8, 0, 8, 8))
        stats.pack(side=tk.TOP, fill=tk.X)
        ttk.Label(stats, textvariable=self._last_count_var).pack(side=tk.LEFT, padx=(0, 16))
        ttk.Label(stats, textvariable=self._fields_var).pack(side=tk.LEFT, padx=(0, 16))
        ttk.Label(stats, textvariable=self._total_var).pack(side=tk.LEFT, padx=(0, 16))
        ttk.Label(stats, textvariable=self._mean_var).pack(side=tk.LEFT)

        view = ttk.Frame(self.root, padding=8)
        view.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self._img_label = tk.Label(view, bg="black")
        self._img_label.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    @staticmethod
    def _parse_float(text: str) -> float | None:
        try:
            return float(text)
        except Exception:
            return None

    @staticmethod
    def _parse_int(text: str) -> int | None:
        try:
            return int(text)
        except Exception:
            return None

    def _live_on_frame(self, arr_obj: object) -> None:
        if not self._live_running:
            return
        try:
            frame16 = np.asarray(arr_obj, dtype=np.uint16, copy=False)
            frame8 = (frame16 >> 4).astype(np.uint8, copy=False)
        except Exception:
            return
        self._live_last_frame = frame8
        try:
            self._live_queue.put_nowait(frame8)
        except queue.Full:
            try:
                self._live_queue.get_nowait()
            except queue.Empty:
                return
            try:
                self._live_queue.put_nowait(frame8)
            except queue.Full:
                pass

    def _apply_live_settings(self) -> None:
        if not self._live_controller:
            return
        exp_ms = self._parse_float(self._exp_ms_var.get())
        gain = self._parse_float(self._gain_var.get())
        try:
            if exp_ms is not None and exp_ms > 0.0:
                self._live_controller.set_timing(20.0, float(exp_ms))
            if gain is not None:
                self._live_controller.set_gains(float(gain), None)
            self._status_var.set("Live settings applied")
        except Exception as e:
            self._status_var.set(f"Apply failed: {e}")

    def _start_live_feed(self) -> None:
        if self._live_running:
            return
        try:
            from PySide6.QtWidgets import QApplication
            from Controlling.controller.controller import Controller
        except Exception as e:
            messagebox.showerror("Live feed", f"Could not start live feed: {e}")
            return

        exp_ms = self._parse_float(self._exp_ms_var.get())
        gain = self._parse_float(self._gain_var.get())
        if exp_ms is None or exp_ms <= 0.0:
            messagebox.showerror("Live feed", "Exposure time must be > 0 ms.")
            return
        if gain is None:
            gain = 0.0

        self._live_app = QApplication.instance() or QApplication([])
        self._live_queue = queue.Queue(maxsize=2)
        self._live_controller = Controller()
        try:
            self._live_controller.open()
            self._live_controller.full_sensor()
            self._live_controller.set_timing(20.0, float(exp_ms))
            self._live_controller.set_gains(float(gain), None)
            self._live_controller.start()
            self._live_controller.cam.frame.connect(self._live_on_frame)
        except Exception as e:
            try:
                self._live_controller.close()
            except Exception:
                pass
            self._live_controller = None
            messagebox.showerror("Live feed", f"Could not start live feed: {e}")
            return

        self._live_running = True
        self._start_btn.state(["disabled"])
        self._stop_btn.state(["!disabled"])
        self._status_var.set("Live feed running (20 fps)")
        self._live_tick()

    def _stop_live_feed(self) -> None:
        self._live_running = False
        if self._live_after_id is not None:
            try:
                self.root.after_cancel(self._live_after_id)
            except Exception:
                pass
            self._live_after_id = None
        if self._live_controller is not None:
            try:
                self._live_controller.cam.frame.disconnect(self._live_on_frame)
            except Exception:
                pass
            try:
                self._live_controller.stop()
            except Exception:
                pass
            try:
                self._live_controller.close()
            except Exception:
                pass
            self._live_controller = None
        self._start_btn.state(["!disabled"])
        self._stop_btn.state(["disabled"])
        self._status_var.set("Live feed stopped")

    def _detect_spots(self, frame8: np.ndarray) -> tuple[int, list[tuple[float, float]]]:
        thr = self._parse_int(self._thr_var.get())
        min_area = self._parse_int(self._min_area_var.get())
        max_area = self._parse_int(self._max_area_var.get())
        if thr is None:
            thr = 180
        if min_area is None or min_area < 1:
            min_area = 1
        if max_area is None or max_area < min_area:
            max_area = 10_000_000
        thr = max(0, min(255, int(thr)))

        blur = cv2.GaussianBlur(frame8, (0, 0), sigmaX=1.2, sigmaY=1.2, borderType=cv2.BORDER_REPLICATE)
        _, bw = cv2.threshold(blur, thr, 255, cv2.THRESH_BINARY)
        bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, np.ones((3, 3), dtype=np.uint8), iterations=1)

        n, _labels, stats, cent = cv2.connectedComponentsWithStats(bw, connectivity=8)
        centers: list[tuple[float, float]] = []
        for i in range(1, int(n)):
            area = int(stats[i, cv2.CC_STAT_AREA])
            if area < min_area or area > max_area:
                continue
            cx = float(cent[i, 0])
            cy = float(cent[i, 1])
            centers.append((cx, cy))

        return len(centers), centers

    def _count_current_field(self) -> None:
        if self._live_last_frame is None:
            self._status_var.set("No live frame to count")
            return
        count, centers = self._detect_spots(self._live_last_frame)
        self._fields_counted += 1
        self._total_spots += int(count)
        mean = float(self._total_spots) / float(max(1, self._fields_counted))

        self._last_count_var.set(f"Last count: {count}")
        self._fields_var.set(f"Fields counted: {self._fields_counted}")
        self._total_var.set(f"Total spots: {self._total_spots}")
        self._mean_var.set(f"Mean spots/field: {mean:.2f}")
        self._status_var.set(f"Counted {count} spots on current field")

        self._overlay_centers = centers
        self._overlay_until = time.time() + 1.5

    def _reset_totals(self) -> None:
        self._fields_counted = 0
        self._total_spots = 0
        self._last_count_var.set("Last count: -")
        self._fields_var.set("Fields counted: 0")
        self._total_var.set("Total spots: 0")
        self._mean_var.set("Mean spots/field: 0.00")
        self._status_var.set("Totals reset")
        self._overlay_centers = []
        self._overlay_until = 0.0

    def _live_tick(self) -> None:
        if not self._live_running:
            return
        try:
            if self._live_app is not None:
                self._live_app.processEvents()
        except Exception:
            pass

        frame = None
        try:
            while True:
                frame = self._live_queue.get_nowait()
        except queue.Empty:
            pass

        if frame is not None and self._img_label is not None:
            img = Image.fromarray(frame)
            try:
                resample = Image.Resampling.BILINEAR
            except Exception:
                resample = Image.BILINEAR

            w = int(self._img_label.winfo_width())
            h = int(self._img_label.winfo_height())
            if w > 10 and h > 10:
                src_w, src_h = img.size
                scale = min(float(w) / float(src_w), float(h) / float(src_h))
                disp_w = max(1, int(round(src_w * scale)))
                disp_h = max(1, int(round(src_h * scale)))
                img = img.resize((disp_w, disp_h), resample=resample).convert("RGB")

                if time.time() <= float(self._overlay_until) and self._overlay_centers:
                    draw = ImageDraw.Draw(img)
                    for cx, cy in self._overlay_centers:
                        px = float(cx) * scale
                        py = float(cy) * scale
                        r = 8
                        draw.ellipse([px - r, py - r, px + r, py + r], outline=(0, 255, 0), width=2)

                canvas = Image.new("RGB", (w, h), (0, 0, 0))
                off_x = max(0, (w - disp_w) // 2)
                off_y = max(0, (h - disp_h) // 2)
                canvas.paste(img, (off_x, off_y))
                img = canvas

            photo = ImageTk.PhotoImage(img)
            self._img_label.configure(image=photo)
            self._live_img_ref = photo

        self._live_after_id = self.root.after(50, self._live_tick)

    def on_close(self) -> None:
        self._stop_live_feed()
        self.root.destroy()


def main() -> None:
    root = tk.Tk()
    app = LiveSpotCounterApp(root)
    root.minsize(900, 600)
    root.mainloop()


if __name__ == "__main__":
    main()
