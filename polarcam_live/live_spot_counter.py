from __future__ import annotations

import queue
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, ttk, messagebox

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageTk


class LiveSpotCounterApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Live Spot Counter")
        self.root.geometry("1100x720")
        self.root.minsize(1100, 720)

        self._live_running = False
        self._live_controller = None
        self._live_app = None
        self._live_queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=2)
        self._live_after_id = None
        self._live_img_ref = None
        self._live_last_frame = None
        self._display_frame = None
        self._offline_mean_frames: list[tuple[Path, np.ndarray]] = []
        self._offline_idx = 0

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
        self._offline_var = tk.StringVar(value="Offline recordings loaded: 0")
        self._stretch_enabled_var = tk.BooleanVar(value=False)
        self._stretch_lo_var = tk.IntVar(value=0)
        self._stretch_hi_var = tk.IntVar(value=255)

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
        ttk.Button(params, text="Load offline .npy", command=self._load_offline_npy).pack(side=tk.LEFT, padx=(12, 0))
        ttk.Button(params, text="Load many offline .npy", command=self._load_many_offline_npy).pack(side=tk.LEFT, padx=(6, 0))
        ttk.Button(params, text="Count offline batch", command=self._count_offline_batch).pack(side=tk.LEFT, padx=(6, 0))
        ttk.Button(params, text="Reset totals", command=self._reset_totals).pack(side=tk.LEFT, padx=(6, 0))
        ttk.Label(params, textvariable=self._status_var).pack(side=tk.RIGHT)

        stats = ttk.Frame(self.root, padding=(8, 0, 8, 8))
        stats.pack(side=tk.TOP, fill=tk.X)
        ttk.Label(stats, textvariable=self._last_count_var).pack(side=tk.LEFT, padx=(0, 16))
        ttk.Label(stats, textvariable=self._fields_var).pack(side=tk.LEFT, padx=(0, 16))
        ttk.Label(stats, textvariable=self._total_var).pack(side=tk.LEFT, padx=(0, 16))
        ttk.Label(stats, textvariable=self._mean_var).pack(side=tk.LEFT)
        ttk.Label(stats, textvariable=self._offline_var).pack(side=tk.LEFT, padx=(16, 0))

        view = ttk.Frame(self.root, padding=8)
        view.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        left = ttk.Frame(view)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        left.pack_propagate(False)
        right = ttk.LabelFrame(view, text="Display grayscale stretch", padding=(8, 6, 8, 8))
        right.pack(side=tk.LEFT, fill=tk.Y, padx=(8, 0))
        self._img_label = tk.Label(left, bg="black")
        self._img_label.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        ttk.Checkbutton(
            right,
            text="Enable stretch",
            variable=self._stretch_enabled_var,
            command=self._refresh_display_frame,
        ).pack(side=tk.TOP, anchor="w")
        ttk.Label(right, text="Low").pack(side=tk.TOP, anchor="w", pady=(8, 2))
        lo_scale = tk.Scale(
            right,
            from_=0,
            to=255,
            orient=tk.HORIZONTAL,
            variable=self._stretch_lo_var,
            command=lambda _value: self._on_stretch_change(),
            length=180,
        )
        lo_scale.pack(side=tk.TOP, anchor="w")
        ttk.Label(right, text="High").pack(side=tk.TOP, anchor="w", pady=(8, 2))
        hi_scale = tk.Scale(
            right,
            from_=0,
            to=255,
            orient=tk.HORIZONTAL,
            variable=self._stretch_hi_var,
            command=lambda _value: self._on_stretch_change(),
            length=180,
        )
        hi_scale.pack(side=tk.TOP, anchor="w")

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
        self._display_frame = frame8
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
        frame = self._display_frame if self._display_frame is not None else self._live_last_frame
        if frame is None:
            self._status_var.set("No frame to count")
            return
        count, centers = self._detect_spots(frame)
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
        self._refresh_display_frame()

    def _mean_frame_from_npy(self, path: Path) -> np.ndarray:
        arr = np.load(path, allow_pickle=False)
        arr = np.asarray(arr)
        if arr.ndim == 2:
            mean_frame = arr.astype(np.float32, copy=False)
        elif arr.ndim == 3:
            # Keep only the average frame in memory; discard the stack immediately.
            mean_frame = np.mean(arr.astype(np.float32, copy=False), axis=0)
        else:
            raise ValueError(f"Expected 2D frame or 3D stack, got shape {arr.shape}")
        if mean_frame.dtype != np.uint8:
            if np.issubdtype(mean_frame.dtype, np.floating):
                max_val = float(np.max(mean_frame)) if mean_frame.size else 0.0
            else:
                max_val = float(np.iinfo(mean_frame.dtype).max) if mean_frame.size else 0.0
            if max_val > 255.0:
                mean_frame = np.clip(np.rint(mean_frame / 16.0), 0, 255).astype(np.uint8)
            else:
                mean_frame = np.clip(np.rint(mean_frame), 0, 255).astype(np.uint8)
        return mean_frame

    def _load_offline_paths(self, paths: list[Path]) -> None:
        if not paths:
            return
        frames: list[tuple[Path, np.ndarray]] = []
        for path in paths:
            try:
                frames.append((path, self._mean_frame_from_npy(path)))
            except Exception as e:
                messagebox.showerror("Offline load", f"Could not load {path.name}: {e}")
                return
        self._offline_mean_frames = frames
        self._offline_idx = 0
        self._display_frame = frames[0][1]
        self._offline_var.set(f"Offline recordings loaded: {len(frames)}")
        self._status_var.set(f"Loaded offline mean frame(s) from {len(frames)} recording(s)")
        self._overlay_centers = []
        self._overlay_until = 0.0
        self._show_frame(self._display_frame)

    def _load_offline_npy(self) -> None:
        path = filedialog.askopenfilename(
            title="Open full-frame NumPy recording",
            filetypes=[("NumPy files", "*.npy"), ("All files", "*.*")],
        )
        if not path:
            return
        self._load_offline_paths([Path(path)])

    def _load_many_offline_npy(self) -> None:
        paths = filedialog.askopenfilenames(
            title="Open many full-frame NumPy recordings",
            filetypes=[("NumPy files", "*.npy"), ("All files", "*.*")],
        )
        if not paths:
            return
        self._load_offline_paths([Path(p) for p in paths])

    def _count_offline_batch(self) -> None:
        if not self._offline_mean_frames:
            self._status_var.set("No offline recordings loaded")
            return
        counts: list[int] = []
        last_centers: list[tuple[float, float]] = []
        for path, frame in self._offline_mean_frames:
            count, centers = self._detect_spots(frame)
            counts.append(int(count))
            last_centers = centers
        self._fields_counted = len(counts)
        self._total_spots = int(sum(counts))
        mean = float(np.mean(counts)) if counts else 0.0
        self._last_count_var.set(f"Last count: {counts[-1] if counts else '-'}")
        self._fields_var.set(f"Fields counted: {self._fields_counted}")
        self._total_var.set(f"Total spots: {self._total_spots}")
        self._mean_var.set(f"Mean spots/field: {mean:.2f}")
        self._status_var.set(f"Offline counts: {counts} | mean {mean:.2f}")
        self._display_frame = self._offline_mean_frames[-1][1]
        self._overlay_centers = last_centers
        self._overlay_until = time.time() + 2.0
        self._refresh_display_frame()

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
        self._refresh_display_frame()

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

        if frame is not None:
            self._display_frame = frame
            self._refresh_display_frame()

        self._live_after_id = self.root.after(50, self._live_tick)

    def _on_stretch_change(self) -> None:
        lo = int(self._stretch_lo_var.get())
        hi = int(self._stretch_hi_var.get())
        if lo > hi:
            self._stretch_hi_var.set(lo)
        self._refresh_display_frame()

    def _refresh_display_frame(self) -> None:
        if self._display_frame is not None:
            self._show_frame(self._display_frame)

    def _apply_display_stretch(self, frame: np.ndarray) -> np.ndarray:
        arr = np.asarray(frame, dtype=np.uint8)
        if not bool(self._stretch_enabled_var.get()):
            return arr
        lo = int(self._stretch_lo_var.get())
        hi = int(self._stretch_hi_var.get())
        if hi <= lo:
            return arr
        stretched = (arr.astype(np.float32) - float(lo)) * (255.0 / float(hi - lo))
        return np.clip(np.rint(stretched), 0, 255).astype(np.uint8)

    def _show_frame(self, frame: np.ndarray) -> None:
        if self._img_label is None:
            return
        img = Image.fromarray(self._apply_display_stretch(frame))
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

    def on_close(self) -> None:
        self._stop_live_feed()
        self.root.destroy()


def main() -> None:
    root = tk.Tk()
    app = LiveSpotCounterApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
