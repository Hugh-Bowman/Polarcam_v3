from __future__ import annotations

import argparse
import queue
import sys
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk


def clamp_int(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, int(v)))


def odd_from_slider(v: int, minimum: int = 1) -> int:
    k = max(minimum, v)
    if k % 2 == 0:
        k += 1
    return k


class SpinnerStyleLiveSource:
    """
    Live frame source matching Spinners_gui_live behavior:
    - Uses Controller + PySide signal callback
    - callback converts uint16 -> uint8 via >>4
    - callback pushes into a tiny queue, replacing old frame when full
    - UI thread calls processEvents() and drains queue to newest frame
    """

    def __init__(self, fps: float, exposure_ms: float, gain: float) -> None:
        self.fps = float(fps)
        self.exposure_ms = float(exposure_ms)
        self.gain = float(gain)
        self.app = None
        self.controller = None
        self.queue: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=2)
        self.latest: Optional[np.ndarray] = None
        self.frame_seq = 0

    def _on_frame(self, arr_obj: object) -> None:
        try:
            frame16 = np.asarray(arr_obj, dtype=np.uint16)
            frame8 = (frame16 >> 4).astype(np.uint8, copy=False)
        except Exception:
            return
        self.latest = frame8
        self.frame_seq += 1
        try:
            self.queue.put_nowait(frame8)
        except queue.Full:
            try:
                _ = self.queue.get_nowait()
            except queue.Empty:
                return
            try:
                self.queue.put_nowait(frame8)
            except queue.Full:
                pass

    def start(self) -> None:
        # Ensure project root is importable so `Controlling` resolves when run from this subfolder.
        project_root = Path(__file__).resolve().parents[1]
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))

        from PySide6.QtWidgets import QApplication
        from Controlling.controller.controller import Controller

        self.app = QApplication.instance() or QApplication([])
        self.controller = Controller()
        self.controller.open()
        self.controller.full_sensor()
        self.controller.set_timing(self.fps, self.exposure_ms)
        self.controller.set_gains(self.gain, None)
        self.controller.start()
        self.controller.cam.frame.connect(self._on_frame)

    def stop(self) -> None:
        if self.controller is not None:
            try:
                self.controller.cam.frame.disconnect(self._on_frame)
            except Exception:
                pass
            try:
                self.controller.stop()
            except Exception:
                pass
            try:
                self.controller.close()
            except Exception:
                pass
        self.controller = None

    def pump_latest(self) -> Optional[np.ndarray]:
        if self.app is not None:
            try:
                self.app.processEvents()
            except Exception:
                pass
        newest = None
        try:
            while True:
                newest = self.queue.get_nowait()
        except queue.Empty:
            pass
        if newest is not None:
            self.latest = newest
        return self.latest

    def capture_single_frame(self, wait_timeout_s: float = 0.35) -> Optional[np.ndarray]:
        start_seq = self.frame_seq
        t0 = time.perf_counter()
        last = None
        while time.perf_counter() - t0 < wait_timeout_s:
            last = self.pump_latest()
            if self.frame_seq != start_seq and last is not None:
                return last
            time.sleep(0.005)
        return last

def linear_gain_preview(frame_u8: np.ndarray, gain: float = 20.0) -> np.ndarray:
    return np.clip(frame_u8.astype(np.float32) * gain, 0, 255).astype(np.uint8)


def build_mask(
    frame_u8: np.ndarray,
    prev_u8: Optional[np.ndarray],
    p: dict[str, int],
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, float]]:
    # Mask-only logic:
    # 1) detect candidate excluded pixels from intensity + temporal constraints
    # 2) expand/close
    # 3) always fill connected-component interiors (fills bright-spot centers)
    frame = frame_u8.astype(np.float32)

    if prev_u8 is None:
        temporal = np.zeros_like(frame, dtype=np.float32)
    else:
        temporal = cv2.absdiff(frame_u8, prev_u8).astype(np.float32)

    intensity_ok = (frame >= p["int_min"]) & (frame <= p["int_max"])
    temporal_ok = temporal <= p["temp_th"]

    base_mask = intensity_ok & temporal_ok

    # Raw detector output is treated as "pixels to EXCLUDE".
    raw_exclude = base_mask.astype(bool)
    exclude = raw_exclude.copy()

    # Smooth exclusion mask: expand to neighbors, close/fill gaps, optional enclosed-hole fill.
    expand_k = odd_from_slider(p["exclude_expand_k"], 1)
    fill_k = odd_from_slider(p["exclude_fill_k"], 1)
    fill_iter = max(1, int(p["exclude_fill_iter"]))
    hole_fill_on = int(p["exclude_hole_fill"]) > 0
    if expand_k > 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (expand_k, expand_k))
        exclude = cv2.dilate(exclude.astype(np.uint8), k, iterations=1).astype(bool)
    if fill_k > 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (fill_k, fill_k))
        exclude_u8 = cv2.morphologyEx(exclude.astype(np.uint8), cv2.MORPH_CLOSE, k, iterations=fill_iter)
        exclude = exclude_u8 > 0

    # Always fill interiors of excluded blobs so donut-like bright spots become fully masked.
    ex_u8 = exclude.astype(np.uint8)
    contours, _ = cv2.findContours(ex_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        filled = np.zeros_like(ex_u8)
        cv2.drawContours(filled, contours, contourIdx=-1, color=1, thickness=cv2.FILLED)
        exclude = filled > 0

    # Optional extra enclosed-hole fill stage for conservative masking.
    if hole_fill_on:
        ex_u8 = exclude.astype(np.uint8)
        inv = (1 - ex_u8).astype(np.uint8)
        ff = inv.copy()
        h, w = ff.shape
        flood_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
        cv2.floodFill(ff, flood_mask, (0, 0), 2)  # mark outside background
        holes = ff == 1
        exclude = (ex_u8 > 0) | holes

    # Keep only excluded components above min area; remove tiny speckles.
    min_area = p["min_area"]
    if min_area > 1:
        nlab, labels, stats, _ = cv2.connectedComponentsWithStats(exclude.astype(np.uint8), connectivity=8)
        keep_ex = np.zeros_like(exclude, dtype=bool)
        for lab in range(1, nlab):
            if stats[lab, cv2.CC_STAT_AREA] >= min_area:
                keep_ex[labels == lab] = True
        exclude = keep_ex

    # Final hard rule: exclude any pixel that is >50, and any of its 8-neighbours.
    bright = (frame_u8 > 30).astype(np.uint8)
    bright_neighbourhood = cv2.dilate(bright, np.ones((3, 3), dtype=np.uint8), iterations=1) > 0
    exclude = exclude | bright_neighbourhood

    include = ~exclude

    parts = {
        "intensity_ok": intensity_ok.astype(np.uint8),
        "temporal_ok": temporal_ok.astype(np.uint8),
        "exclude_raw": raw_exclude.astype(np.uint8),
        "exclude_final": exclude.astype(np.uint8),
        "include_final": include.astype(np.uint8),
    }
    metrics = {
        "include_fraction": float(np.mean(include)),
        "exclude_fraction": float(np.mean(exclude)),
        "temp_p90": float(np.percentile(temporal, 90)),
        "int_p10": float(np.percentile(frame, 10)),
        "int_p90": float(np.percentile(frame, 90)),
    }
    return include.astype(np.uint8), parts, metrics


class BackgroundFitRegionTkApp:
    def __init__(self, live: SpinnerStyleLiveSource, masks_out_dir: Path) -> None:
        self.live = live
        self.masks_out_dir = masks_out_dir
        self.masks_out_dir.mkdir(parents=True, exist_ok=True)

        self.p_ranges = {
            "int_min": (0, 255),
            "int_max": (0, 255),
            "temp_th": (0, 100),
            "exclude_expand_k": (0, 61),
            "exclude_fill_k": (0, 61),
            "exclude_fill_iter": (0, 8),
            "exclude_hole_fill": (0, 1),
            "min_area": (0, 5000),
            "font_scale_x10": (6, 30),
        }
        self.p = {
            "int_min": 2,
            "int_max": 255,
            "temp_th": 4,
            "exclude_expand_k": 5,
            "exclude_fill_k": 5,
            "exclude_fill_iter": 1,
            "exclude_hole_fill": 0,
            "min_area": 40,
            "font_scale_x10": 10,
        }
        self.param_order = list(self.p.keys())

        self.root = tk.Tk()
        self.root.title("Background Fit Region GUI")
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.root.geometry("1700x980")

        self.status_var = tk.StringVar(value="Ready. Capture (c), Process (p), Save (s).")
        self.frame_var = tk.StringVar(value="frame=0")
        self.fit_var = tk.StringVar(value="mask-only mode")
        self.coverage_var = tk.StringVar(value="include=0.0% exclude=0.0%")

        self.frame_idx = 0
        self.running = True

        self.current_frame: Optional[np.ndarray] = None
        self.prev_frame: Optional[np.ndarray] = None
        self.last_mask: Optional[np.ndarray] = None
        self.last_parts: Optional[dict[str, np.ndarray]] = None
        self.last_metrics: Optional[dict[str, float]] = None

        self.entry_vars: dict[str, tk.StringVar] = {}
        self._photo_main: Optional[ImageTk.PhotoImage] = None
        self._photo_overlay: Optional[ImageTk.PhotoImage] = None

        self._build_ui()
        self._bind_keys()
        self.root.after(30, self._tick)

    def _build_ui(self) -> None:
        style = ttk.Style()
        style.configure("TLabel", padding=2)
        style.configure("TButton", padding=4)

        outer = ttk.Frame(self.root, padding=8)
        outer.pack(fill=tk.BOTH, expand=True)
        outer.columnconfigure(0, weight=0)
        outer.columnconfigure(1, weight=1)
        outer.rowconfigure(0, weight=1)

        controls = ttk.LabelFrame(outer, text="Controls", padding=8)
        controls.grid(row=0, column=0, sticky="nsw", padx=(0, 8))

        row = 0
        ttk.Label(controls, text="Enter value and press Enter (manual process only)").grid(row=row, column=0, columnspan=2, sticky="w")
        row += 1
        for name in self.param_order:
            ttk.Label(controls, text=name).grid(row=row, column=0, sticky="w", padx=(0, 6), pady=2)
            var = tk.StringVar(value=str(self.p[name]))
            self.entry_vars[name] = var
            ent = ttk.Entry(controls, textvariable=var, width=10)
            ent.grid(row=row, column=1, sticky="w", pady=2)
            ent.bind("<Return>", self._on_param_enter)
            row += 1

        ttk.Separator(controls).grid(row=row, column=0, columnspan=2, sticky="ew", pady=6)
        row += 1
        ttk.Button(controls, text="Capture (c)", command=self.capture).grid(row=row, column=0, sticky="ew", pady=2)
        ttk.Button(controls, text="Process (p)", command=self.process).grid(row=row, column=1, sticky="ew", pady=2)
        row += 1
        ttk.Button(controls, text="Save Mask (s)", command=self.save_mask).grid(row=row, column=0, columnspan=2, sticky="ew", pady=2)
        row += 1
        ttk.Button(controls, text="Quit", command=self.on_close).grid(row=row, column=0, columnspan=2, sticky="ew", pady=6)
        row += 1

        ttk.Label(controls, text="Mask-only mode (no background fit)").grid(row=row, column=0, columnspan=2, sticky="w", pady=(8, 2))
        row += 1
        ttk.Label(controls, text=f"Mask dir: {self.masks_out_dir}").grid(row=row, column=0, columnspan=2, sticky="w")

        status = ttk.LabelFrame(controls, text="Status", padding=6)
        status.grid(row=row + 1, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        ttk.Label(status, textvariable=self.frame_var).grid(row=0, column=0, sticky="w")
        ttk.Label(status, textvariable=self.coverage_var).grid(row=1, column=0, sticky="w")
        ttk.Label(status, textvariable=self.fit_var).grid(row=2, column=0, sticky="w")
        ttk.Label(status, textvariable=self.status_var, wraplength=360).grid(row=3, column=0, sticky="w")

        previews = ttk.Frame(outer)
        previews.grid(row=0, column=1, sticky="nsew")
        previews.columnconfigure(0, weight=1)
        previews.rowconfigure(1, weight=1)
        previews.rowconfigure(3, weight=1)

        ttk.Label(previews, text="Captured Frame").grid(row=0, column=0, sticky="w")
        ttk.Label(previews, text="Mask Overlay (red=hidden)").grid(row=2, column=0, sticky="w", pady=(10, 0))

        self.lbl_main = ttk.Label(previews)
        self.lbl_main.grid(row=1, column=0, sticky="nsew", padx=2)
        self.lbl_overlay = ttk.Label(previews)
        self.lbl_overlay.grid(row=3, column=0, sticky="nsew", padx=2)

    def _bind_keys(self) -> None:
        self.root.bind("<KeyPress-c>", lambda _e: self.capture())
        self.root.bind("<KeyPress-p>", lambda _e: self.process())
        self.root.bind("<KeyPress-s>", lambda _e: self.save_mask())
        self.root.bind("<KeyPress-q>", lambda _e: self.on_close())
        self.root.bind("<Escape>", lambda _e: self.on_close())

    def _on_param_enter(self, _event: object) -> None:
        self.commit_params()

    def commit_params(self) -> bool:
        for name, var in self.entry_vars.items():
            raw = var.get().strip()
            try:
                value_i = int(float(raw))
            except ValueError:
                self.status_var.set(f"Invalid value for {name}: {raw}")
                return False
            lo, hi = self.p_ranges[name]
            self.p[name] = clamp_int(value_i, lo, hi)
            var.set(str(self.p[name]))
        if self.p["int_max"] < self.p["int_min"]:
            self.p["int_max"] = self.p["int_min"]
            self.entry_vars["int_max"].set(str(self.p["int_max"]))
        self.status_var.set("Parameters updated. Press Process (p) to recompute.")
        return True

    def capture(self) -> None:
        fr = self.live.capture_single_frame(wait_timeout_s=0.4)
        if fr is None:
            self.status_var.set("Capture failed (no frame).")
            return
        self.frame_idx += 1
        self.current_frame = fr
        self.last_mask = None
        self.last_parts = None
        self.last_metrics = None
        self.frame_var.set(f"frame={self.frame_idx}")
        self.status_var.set("Captured one frame. Press Process (p).")

    def process(self) -> None:
        if self.current_frame is None:
            self.status_var.set("No captured frame. Press Capture (c) first.")
            return
        if not self.commit_params():
            return
        m, prt, met = build_mask(self.current_frame, self.prev_frame, self.p)
        self.last_mask = m
        self.last_parts = prt
        self.last_metrics = met
        self.prev_frame = self.current_frame.copy()
        self._update_metrics_text()
        self.status_var.set("Processed current frame.")

    def save_mask(self) -> None:
        if self.last_parts is None:
            self.status_var.set("No processed mask to save. Capture and Process first.")
            return
        hidden_mask_u8 = (self.last_parts["exclude_final"] > 0).astype(np.uint8)
        out_base = f"mask_frame{self.frame_idx:05d}"
        out_mask_npy = self.masks_out_dir / f"{out_base}.npy"
        out_mask_png = self.masks_out_dir / f"{out_base}.png"
        out_overlay = self.masks_out_dir / f"{out_base}_overlay.png"

        np.save(out_mask_npy, hidden_mask_u8.astype(np.float32))
        cv2.imwrite(str(out_mask_png), (hidden_mask_u8 * 255).astype(np.uint8))

        overlay_img = self._build_overlay_panel(self.current_frame, self.last_parts)
        if overlay_img is not None:
            cv2.imwrite(str(out_overlay), overlay_img)

        self.status_var.set(f"Saved {out_base} to masks folder.")
        print(f"Saved mask npy: {out_mask_npy}")
        print(f"Saved mask png: {out_mask_png}")
        print(f"Saved overlay:  {out_overlay}")

    def _update_metrics_text(self) -> None:
        if self.last_metrics is None:
            self.coverage_var.set("include=0.0% exclude=0.0%")
            self.fit_var.set("mask-only mode")
            return
        m = self.last_metrics
        self.coverage_var.set(f"include={m['include_fraction']*100:.1f}% exclude={m['exclude_fraction']*100:.1f}%")
        self.fit_var.set(f"temp p90={m['temp_p90']:.2f} int p10/p90={m['int_p10']:.1f}/{m['int_p90']:.1f}")

    def _to_photo(self, bgr_img: np.ndarray) -> ImageTk.PhotoImage:
        rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        return ImageTk.PhotoImage(img)

    def _build_overlay_panel(self, frame: Optional[np.ndarray], parts: Optional[dict[str, np.ndarray]]) -> Optional[np.ndarray]:
        if frame is None:
            return None
        stretched = linear_gain_preview(frame, gain=20.0)
        gray_bgr = cv2.cvtColor(stretched, cv2.COLOR_GRAY2BGR)
        overlay = gray_bgr.copy()
        if parts is not None and "exclude_final" in parts:
            hidden = parts["exclude_final"] > 0
            overlay[hidden] = (0, 0, 255)
        return cv2.addWeighted(gray_bgr, 0.35, overlay, 0.65, 0)

    def _tick(self) -> None:
        if not self.running:
            return
        try:
            # Keep event pump active for PySide camera callback path.
            self.live.pump_latest()
            if self.current_frame is None and self.live.latest is not None:
                self.current_frame = self.live.latest.copy()

            if self.current_frame is None:
                frame = np.zeros((480, 640), dtype=np.uint8)
            else:
                frame = self.current_frame

            main_panel = cv2.cvtColor(linear_gain_preview(frame, gain=20.0), cv2.COLOR_GRAY2BGR)
            overlay_panel = self._build_overlay_panel(frame, self.last_parts)
            if overlay_panel is None:
                overlay_panel = np.zeros_like(main_panel)

            self._photo_main = self._to_photo(main_panel)
            self._photo_overlay = self._to_photo(overlay_panel)
            self.lbl_main.configure(image=self._photo_main)
            self.lbl_overlay.configure(image=self._photo_overlay)
        except Exception as exc:
            self.status_var.set(f"Update error: {exc}")
        finally:
            self.root.after(30, self._tick)

    def on_close(self) -> None:
        self.running = False
        try:
            self.live.stop()
        except Exception:
            pass
        self.root.destroy()

    def run(self) -> None:
        self.root.mainloop()


def main() -> None:
    parser = argparse.ArgumentParser(description="Live GUI for selecting usable background-fit regions.")
    parser.add_argument("--live-fps", type=float, default=20.0, help="Requested camera FPS setting.")
    parser.add_argument("--exposure-ms", type=float, default=0.02, help="Requested camera exposure in milliseconds.")
    parser.add_argument("--gain", type=float, default=0.0, help="Requested camera gain.")
    args = parser.parse_args()

    live = SpinnerStyleLiveSource(fps=args.live_fps, exposure_ms=args.exposure_ms, gain=args.gain)
    live.start()
    masks_out_dir = Path(__file__).resolve().parents[1] / "gpt prompted exploration" / "Background ml recon" / "data" / "masks"
    app = BackgroundFitRegionTkApp(live=live, masks_out_dir=masks_out_dir)
    app.run()


if __name__ == "__main__":
    main()
