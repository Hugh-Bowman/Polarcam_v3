from __future__ import annotations

import json
import math
import queue
import subprocess
import sys
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageTk

from Spinners_gui_live import BasicVideoPlayer, Figure, FigureCanvas


class ManualRotationApp(BasicVideoPlayer):
    PROTRACTOR_HIT_RADIUS_PX = 16.0

    def __init__(self, root: tk.Tk):
        self._protractor_point1: tuple[float, float] | None = None
        self._protractor_point2: tuple[float, float] | None = None
        self._protractor_pending_point = 1
        self._protractor_mode = "normal"
        self._protractor_drag_index: int | None = None
        self._protractor_track_var = tk.BooleanVar(master=root, value=True)
        self._protractor_status_var = tk.StringVar(master=root, value="Set point 1")
        self._protractor_angle_var = tk.StringVar(master=root, value="Angle from vertical: -")
        self._protractor_right_frame = None
        self._box_record_point1: tuple[float, float] | None = None
        self._box_record_point2: tuple[float, float] | None = None
        self._box_record_mode = "normal"
        self._box_record_dragging = False
        self._box_record_btn = None
        self._box_record_status_var = tk.StringVar(master=root, value="Box recording idle.")
        self._box_review_records: list[dict] = []
        self._box_review_current_index = -1
        self._box_review_img_label = None
        self._box_review_status_var = tk.StringVar(master=root, value="No box recordings loaded.")
        self._box_review_angle_var = tk.StringVar(master=root, value="0")
        self._box_review_photo = None
        self._box_review_disp_scale = 1.0
        self._box_review_disp_offset = (0, 0)
        self._rotation_plot_label = None
        self._rotation_plot_photo = None
        self._rotation_plot_status_var = tk.StringVar(master=root, value="Need at least two picked recordings.")
        super().__init__(root)
        self._live_capture_frames_var.set("1600")
        self.root.title("Manual Rotation")

    def _build_ui(self) -> None:
        notebook = ttk.Notebook(self.root)
        notebook.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self._notebook = notebook

        self._live_tab = ttk.Frame(notebook)
        notebook.add(self._live_tab, text="Manual rotation")
        self._box_review_tab = ttk.Frame(notebook)
        notebook.add(self._box_review_tab, text="Pick box spots")
        self._rotation_plot_tab = ttk.Frame(notebook)
        notebook.add(self._rotation_plot_tab, text="Rotation plot")
        notebook.bind("<<NotebookTabChanged>>", self._on_tab_changed)

        self._build_live_ui(self._live_tab)
        self._build_box_review_ui(self._box_review_tab)
        self._build_rotation_plot_ui(self._rotation_plot_tab)
        self.bottom_var = tk.StringVar(value="")
        ttk.Label(self.root, textvariable=self.bottom_var, anchor="w").pack(
            side=tk.BOTTOM, fill=tk.X, padx=8, pady=(0, 8)
        )

    def _build_box_review_ui(self, parent: tk.Widget) -> None:
        top = ttk.Frame(parent, padding=8)
        top.pack(side=tk.TOP, fill=tk.X)
        ttk.Button(top, text="Load saved box recordings", command=self._box_review_load_saved_recordings).pack(side=tk.LEFT)
        ttk.Button(top, text="Prev recording", command=lambda: self._box_review_step(-1)).pack(side=tk.LEFT)
        ttk.Button(top, text="Next recording", command=lambda: self._box_review_step(1)).pack(side=tk.LEFT, padx=(6, 0))
        ttk.Button(top, text="Undo last spot", command=self._box_review_undo_spot).pack(side=tk.LEFT, padx=(12, 0))
        ttk.Button(top, text="Clear spots", command=self._box_review_clear_spots).pack(side=tk.LEFT, padx=(6, 0))
        ttk.Label(top, text="Manual angle (deg)").pack(side=tk.LEFT, padx=(16, 0))
        angle_entry = ttk.Entry(top, textvariable=self._box_review_angle_var, width=8)
        angle_entry.pack(side=tk.LEFT, padx=(6, 0))
        angle_entry.bind("<Return>", lambda _e: self._box_review_apply_angle())
        ttk.Button(top, text="Apply angle", command=self._box_review_apply_angle).pack(side=tk.LEFT, padx=(6, 0))
        ttk.Button(top, text="Save and update plot", command=self._box_review_save_and_update).pack(side=tk.LEFT, padx=(12, 0))
        ttk.Label(top, textvariable=self._box_review_status_var).pack(side=tk.RIGHT)

        self._box_review_img_label = tk.Label(parent, bg="black")
        self._box_review_img_label.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=8, pady=(0, 8))
        self._box_review_img_label.bind("<Button-1>", self._box_review_on_click)

    def _build_rotation_plot_ui(self, parent: tk.Widget) -> None:
        top = ttk.Frame(parent, padding=8)
        top.pack(side=tk.TOP, fill=tk.X)
        ttk.Button(top, text="Update plot", command=self._update_rotation_plot).pack(side=tk.LEFT)
        ttk.Button(top, text="Auto analyse saved boxes", command=self._auto_analyse_saved_boxes).pack(
            side=tk.LEFT, padx=(8, 0)
        )
        ttk.Label(top, textvariable=self._rotation_plot_status_var).pack(side=tk.RIGHT)
        self._rotation_plot_label = tk.Label(parent, bg="white")
        self._rotation_plot_label.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=8, pady=(0, 8))

    def _build_live_ui(self, parent: tk.Widget) -> None:
        top = ttk.Frame(parent, padding=8)
        top.pack(side=tk.TOP, fill=tk.X)

        self._live_start_btn = ttk.Button(top, text="Start live feed", command=self._start_live_feed)
        self._live_start_btn.pack(side=tk.LEFT)
        self._live_stop_btn = ttk.Button(top, text="Stop live feed", command=self._stop_live_feed)
        self._live_stop_btn.state(["disabled"])
        self._live_stop_btn.pack(side=tk.LEFT, padx=(6, 0))

        ttk.Label(top, text="FPS 20").pack(side=tk.LEFT, padx=(12, 0))
        ttk.Label(top, text="Exp (ms)").pack(side=tk.LEFT, padx=(12, 0))
        ttk.Entry(top, textvariable=self._live_exp_ms_var, width=7).pack(side=tk.LEFT)
        ttk.Label(top, text="Gain").pack(side=tk.LEFT, padx=(12, 0))
        ttk.Entry(top, textvariable=self._live_gain_var, width=7).pack(side=tk.LEFT)
        ttk.Button(top, text="Apply", command=self._apply_live_settings).pack(side=tk.LEFT, padx=(8, 0))
        ttk.Checkbutton(top, text="Magnifier", variable=self._live_mag_enabled_var).pack(side=tk.LEFT, padx=(12, 0))
        ttk.Label(top, text="Zoom x").pack(side=tk.LEFT, padx=(6, 0))
        ttk.Entry(top, textvariable=self._live_zoom_var, width=5).pack(side=tk.LEFT)
        ttk.Label(top, textvariable=self._live_status_var).pack(side=tk.RIGHT)

        view = ttk.Frame(parent, padding=8)
        view.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        view.columnconfigure(0, weight=1)
        view.columnconfigure(1, weight=0, minsize=self._live_zoom_output_px + 10)
        view.rowconfigure(0, weight=1)

        left = ttk.Frame(view)
        left.grid(row=0, column=0, sticky="nsew")
        left.grid_propagate(False)
        self._live_left_frame = left
        self._live_img_label = tk.Label(left, bg="black")
        self._live_img_label.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self._live_img_label.bind("<Button-1>", self._on_live_click)
        self._live_img_label.bind("<B1-Motion>", self._on_live_drag)
        self._live_img_label.bind("<ButtonRelease-1>", self._on_live_release)

        right = ttk.Frame(view, width=self._live_zoom_output_px + 60, height=self._live_zoom_output_px + 560)
        right.grid(row=0, column=1, sticky="n", padx=(10, 0))
        right.grid_propagate(False)
        self._protractor_right_frame = right

        self._live_zoom_label = tk.Label(right, bg="black")
        self._live_zoom_label.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        ttk.Label(right, text="Magnifier histogram").pack(side=tk.TOP, anchor="w", pady=(6, 0))
        self._live_hist_label = tk.Label(right, bg="white")
        self._live_hist_label.pack(side=tk.TOP, fill=tk.X)
        ttk.Label(right, textvariable=self._live_theta_var, justify=tk.LEFT, width=64).pack(
            side=tk.TOP, anchor="w", pady=(6, 0)
        )
        ttk.Label(right, text="Live XY (magnifier center)").pack(side=tk.TOP, anchor="w", pady=(8, 0))
        self._live_xy_label = ttk.Label(right)
        self._live_xy_label.pack(side=tk.TOP, anchor="w", pady=(2, 0))

        protractor = ttk.LabelFrame(right, text="Protractor")
        protractor.pack(side=tk.TOP, fill=tk.X, pady=(10, 8))
        btn_row = ttk.Frame(protractor)
        btn_row.pack(side=tk.TOP, fill=tk.X, padx=6, pady=(6, 2))
        ttk.Button(btn_row, text="Set point 1", command=lambda: self._arm_point_set(1)).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(btn_row, text="Set point 2", command=lambda: self._arm_point_set(2)).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(6, 0))
        btn_row2 = ttk.Frame(protractor)
        btn_row2.pack(side=tk.TOP, fill=tk.X, padx=6, pady=(2, 6))
        ttk.Button(btn_row2, text="Redraw points", command=self._reset_protractor_points).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(btn_row2, text="Clear", command=self._clear_protractor_points).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(6, 0))
        ttk.Checkbutton(
            protractor,
            text="Track points with stage motion",
            variable=self._protractor_track_var,
        ).pack(side=tk.TOP, anchor="w", padx=6, pady=(0, 4))
        ttk.Label(protractor, textvariable=self._protractor_status_var, justify=tk.LEFT).pack(
            side=tk.TOP, anchor="w", padx=6, pady=(0, 2)
        )
        ttk.Label(protractor, textvariable=self._protractor_angle_var, justify=tk.LEFT).pack(
            side=tk.TOP, anchor="w", padx=6, pady=(0, 6)
        )

        box_record = ttk.LabelFrame(right, text="Box recording")
        box_record.pack(side=tk.TOP, fill=tk.X, pady=(0, 8))
        box_buttons = ttk.Frame(box_record)
        box_buttons.pack(side=tk.TOP, fill=tk.X, padx=6, pady=(6, 2))
        ttk.Button(box_buttons, text="Draw box", command=self._arm_box_record_draw).pack(
            side=tk.LEFT, fill=tk.X, expand=True
        )
        ttk.Button(box_buttons, text="Clear box", command=self._clear_box_record).pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=(6, 0)
        )
        self._box_record_btn = ttk.Button(
            box_record,
            text="Record selected box",
            command=self._live_capture_selected_box,
        )
        self._box_record_btn.pack(side=tk.TOP, anchor="w", padx=6, pady=(2, 4))
        ttk.Label(box_record, textvariable=self._box_record_status_var, justify=tk.LEFT).pack(
            side=tk.TOP, anchor="w", padx=6, pady=(0, 6)
        )

        cap_row = ttk.Frame(right)
        cap_row.pack(side=tk.TOP, fill=tk.X, pady=(8, 0))
        settings = ttk.LabelFrame(cap_row, text="Magnifier capture settings")
        settings.pack(side=tk.TOP, fill=tk.X, pady=(0, 8))
        ttk.Label(settings, text="Exp (ms)").grid(row=0, column=0, sticky="w")
        ttk.Entry(settings, textvariable=self._live_capture_exp_ms_var, width=7).grid(
            row=0, column=1, sticky="w", padx=(6, 10)
        )
        ttk.Label(settings, text="FPS req").grid(row=0, column=2, sticky="w")
        ttk.Entry(settings, textvariable=self._live_capture_fps_var, width=7).grid(
            row=0, column=3, sticky="w", padx=(6, 0)
        )
        ttk.Label(settings, text="Gain A").grid(row=1, column=0, sticky="w", pady=(4, 0))
        ttk.Entry(settings, textvariable=self._live_capture_gain_analog_var, width=7).grid(
            row=1, column=1, sticky="w", padx=(6, 10), pady=(4, 0)
        )
        ttk.Label(settings, text="Gain D").grid(row=1, column=2, sticky="w", pady=(4, 0))
        ttk.Entry(settings, textvariable=self._live_capture_gain_digital_var, width=7).grid(
            row=1, column=3, sticky="w", padx=(6, 0), pady=(4, 0)
        )
        ttk.Label(settings, text="Frames").grid(row=2, column=0, sticky="w", pady=(4, 0))
        ttk.Entry(settings, textvariable=self._live_capture_frames_var, width=7).grid(
            row=2, column=1, sticky="w", padx=(6, 10), pady=(4, 0)
        )
        ttk.Label(settings, text="ROI raw").grid(row=2, column=2, sticky="w", pady=(4, 0))
        ttk.Entry(settings, textvariable=self._live_capture_roi_var, width=7).grid(
            row=2, column=3, sticky="w", padx=(6, 0), pady=(4, 0)
        )
        self._live_capture_btn = ttk.Button(
            cap_row,
            text="Capture stationary from magnifier",
            command=self._live_capture_stationary_from_magnifier,
        )
        self._live_capture_btn.pack(side=tk.TOP, anchor="w")
        ttk.Label(cap_row, textvariable=self._live_capture_status_var, justify=tk.LEFT).pack(
            side=tk.TOP, anchor="w", pady=(4, 0)
        )

        blank = Image.new("L", (self._live_zoom_output_px, self._live_zoom_output_px), 0)
        self._live_zoom_blank_ref = ImageTk.PhotoImage(blank)
        self._live_zoom_label.configure(image=self._live_zoom_blank_ref)
        hist_blank = Image.new("RGB", (self._live_zoom_output_px, 90), "white")
        self._live_hist_blank_ref = ImageTk.PhotoImage(hist_blank)
        self._live_hist_label.configure(image=self._live_hist_blank_ref)
        live_xy_blank = self._make_xy_scatter_image([])
        self._live_xy_ref = ImageTk.PhotoImage(live_xy_blank)
        self._live_xy_label.configure(image=self._live_xy_ref)

    def _arm_point_set(self, point_index: int) -> None:
        self._protractor_mode = f"set_{point_index}"
        self._protractor_pending_point = point_index
        self._protractor_status_var.set(f"Click to place point {point_index}")

    def _reset_protractor_points(self) -> None:
        self._protractor_point1 = None
        self._protractor_point2 = None
        self._arm_point_set(1)
        self._update_protractor_text()

    def _clear_protractor_points(self) -> None:
        self._protractor_point1 = None
        self._protractor_point2 = None
        self._protractor_mode = "normal"
        self._protractor_drag_index = None
        self._update_protractor_text()

    def _frame_coords_from_event(self, event) -> tuple[float, float] | None:
        if self._live_last_frame is None:
            return None
        off_x, off_y = self._live_disp_offset
        scale = float(self._live_disp_scale) if self._live_disp_scale else 1.0
        x = float(event.x) - float(off_x)
        y = float(event.y) - float(off_y)
        if x < 0 or y < 0:
            return None
        src_h, src_w = self._live_last_frame.shape
        fx = x / scale
        fy = y / scale
        if fx < 0 or fy < 0 or fx >= src_w or fy >= src_h:
            return None
        return (float(fx), float(fy))

    def _tracked_point(self, point: tuple[float, float] | None) -> tuple[float, float] | None:
        if point is None:
            return None
        x, y = point
        if self._protractor_track_var.get():
            dx, dy = self._live_track_shift
            x += float(dx)
            y += float(dy)
        if self._live_last_frame is not None:
            h, w = self._live_last_frame.shape
            x = max(0.0, min(float(w - 1), x))
            y = max(0.0, min(float(h - 1), y))
        return (x, y)

    def _base_point_from_current(self, point: tuple[float, float]) -> tuple[float, float]:
        x, y = float(point[0]), float(point[1])
        if self._protractor_track_var.get():
            dx, dy = self._live_track_shift
            x -= float(dx)
            y -= float(dy)
        return (x, y)

    def _arm_box_record_draw(self) -> None:
        self._box_record_mode = "draw"
        self._box_record_dragging = False
        self._box_record_point1 = None
        self._box_record_point2 = None
        self._box_record_status_var.set("Drag on the live image to select a box.")

    def _clear_box_record(self) -> None:
        self._box_record_mode = "normal"
        self._box_record_dragging = False
        self._box_record_point1 = None
        self._box_record_point2 = None
        self._box_record_status_var.set("Box recording idle.")

    def _box_point_from_current(self, point: tuple[float, float]) -> tuple[float, float]:
        x, y = float(point[0]), float(point[1])
        if self._protractor_track_var.get():
            dx, dy = self._live_track_shift
            x -= float(dx)
            y -= float(dy)
        return (x, y)

    def _tracked_box_point(self, point: tuple[float, float] | None) -> tuple[float, float] | None:
        if point is None:
            return None
        x, y = float(point[0]), float(point[1])
        if self._protractor_track_var.get():
            dx, dy = self._live_track_shift
            x += float(dx)
            y += float(dy)
        if self._live_last_frame is not None:
            h, w = self._live_last_frame.shape
            x = max(0.0, min(float(w - 1), x))
            y = max(0.0, min(float(h - 1), y))
        return (x, y)

    def _selected_box_bounds(self) -> tuple[float, float, float, float] | None:
        p1 = self._tracked_box_point(self._box_record_point1)
        p2 = self._tracked_box_point(self._box_record_point2)
        if p1 is None or p2 is None:
            return None
        x0 = min(float(p1[0]), float(p2[0]))
        y0 = min(float(p1[1]), float(p2[1]))
        x1 = max(float(p1[0]), float(p2[0]))
        y1 = max(float(p1[1]), float(p2[1]))
        if (x1 - x0) < 2.0 or (y1 - y0) < 2.0:
            return None
        return (x0, y0, x1, y1)

    def _box_roi_from_bounds(self, bounds: tuple[float, float, float, float]) -> dict:
        if self._live_last_frame is None:
            raise RuntimeError("No live frame available.")
        h, w = self._live_last_frame.shape
        x0, y0, x1, y1 = bounds
        x = int(math.floor(x0))
        y = int(math.floor(y0))
        rw = int(math.ceil(x1)) - x
        rh = int(math.ceil(y1)) - y
        rw = max(2, int(rw))
        rh = max(2, int(rh))
        if rw % 2:
            rw += 1
        if rh % 2:
            rh += 1
        x = max(0, min(int(w) - rw, x))
        y = max(0, min(int(h) - rh, y))
        if x % 2:
            x = max(0, x - 1)
        if y % 2:
            y = max(0, y - 1)
        rw = min(rw, int(w) - x)
        rh = min(rh, int(h) - y)
        if rw % 2:
            rw -= 1
        if rh % 2:
            rh -= 1
        if rw < 2 or rh < 2:
            raise RuntimeError("Selected box is too small.")
        return {
            "x": int(x),
            "y": int(y),
            "w": int(rw),
            "h": int(rh),
            "cx": float(x) + 0.5 * float(rw),
            "cy": float(y) + 0.5 * float(rh),
            "phase_x": int(x) % 2,
            "phase_y": int(y) % 2,
        }

    def _box_review_record_from_dir(self, rec_dir: Path) -> dict:
        meta_path = rec_dir / "meta.json"
        arr_path = rec_dir / "box_recording.npy"
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        arr = np.load(arr_path, mmap_mode="r", allow_pickle=False)
        n = int(arr.shape[0]) if getattr(arr, "ndim", 0) >= 3 else 0
        idx = np.linspace(0, max(0, n - 1), min(max(1, n), 400)).round().astype(int)
        mean_img = np.mean(np.asarray(arr[idx], dtype=np.float32), axis=0) if n else np.zeros((1, 1), dtype=np.float32)
        actual_roi = meta.get("actual", {}).get("roi") or meta.get("requested", {}).get("roi") or {}
        origin_x = int(round(float(actual_roi.get("OffsetX", actual_roi.get("x", 0)))))
        origin_y = int(round(float(actual_roi.get("OffsetY", actual_roi.get("y", 0)))))
        manual_angle = meta.get("manual_angle_deg")
        if manual_angle is None:
            manual_angle = 20.0 * float(len(self._box_review_records))
        return {
            "dir": rec_dir,
            "name": rec_dir.name,
            "meta": meta,
            "arr_path": arr_path,
            "mean_img": mean_img,
            "origin_x": origin_x,
            "origin_y": origin_y,
            "manual_angle_deg": float(manual_angle),
            "spots": list(meta.get("manual_spots", []) or []),
        }

    def _box_review_add_recording(self, rec_dir: Path) -> None:
        try:
            record = self._box_review_record_from_dir(rec_dir)
        except Exception as e:
            self._ui_call(messagebox.showerror, "Box review", f"Could not load recording: {e}")
            return
        self._box_review_records.append(record)
        self._box_review_current_index = len(self._box_review_records) - 1
        self._ui_call(self._box_review_show_current)
        self._ui_call(self._notebook.select, self._box_review_tab)

    def _box_review_load_saved_recordings(self) -> None:
        _root, pending_dir, _good = self._stationary_dataset_paths()
        dirs = []
        for d in pending_dir.iterdir():
            if not d.is_dir() or not d.name.startswith("field_roi_"):
                continue
            if not (d / "box_recording.npy").exists() or not (d / "meta.json").exists():
                continue
            try:
                meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
                created = str(meta.get("created_local", ""))
            except Exception:
                created = d.name
            dirs.append((created, d))
        dirs.sort(key=lambda v: v[0])
        records = []
        errors = []
        for _created, d in dirs:
            try:
                records.append(self._box_review_record_from_dir(d))
            except Exception as e:
                errors.append(f"{d.name}: {e}")
        self._box_review_records = records
        self._box_review_current_index = 0 if records else -1
        if records:
            self._box_review_show_current()
            self._box_review_status_var.set(f"Loaded {len(records)} saved box recordings.")
            self._update_rotation_plot()
        else:
            self._box_review_status_var.set("No saved box recordings found in pending.")
        if errors:
            messagebox.showwarning("Box review", "Some recordings could not be loaded:\n" + "\n".join(errors[:5]))

    def _box_review_step(self, direction: int) -> None:
        if not self._box_review_records:
            return
        self._box_review_current_index = (self._box_review_current_index + int(direction)) % len(self._box_review_records)
        self._box_review_show_current()

    def _box_review_apply_angle(self) -> None:
        rec = self._box_review_current_record()
        if rec is None:
            return
        value = self._parse_float(self._box_review_angle_var.get())
        if value is None:
            messagebox.showerror("Box review", "Manual angle must be a number.")
            return
        rec["manual_angle_deg"] = float(value)
        rec["meta"]["manual_angle_deg"] = float(value)
        self._write_json_atomic(Path(rec["dir"]) / "meta.json", rec["meta"])
        self._box_review_show_current()
        self._update_rotation_plot()

    def _box_review_current_record(self) -> dict | None:
        if not self._box_review_records:
            return None
        i = max(0, min(len(self._box_review_records) - 1, int(self._box_review_current_index)))
        self._box_review_current_index = i
        return self._box_review_records[i]

    def _box_review_to_u8(self, img: np.ndarray) -> np.ndarray:
        arr = np.asarray(img, dtype=np.float32)
        finite = np.isfinite(arr)
        if not finite.any():
            return np.zeros(arr.shape, dtype=np.uint8)
        lo, hi = np.percentile(arr[finite], [1.0, 99.7])
        if hi <= lo:
            hi = lo + 1.0
        return np.clip((arr - lo) * 255.0 / (hi - lo), 0, 255).astype(np.uint8)

    def _box_review_show_current(self) -> None:
        rec = self._box_review_current_record()
        if rec is None or self._box_review_img_label is None:
            self._box_review_status_var.set("No box recordings loaded.")
            return
        self._box_review_angle_var.set(f"{float(rec['manual_angle_deg']):.3g}")
        base = Image.fromarray(self._box_review_to_u8(rec["mean_img"])).convert("RGB")
        label_w = max(100, int(self._box_review_img_label.winfo_width()))
        label_h = max(100, int(self._box_review_img_label.winfo_height()))
        src_w, src_h = base.size
        scale = min(float(label_w) / float(src_w), float(label_h) / float(src_h))
        scale = max(0.05, scale)
        disp_w = max(1, int(round(src_w * scale)))
        disp_h = max(1, int(round(src_h * scale)))
        try:
            resample = Image.Resampling.NEAREST
        except Exception:
            resample = Image.NEAREST
        img = base.resize((disp_w, disp_h), resample=resample)
        canvas = Image.new("RGB", (label_w, label_h), 0)
        off_x = max(0, (label_w - disp_w) // 2)
        off_y = max(0, (label_h - disp_h) // 2)
        canvas.paste(img, (off_x, off_y))
        draw = ImageDraw.Draw(canvas)
        for i, spot in enumerate(rec["spots"]):
            x = off_x + float(spot["x_px"]) * scale
            y = off_y + float(spot["y_px"]) * scale
            r = 8
            draw.ellipse([x - r, y - r, x + r, y + r], outline=(255, 80, 40), width=2)
            draw.text((x + r + 2, y - r), str(i + 1), fill=(255, 80, 40))
        self._box_review_disp_scale = scale
        self._box_review_disp_offset = (off_x, off_y)
        self._box_review_photo = ImageTk.PhotoImage(canvas)
        self._box_review_img_label.configure(image=self._box_review_photo)
        self._box_review_status_var.set(
            f"{self._box_review_current_index + 1}/{len(self._box_review_records)} "
            f"{rec['name']} | angle={float(rec['manual_angle_deg']):.1f} deg | spots={len(rec['spots'])}"
        )

    def _box_review_on_click(self, event) -> None:
        rec = self._box_review_current_record()
        if rec is None:
            return
        off_x, off_y = self._box_review_disp_offset
        scale = float(self._box_review_disp_scale) if self._box_review_disp_scale else 1.0
        x = (float(event.x) - float(off_x)) / scale
        y = (float(event.y) - float(off_y)) / scale
        h, w = np.asarray(rec["mean_img"]).shape[:2]
        if x < 0 or y < 0 or x >= w or y >= h:
            return
        rec["spots"].append({"x_px": float(x), "y_px": float(y)})
        self._box_review_save_current_spots()
        self._box_review_show_current()

    def _box_review_undo_spot(self) -> None:
        rec = self._box_review_current_record()
        if rec is None or not rec["spots"]:
            return
        rec["spots"].pop()
        self._box_review_save_current_spots()
        self._box_review_show_current()
        self._update_rotation_plot()

    def _box_review_clear_spots(self) -> None:
        rec = self._box_review_current_record()
        if rec is None:
            return
        rec["spots"] = []
        self._box_review_save_current_spots()
        self._box_review_show_current()
        self._update_rotation_plot()

    def _box_review_save_current_spots(self) -> None:
        rec = self._box_review_current_record()
        if rec is None:
            return
        rec["meta"]["manual_angle_deg"] = float(rec["manual_angle_deg"])
        rec["meta"]["manual_spots"] = list(rec["spots"])
        self._write_json_atomic(Path(rec["dir"]) / "meta.json", rec["meta"])

    def _box_review_save_and_update(self) -> None:
        self._box_review_apply_angle()
        self._box_review_save_current_spots()
        self._update_rotation_plot()

    def _box_record_xy_phi_for_spot(self, rec: dict, spot: dict) -> dict:
        arr = np.load(Path(rec["arr_path"]), mmap_mode="r", allow_pickle=False)
        n = int(arr.shape[0]) if getattr(arr, "ndim", 0) >= 3 else 0
        if n <= 0:
            return {"x": 0.0, "y": 0.0, "phi_deg": 0.0, "r": 0.0}
        idx = np.linspace(0, n - 1, min(n, 700)).round().astype(int)
        win = 14
        cx = float(spot["x_px"])
        cy = float(spot["y_px"])
        h, w = int(arr.shape[1]), int(arr.shape[2])
        x0 = int(round(cx)) - win // 2
        y0 = int(round(cy)) - win // 2
        x0 = max(0, min(w - win, x0))
        y0 = max(0, min(h - win, y0))
        if x0 % 2:
            x0 = max(0, x0 - 1)
        if y0 % 2:
            y0 = max(0, y0 - 1)
        xs: list[float] = []
        ys: list[float] = []
        for fi in idx:
            xv, yv, _phi, _pass = self._xy_phi_stats_from_frame(
                gray=np.asarray(arr[fi]),
                roi_meta=None,
                win_raw=win,
                center=(float(x0 + win / 2), float(y0 + win / 2)),
            )
            xs.append(float(xv))
            ys.append(float(yv))
        mx = float(np.mean(xs)) if xs else 0.0
        my = float(np.mean(ys)) if ys else 0.0
        return {
            "x": mx,
            "y": my,
            "phi_deg": float(0.5 * math.degrees(math.atan2(my, mx))),
            "r": float(math.hypot(mx, my)),
            "global_x_px": float(rec["origin_x"]) + cx,
            "global_y_px": float(rec["origin_y"]) + cy,
        }

    def _wrap_pm90(self, deg: float) -> float:
        x = float(deg)
        while x <= -90.0:
            x += 180.0
        while x > 90.0:
            x -= 180.0
        return x

    def _rotation_analysis_outputs_dir(self) -> Path:
        root, _pending, _good = self._stationary_dataset_paths()
        out = root / "plots" / "manual_box_rotation_gui"
        out.mkdir(parents=True, exist_ok=True)
        return out

    def _rotation_compute_rows(self) -> tuple[list[dict], list[dict], list[dict]]:
        records = [r for r in self._box_review_records if r.get("spots")]
        records.sort(key=lambda r: float(r.get("manual_angle_deg", 0.0)))
        spot_rows: list[dict] = []
        per_record: list[list[dict]] = []
        for ri, rec in enumerate(records):
            rec_spots = []
            for si, spot in enumerate(rec["spots"]):
                stats = self._box_record_xy_phi_for_spot(rec, spot)
                row = {
                    "record_index": ri,
                    "recording": rec["name"],
                    "manual_angle_deg": float(rec["manual_angle_deg"]),
                    "spot_index": si,
                    "local_x_px": float(spot["x_px"]),
                    "local_y_px": float(spot["y_px"]),
                    **stats,
                }
                rec_spots.append(row)
                spot_rows.append(row)
            per_record.append(rec_spots)

        if len(per_record) < 2:
            return records, spot_rows, []

        def build_tracks(sign: float) -> tuple[list[list[tuple[int, int]]], int]:
            links: dict[int, list[tuple[int, int]]] = {}
            total_links = 0
            for ri in range(len(per_record) - 1):
                a = per_record[ri]
                b = per_record[ri + 1]
                expected = float(records[ri + 1]["manual_angle_deg"]) - float(records[ri]["manual_angle_deg"])
                candidates = []
                for ia, sa in enumerate(a):
                    for ib, sb in enumerate(b):
                        dpos = math.hypot(float(sa["global_x_px"]) - float(sb["global_x_px"]), float(sa["global_y_px"]) - float(sb["global_y_px"]))
                        dphi = self._wrap_pm90(sign * (float(sb["phi_deg"]) - float(sa["phi_deg"])))
                        err = abs(dphi - expected)
                        if dpos <= 80.0 and err <= max(12.0, abs(expected) * 0.7):
                            candidates.append((err + 0.04 * dpos, ia, ib))
                candidates.sort()
                used_a = set()
                used_b = set()
                chosen = []
                for _cost, ia, ib in candidates:
                    if ia in used_a or ib in used_b:
                        continue
                    used_a.add(ia)
                    used_b.add(ib)
                    chosen.append((ia, ib))
                links[ri] = chosen
                total_links += len(chosen)

            tracks: list[list[tuple[int, int]]] = []
            for ia in range(len(per_record[0])):
                track = [(0, ia)]
                cur = ia
                for ri in range(len(per_record) - 1):
                    nxt = None
                    for a_i, b_i in links.get(ri, []):
                        if a_i == cur:
                            nxt = b_i
                            break
                    if nxt is None:
                        break
                    track.append((ri + 1, nxt))
                    cur = nxt
                if len(track) >= 2:
                    tracks.append(track)
            return tracks, total_links

        choices = []
        for sign in (1.0, -1.0):
            tracks, total_links = build_tracks(sign)
            common_all = sum(1 for t in tracks if len(t) == len(records))
            total_len = sum(len(t) for t in tracks)
            choices.append(((common_all, total_len, total_links), sign, tracks))
        choices.sort(reverse=True, key=lambda v: v[0])
        _score, sign, tracks = choices[0]

        track_rows: list[dict] = []
        for ti, track in enumerate(tracks):
            base_ri, base_si = track[0]
            base_phi = float(per_record[base_ri][base_si]["phi_deg"])
            for ri, si in track:
                row = dict(per_record[ri][si])
                row["track_id"] = ti
                row["track_length"] = len(track)
                row["phi_change_from_first_deg"] = self._wrap_pm90(sign * (float(row["phi_deg"]) - base_phi))
                track_rows.append(row)
        return records, spot_rows, track_rows

    def _update_rotation_plot(self) -> None:
        try:
            records, spot_rows, track_rows = self._rotation_compute_rows()
        except Exception as e:
            self._rotation_plot_status_var.set(f"Analysis failed: {e}")
            return

        out_dir = self._rotation_analysis_outputs_dir()
        spots_csv = out_dir / "manual_box_spot_phi_values.csv"
        tracks_csv = out_dir / "manual_box_matched_tracks.csv"
        mean_csv = out_dir / "manual_box_mean_delta_phi_values.csv"
        summary_json = out_dir / "manual_box_rotation_summary.json"
        plot_png = out_dir / "manual_box_mean_phi_vs_manual_rotation.png"

        def write_csv(path: Path, rows: list[dict]) -> None:
            if not rows:
                path.write_text("", encoding="utf-8")
                return
            keys = list(rows[0].keys())
            with path.open("w", encoding="utf-8", newline="") as f:
                import csv

                writer = csv.DictWriter(f, fieldnames=keys)
                writer.writeheader()
                writer.writerows(rows)

        write_csv(spots_csv, spot_rows)
        write_csv(tracks_csv, track_rows)

        record_angles = [float(r["manual_angle_deg"]) for r in records]
        mean_rows: list[dict] = []
        if records and track_rows:
            all_lengths = [int(r["track_length"]) for r in track_rows]
            max_len = max(all_lengths) if all_lengths else 0
            use_track_ids = {
                int(r["track_id"])
                for r in track_rows
                if int(r["track_length"]) == max_len and max_len >= 2
            }
            for ri, angle in enumerate(record_angles):
                vals = [
                    float(r["phi_change_from_first_deg"])
                    for r in track_rows
                    if int(r["track_id"]) in use_track_ids and int(r["record_index"]) == ri
                ]
                mean_rows.append(
                    {
                        "manual_angle_deg": angle - record_angles[0],
                        "mean_delta_phi_deg": float(np.mean(vals)) if vals else float("nan"),
                        "std_delta_phi_deg": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                        "n_rods": len(vals),
                    }
                )
        write_csv(mean_csv, mean_rows)

        if Figure is not None and FigureCanvas is not None:
            fig = Figure(figsize=(6.0, 4.8), dpi=120)
            ax = fig.add_subplot(111)
            if mean_rows:
                xs = [float(r["manual_angle_deg"]) for r in mean_rows]
                ys = [float(r["mean_delta_phi_deg"]) for r in mean_rows]
                es = [float(r["std_delta_phi_deg"]) for r in mean_rows]
                ax.errorbar(xs, ys, yerr=es, fmt="o-", capsize=4, color="tab:blue", label="Mean matched rods")
                hi = max(20.0, max([v for v in xs + ys if np.isfinite(v)] or [20.0]) + 5.0)
            else:
                hi = 20.0
            ax.plot([0.0, hi], [0.0, hi], "--", color="0.45", linewidth=1.2, label="y = x")
            ax.set_xlim(0.0, hi)
            ax.set_ylim(0.0, hi)
            ax.set_aspect("equal", adjustable="box")
            ax.set_xlabel("Manual rotation change (deg)")
            ax.set_ylabel("Mean phi change (deg)")
            ax.set_title("Manual box rotation")
            ax.grid(True, alpha=0.25)
            ax.legend(frameon=False)
            fig.tight_layout()
            fig.savefig(plot_png, bbox_inches="tight")
            canvas = FigureCanvas(fig)
            canvas.draw()
            w, h = canvas.get_width_height()
            img = Image.frombuffer("RGBA", (w, h), canvas.buffer_rgba(), "raw", "RGBA", 0, 1).convert("RGB")
            self._rotation_plot_photo = ImageTk.PhotoImage(img)
            if self._rotation_plot_label is not None:
                self._rotation_plot_label.configure(image=self._rotation_plot_photo)

        payload = {
            "records": [{"name": r["name"], "manual_angle_deg": float(r["manual_angle_deg"]), "spots": len(r["spots"])} for r in records],
            "spots_total": len(spot_rows),
            "tracks_total": len({int(r["track_id"]) for r in track_rows}) if track_rows else 0,
            "mean_rows": mean_rows,
            "outputs": {
                "mean_csv": str(mean_csv),
                "spots_csv": str(spots_csv),
                "tracks_csv": str(tracks_csv),
                "plot_png": str(plot_png),
            },
        }
        self._write_json_atomic(summary_json, payload)
        self._rotation_plot_status_var.set(
            f"Saved mean delta phi table: {mean_csv.name} | tracks={payload['tracks_total']}"
        )

    def _saved_box_recording_dirs(self) -> list[Path]:
        _root, pending_dir, _good = self._stationary_dataset_paths()
        dirs = []
        for d in pending_dir.iterdir():
            if not d.is_dir() or not d.name.startswith("field_roi_"):
                continue
            if not (d / "box_recording.npy").exists() or not (d / "meta.json").exists():
                continue
            try:
                meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
                created = str(meta.get("created_local", ""))
            except Exception:
                created = d.name
            dirs.append((created, d))
        dirs.sort(key=lambda v: v[0])
        return [d for _created, d in dirs]

    def _auto_record_from_dir(self, rec_dir: Path, index: int) -> dict:
        meta = json.loads((rec_dir / "meta.json").read_text(encoding="utf-8"))
        arr_path = rec_dir / "box_recording.npy"
        arr = np.load(arr_path, mmap_mode="r", allow_pickle=False)
        actual_roi = meta.get("actual", {}).get("roi") or meta.get("requested", {}).get("roi") or {}
        origin_x = int(round(float(actual_roi.get("OffsetX", actual_roi.get("x", 0)))))
        origin_y = int(round(float(actual_roi.get("OffsetY", actual_roi.get("y", 0)))))
        manual_angle = meta.get("manual_angle_deg")
        if manual_angle is None:
            manual_angle = 20.0 * float(index)
        return {
            "dir": rec_dir,
            "name": rec_dir.name,
            "meta": meta,
            "arr_path": arr_path,
            "origin_x": origin_x,
            "origin_y": origin_y,
            "manual_angle_deg": float(manual_angle),
        }

    def _auto_spots_for_record(self, rec: dict) -> list[dict]:
        arr = np.load(Path(rec["arr_path"]), mmap_mode="r", allow_pickle=False)
        n = int(arr.shape[0]) if getattr(arr, "ndim", 0) >= 3 else 0
        if n <= 0:
            return []
        idx = np.linspace(0, n - 1, min(n, 700)).round().astype(int)
        frames = np.asarray(arr[idx], dtype=np.float32)
        mean_img = np.mean(frames, axis=0)
        h, w = mean_img.shape

        # Build an intensity-weighted absolute anisotropy image on raw-pixel coordinates.
        even_h = h - (h % 2)
        even_w = w - (w % 2)
        if even_h < 4 or even_w < 4:
            return []
        g = frames[:, :even_h, :even_w]
        ox = int(rec["origin_x"])
        oy = int(rec["origin_y"])
        px = ox % 2
        py = oy % 2
        I90 = g[:, py::2, px::2]
        I45 = g[:, py::2, (1 - px) :: 2]
        I135 = g[:, (1 - py) :: 2, px::2]
        I0 = g[:, (1 - py) :: 2, (1 - px) :: 2]
        ch_h = min(I0.shape[1], I45.shape[1], I135.shape[1], I90.shape[1])
        ch_w = min(I0.shape[2], I45.shape[2], I135.shape[2], I90.shape[2])
        I0 = I0[:, :ch_h, :ch_w]
        I45 = I45[:, :ch_h, :ch_w]
        I135 = I135[:, :ch_h, :ch_w]
        I90 = I90[:, :ch_h, :ch_w]
        m0 = np.mean(I0, axis=0)
        m45 = np.mean(I45, axis=0)
        m135 = np.mean(I135, axis=0)
        m90 = np.mean(I90, axis=0)
        eps = 1e-6
        x_map = (m0 - m90) / (m0 + m90 + eps)
        y_map = (m45 - m135) / (m45 + m135 + eps)
        i_map = 0.25 * (m0 + m45 + m135 + m90)
        score = i_map * (np.abs(x_map) + np.abs(y_map))
        score = cv2.GaussianBlur(score.astype(np.float32), (0, 0), sigmaX=1.2, borderType=cv2.BORDER_REPLICATE)
        finite = np.isfinite(score)
        if not finite.any():
            return []
        threshold = max(float(np.percentile(score[finite], 98.5)), float(np.mean(score[finite]) + 2.5 * np.std(score[finite])))
        dilated = cv2.dilate(score, np.ones((5, 5), np.uint8))
        peaks = (score >= dilated - 1e-6) & (score >= threshold)
        ys, xs = np.nonzero(peaks)
        candidates = sorted(
            [(float(score[y, x]), int(x), int(y)) for y, x in zip(ys, xs)],
            reverse=True,
        )
        accepted: list[tuple[float, int, int]] = []
        min_sep = 5.0
        for val, cx_ch, cy_ch in candidates:
            raw_x = float(cx_ch) * 2.0 + 1.0
            raw_y = float(cy_ch) * 2.0 + 1.0
            if raw_x < 8 or raw_y < 8 or raw_x > w - 9 or raw_y > h - 9:
                continue
            if any(math.hypot(raw_x - (ax * 2.0 + 1.0), raw_y - (ay * 2.0 + 1.0)) < min_sep for _av, ax, ay in accepted):
                continue
            accepted.append((val, cx_ch, cy_ch))
            if len(accepted) >= 80:
                break

        spots = []
        for spot_i, (val, cx_ch, cy_ch) in enumerate(accepted):
            cx = float(cx_ch) * 2.0 + 1.0
            cy = float(cy_ch) * 2.0 + 1.0
            stats = self._auto_xy_phi_for_center(arr, cx, cy)
            if stats["r"] < 0.08:
                continue
            spots.append(
                {
                    "recording": rec["name"],
                    "manual_angle_deg": float(rec["manual_angle_deg"]),
                    "spot_index": int(spot_i),
                    "local_x_px": cx,
                    "local_y_px": cy,
                    "global_x_px": float(rec["origin_x"]) + cx,
                    "global_y_px": float(rec["origin_y"]) + cy,
                    "score": float(val),
                    **stats,
                }
            )
        return spots

    def _auto_xy_phi_for_center(self, arr: np.ndarray, cx: float, cy: float) -> dict:
        n = int(arr.shape[0]) if getattr(arr, "ndim", 0) >= 3 else 0
        idx = np.linspace(0, n - 1, min(n, 700)).round().astype(int)
        win = 14
        h, w = int(arr.shape[1]), int(arr.shape[2])
        x0 = int(round(float(cx))) - win // 2
        y0 = int(round(float(cy))) - win // 2
        x0 = max(0, min(w - win, x0))
        y0 = max(0, min(h - win, y0))
        if x0 % 2:
            x0 = max(0, x0 - 1)
        if y0 % 2:
            y0 = max(0, y0 - 1)
        xs: list[float] = []
        ys: list[float] = []
        brightness: list[float] = []
        for fi in idx:
            frame = np.asarray(arr[fi])
            xv, yv, _phi, _pass = self._xy_phi_stats_from_frame(
                gray=frame,
                roi_meta=None,
                win_raw=win,
                center=(float(x0 + win / 2), float(y0 + win / 2)),
            )
            xs.append(float(xv))
            ys.append(float(yv))
            brightness.append(float(np.mean(frame[y0 : y0 + win, x0 : x0 + win])))
        mx = float(np.mean(xs)) if xs else 0.0
        my = float(np.mean(ys)) if ys else 0.0
        return {
            "mean_x": mx,
            "mean_y": my,
            "phi_deg": float(0.5 * math.degrees(math.atan2(my, mx))),
            "r": float(math.hypot(mx, my)),
            "brightness": float(np.mean(brightness)) if brightness else 0.0,
        }

    def _auto_match_adjacent(self, records: list[dict], spots_by_record: list[list[dict]]) -> tuple[float, list[dict], list[dict]]:
        def run(sign: float) -> tuple[float, list[dict], list[dict]]:
            pair_rows: list[dict] = []
            mean_rows = [{"manual_angle_deg": 0.0, "mean_delta_phi_deg": 0.0, "std_delta_phi_deg": 0.0, "n_rods": len(spots_by_record[0]) if spots_by_record else 0}]
            cumulative = 0.0
            score = 0.0
            for ri in range(len(records) - 1):
                a = spots_by_record[ri]
                b = spots_by_record[ri + 1]
                expected = float(records[ri + 1]["manual_angle_deg"]) - float(records[ri]["manual_angle_deg"])
                candidates = []
                for ia, sa in enumerate(a):
                    for ib, sb in enumerate(b):
                        dpos = math.hypot(float(sa["global_x_px"]) - float(sb["global_x_px"]), float(sa["global_y_px"]) - float(sb["global_y_px"]))
                        dphi = self._wrap_pm90(sign * (float(sb["phi_deg"]) - float(sa["phi_deg"])))
                        err = abs(dphi - expected)
                        if dpos <= 90.0 and err <= max(12.0, abs(expected) * 0.75):
                            candidates.append((err + 0.04 * dpos, ia, ib, dpos, dphi, err))
                candidates.sort()
                used_a = set()
                used_b = set()
                deltas = []
                for _cost, ia, ib, dpos, dphi, err in candidates:
                    if ia in used_a or ib in used_b:
                        continue
                    used_a.add(ia)
                    used_b.add(ib)
                    deltas.append(float(dphi))
                    pair_rows.append(
                        {
                            "from_record_index": ri,
                            "to_record_index": ri + 1,
                            "from_recording": records[ri]["name"],
                            "to_recording": records[ri + 1]["name"],
                            "from_spot_index": int(a[ia]["spot_index"]),
                            "to_spot_index": int(b[ib]["spot_index"]),
                            "manual_delta_deg": expected,
                            "phi_delta_deg": float(dphi),
                            "position_delta_px": float(dpos),
                            "phi_error_from_expected_deg": float(err),
                        }
                    )
                mean_delta = float(np.mean(deltas)) if deltas else float("nan")
                std_delta = float(np.std(deltas, ddof=1)) if len(deltas) > 1 else 0.0
                if np.isfinite(mean_delta):
                    cumulative += mean_delta
                    score += len(deltas) * 10.0 - abs(mean_delta - expected)
                else:
                    cumulative = float("nan")
                mean_rows.append(
                    {
                        "manual_angle_deg": float(records[ri + 1]["manual_angle_deg"]) - float(records[0]["manual_angle_deg"]),
                        "mean_delta_phi_deg": cumulative,
                        "std_delta_phi_deg": std_delta,
                        "n_rods": len(deltas),
                    }
                )
            return score, mean_rows, pair_rows

        choices = [run(1.0), run(-1.0)]
        if choices[1][0] > choices[0][0]:
            return -1.0, choices[1][1], choices[1][2]
        return 1.0, choices[0][1], choices[0][2]

    def _auto_analyse_saved_boxes(self) -> None:
        try:
            dirs = self._saved_box_recording_dirs()
            if not dirs:
                messagebox.showinfo("Auto analysis", "No saved box recordings found in pending.")
                return
            records = [self._auto_record_from_dir(d, i) for i, d in enumerate(dirs)]
            spots_by_record = [self._auto_spots_for_record(r) for r in records]
            sign, mean_rows, pair_rows = self._auto_match_adjacent(records, spots_by_record)
        except Exception as e:
            self._rotation_plot_status_var.set(f"Auto analysis failed: {e}")
            messagebox.showerror("Auto analysis", str(e))
            return

        out_dir = self._rotation_analysis_outputs_dir()
        spots_csv = out_dir / "auto_box_anisotropy_spots.csv"
        pairs_csv = out_dir / "auto_box_adjacent_matches.csv"
        mean_csv = out_dir / "auto_box_mean_delta_phi_values.csv"
        summary_json = out_dir / "auto_box_rotation_summary.json"
        plot_png = out_dir / "auto_box_mean_phi_vs_manual_rotation.png"

        def write_csv(path: Path, rows: list[dict]) -> None:
            if not rows:
                path.write_text("", encoding="utf-8")
                return
            keys = list(rows[0].keys())
            with path.open("w", encoding="utf-8", newline="") as f:
                import csv

                writer = csv.DictWriter(f, fieldnames=keys)
                writer.writeheader()
                writer.writerows(rows)

        spot_rows = []
        for ri, spots in enumerate(spots_by_record):
            for s in spots:
                row = dict(s)
                row["record_index"] = ri
                spot_rows.append(row)
        write_csv(spots_csv, spot_rows)
        write_csv(pairs_csv, pair_rows)
        write_csv(mean_csv, mean_rows)

        if Figure is not None and FigureCanvas is not None:
            fig = Figure(figsize=(6.0, 4.8), dpi=120)
            ax = fig.add_subplot(111)
            xs = [float(r["manual_angle_deg"]) for r in mean_rows]
            ys = [float(r["mean_delta_phi_deg"]) for r in mean_rows]
            es = [float(r["std_delta_phi_deg"]) for r in mean_rows]
            ax.errorbar(xs, ys, yerr=es, fmt="o-", capsize=4, color="tab:blue", label="Adjacent common rods")
            finite_vals = [v for v in xs + ys if np.isfinite(v)]
            hi = max(20.0, max(finite_vals or [20.0]) + 5.0)
            ax.plot([0.0, hi], [0.0, hi], "--", color="0.45", linewidth=1.2, label="y = x")
            ax.set_xlim(0.0, hi)
            ax.set_ylim(0.0, hi)
            ax.set_aspect("equal", adjustable="box")
            ax.set_xlabel("Manual rotation change (deg)")
            ax.set_ylabel("Cumulative mean phi change (deg)")
            ax.set_title("Automatic box rotation")
            ax.grid(True, alpha=0.25)
            ax.legend(frameon=False)
            fig.tight_layout()
            fig.savefig(plot_png, bbox_inches="tight")
            canvas = FigureCanvas(fig)
            canvas.draw()
            w, h = canvas.get_width_height()
            img = Image.frombuffer("RGBA", (w, h), canvas.buffer_rgba(), "raw", "RGBA", 0, 1).convert("RGB")
            self._rotation_plot_photo = ImageTk.PhotoImage(img)
            if self._rotation_plot_label is not None:
                self._rotation_plot_label.configure(image=self._rotation_plot_photo)

        payload = {
            "records": [
                {"name": r["name"], "manual_angle_deg": float(r["manual_angle_deg"]), "spots": len(spots_by_record[i])}
                for i, r in enumerate(records)
            ],
            "phi_sign_used": sign,
            "adjacent_match_counts": [int(r["n_rods"]) for r in mean_rows[1:]],
            "mean_rows": mean_rows,
            "outputs": {
                "mean_csv": str(mean_csv),
                "spots_csv": str(spots_csv),
                "pairs_csv": str(pairs_csv),
                "plot_png": str(plot_png),
            },
        }
        self._write_json_atomic(summary_json, payload)
        self._rotation_plot_status_var.set(
            f"Auto analysis saved: {mean_csv.name} | matches={payload['adjacent_match_counts']}"
        )

    def _nearest_protractor_point_index(self, point: tuple[float, float]) -> int | None:
        candidates = []
        p1 = self._tracked_point(self._protractor_point1)
        p2 = self._tracked_point(self._protractor_point2)
        if p1 is not None:
            candidates.append((1, p1))
        if p2 is not None:
            candidates.append((2, p2))
        if not candidates:
            return None
        best_idx = None
        best_d2 = None
        for idx, p in candidates:
            d2 = (float(point[0]) - float(p[0])) ** 2 + (float(point[1]) - float(p[1])) ** 2
            if best_d2 is None or d2 < best_d2:
                best_idx = idx
                best_d2 = d2
        if best_d2 is None or math.sqrt(best_d2) > self.PROTRACTOR_HIT_RADIUS_PX:
            return None
        return best_idx

    def _set_protractor_point(self, point_index: int, point: tuple[float, float]) -> None:
        base = self._base_point_from_current(point)
        if point_index == 1:
            self._protractor_point1 = base
        else:
            self._protractor_point2 = base
        self._protractor_mode = "normal"
        self._update_protractor_text()

    def _on_live_click(self, event) -> None:
        point = self._frame_coords_from_event(event)
        if point is None:
            return

        if self._box_record_mode == "draw":
            base = self._box_point_from_current(point)
            self._box_record_point1 = base
            self._box_record_point2 = base
            self._box_record_dragging = True
            self._box_record_status_var.set("Drawing box...")
            return

        if self._protractor_mode.startswith("set_"):
            idx = 1 if self._protractor_mode.endswith("1") else 2
            self._set_protractor_point(idx, point)
            return

        drag_idx = self._nearest_protractor_point_index(point)
        if drag_idx is not None:
            self._protractor_drag_index = drag_idx
            self._protractor_status_var.set(f"Dragging point {drag_idx}")
            return

        super()._on_live_click(event)
        self._update_protractor_text()

    def _on_live_drag(self, event) -> None:
        if self._box_record_dragging:
            point = self._frame_coords_from_event(event)
            if point is None:
                return
            self._box_record_point2 = self._box_point_from_current(point)
            bounds = self._selected_box_bounds()
            if bounds is not None:
                x0, y0, x1, y1 = bounds
                self._box_record_status_var.set(
                    f"Box {int(round(x1 - x0))} x {int(round(y1 - y0))} px"
                )
            return

        if self._protractor_drag_index is None:
            return
        point = self._frame_coords_from_event(event)
        if point is None:
            return
        self._set_protractor_point(self._protractor_drag_index, point)
        self._protractor_drag_index = self._protractor_drag_index

    def _on_live_release(self, _event) -> None:
        if self._box_record_dragging:
            self._box_record_dragging = False
            self._box_record_mode = "normal"
            bounds = self._selected_box_bounds()
            if bounds is None:
                self._box_record_status_var.set("Box too small. Draw again.")
            else:
                x0, y0, x1, y1 = bounds
                self._box_record_status_var.set(
                    f"Selected box {int(round(x1 - x0))} x {int(round(y1 - y0))} px"
                )
            return

        if self._protractor_drag_index is not None:
            idx = self._protractor_drag_index
            self._protractor_drag_index = None
            self._protractor_status_var.set(f"Moved point {idx}")
            self._update_protractor_text()

    def _get_selected_spot_center(self, tracked: bool = True):
        center = super()._get_selected_spot_center(tracked=tracked)
        if center is not None:
            return center
        p1 = self._tracked_point(self._protractor_point1) if tracked else self._protractor_point1
        p2 = self._tracked_point(self._protractor_point2) if tracked else self._protractor_point2
        if p1 is not None and p2 is not None:
            return ((float(p1[0]) + float(p2[0])) * 0.5, (float(p1[1]) + float(p2[1])) * 0.5)
        if p1 is not None:
            return (float(p1[0]), float(p1[1]))
        if p2 is not None:
            return (float(p2[0]), float(p2[1]))
        return None

    def _protractor_angle_info(self) -> dict | None:
        p1 = self._tracked_point(self._protractor_point1)
        p2 = self._tracked_point(self._protractor_point2)
        if p1 is None or p2 is None:
            return None
        dx = float(p2[0]) - float(p1[0])
        dy = float(p2[1]) - float(p1[1])
        length_px = math.hypot(dx, dy)
        if length_px <= 0.0:
            return None
        angle_from_vertical_deg = math.degrees(math.atan2(dx, -dy))
        return {
            "point1_px": {"x": float(p1[0]), "y": float(p1[1])},
            "point2_px": {"x": float(p2[0]), "y": float(p2[1])},
            "dx_px": dx,
            "dy_px": dy,
            "length_px": length_px,
            "angle_from_vertical_deg": angle_from_vertical_deg,
            "angle_from_vertical_abs_deg": abs(angle_from_vertical_deg),
        }

    def _update_protractor_text(self) -> None:
        info = self._protractor_angle_info()
        if self._protractor_point1 is None and self._protractor_point2 is None:
            self._protractor_status_var.set("Set point 1")
            self._protractor_angle_var.set("Angle from vertical: -")
            return
        if info is None:
            if self._protractor_point1 is not None and self._protractor_point2 is None:
                self._protractor_status_var.set("Point 1 set. Set point 2")
            elif self._protractor_point2 is not None and self._protractor_point1 is None:
                self._protractor_status_var.set("Point 2 set. Set point 1")
            self._protractor_angle_var.set("Angle from vertical: -")
            return
        self._protractor_status_var.set(
            f"Length {info['length_px']:.1f} px | track={'on' if self._protractor_track_var.get() else 'off'}"
        )
        self._protractor_angle_var.set(
            f"Angle from vertical: {info['angle_from_vertical_deg']:+.2f} deg"
        )

    def _draw_protractor_overlay(self, draw: ImageDraw.ImageDraw, scale: float, off_x: int, off_y: int) -> None:
        p1 = self._tracked_point(self._protractor_point1)
        p2 = self._tracked_point(self._protractor_point2)
        if p1 is None and p2 is None:
            return
        r = 7
        if p1 is not None:
            x1 = int(round(off_x + float(p1[0]) * scale))
            y1 = int(round(off_y + float(p1[1]) * scale))
            draw.ellipse([x1 - r, y1 - r, x1 + r, y1 + r], outline=(255, 220, 0), width=2)
            draw.text((x1 + 8, y1 - 8), "1", fill=(255, 220, 0))
        if p2 is not None:
            x2 = int(round(off_x + float(p2[0]) * scale))
            y2 = int(round(off_y + float(p2[1]) * scale))
            draw.ellipse([x2 - r, y2 - r, x2 + r, y2 + r], outline=(255, 220, 0), width=2)
            draw.text((x2 + 8, y2 - 8), "2", fill=(255, 220, 0))
        if p1 is not None and p2 is not None:
            draw.line(
                [
                    (int(round(off_x + float(p1[0]) * scale)), int(round(off_y + float(p1[1]) * scale))),
                    (int(round(off_x + float(p2[0]) * scale)), int(round(off_y + float(p2[1]) * scale))),
                ],
                fill=(255, 220, 0),
                width=2,
            )

    def _draw_box_record_overlay(self, draw: ImageDraw.ImageDraw, scale: float, off_x: int, off_y: int) -> None:
        bounds = self._selected_box_bounds()
        if bounds is None:
            return
        x0, y0, x1, y1 = bounds
        dx0 = int(round(off_x + x0 * scale))
        dy0 = int(round(off_y + y0 * scale))
        dx1 = int(round(off_x + x1 * scale))
        dy1 = int(round(off_y + y1 * scale))
        draw.rectangle([dx0, dy0, dx1, dy1], outline=(0, 200, 255), width=2)
        draw.text((dx0 + 5, dy0 + 5), "record ROI", fill=(0, 200, 255))

    def _set_live_capture_busy(self, busy: bool) -> None:
        super()._set_live_capture_busy(busy)
        if self._box_record_btn is not None:
            self._box_record_btn.configure(state=(tk.DISABLED if busy else tk.NORMAL))

    def _live_capture_selected_box(self) -> None:
        if self._spotrec_running or self._spotrec_proc is not None:
            messagebox.showerror("Box recording", "Stop Spot examine recording before capture.")
            return
        cfg = self._live_capture_parse_config()
        if cfg is None:
            return
        bounds = self._selected_box_bounds()
        if bounds is None:
            messagebox.showerror("Box recording", "Draw a box on the live image first.")
            return
        try:
            roi_req = self._box_roi_from_bounds(bounds)
        except Exception as e:
            messagebox.showerror("Box recording", str(e))
            return

        with self._live_capture_lock:
            if self._live_capture_running:
                messagebox.showinfo("Box recording", "A live capture is already running.")
                return
            self._live_capture_running = True

        capture_exp_ms = float(cfg["exp_ms"])
        capture_gain_analog = float(cfg["gain_analog"])
        capture_gain_digital = float(cfg["gain_digital"])
        capture_fps_req = float(cfg["fps_req"])
        capture_n_frames = int(cfg["n_frames"])

        self._set_live_capture_busy(True)
        self._box_record_status_var.set("Stopping live feed and recording selected box...")
        self._stop_live_feed()

        def _worker() -> None:
            try:
                _, pending_dir, _ = self._stationary_dataset_paths()
                token = f"{time.strftime('%Y%m%d-%H%M%S')}_{time.time_ns()}"
                rec_id = f"field_roi_x{roi_req['x']}_y{roi_req['y']}_{token}"
                rec_dir = pending_dir / rec_id
                rec_dir.mkdir(parents=True, exist_ok=True)
                out_path = rec_dir / "box_recording.npy"
                script = Path(__file__).resolve().parent / "fetch_frames.py"

                def _run_fetch(req_fps: float | None) -> tuple[Path, float | None, object, object, object, object, object]:
                    args = [
                        sys.executable,
                        str(script),
                        "--out-dir",
                        str(rec_dir),
                        "--out-path",
                        str(out_path),
                        "--roi",
                        str(int(roi_req["x"])),
                        str(int(roi_req["y"])),
                        str(int(roi_req["w"])),
                        str(int(roi_req["h"])),
                        "--json",
                        "--n-frames",
                        str(int(capture_n_frames)),
                        "--stop-after",
                        str(int(capture_n_frames)),
                    ]
                    if req_fps is not None and float(req_fps) > 0.0:
                        args.extend(["--fps", str(float(req_fps))])
                    args.extend(["--exp-ms", str(float(capture_exp_ms))])
                    args.extend(["--gain-analog", str(float(capture_gain_analog))])
                    args.extend(["--gain-digital", str(float(capture_gain_digital))])
                    proc = subprocess.run(args, capture_output=True, text=True, check=False)
                    if proc.returncode != 0:
                        err = (proc.stderr or proc.stdout or "").strip()
                        raise RuntimeError(err or "Box recording failed.")
                    payload = (proc.stdout or "").strip().splitlines()
                    if not payload:
                        raise RuntimeError("Box recorder produced no output.")
                    data = json.loads(payload[-1])
                    path_l = Path(str(data.get("path", "")))
                    actual_fps_l = data.get("actual_fps")
                    actual_fps_l = float(actual_fps_l) if actual_fps_l is not None else None
                    return (
                        path_l,
                        actual_fps_l,
                        data.get("roi"),
                        data.get("timing"),
                        data.get("gains"),
                        data.get("max_raw_value"),
                        data.get("max_saved_value"),
                    )

                used_fps_fallback = False
                try:
                    path, actual_fps, actual_roi, timing, gains, max_raw, max_saved = _run_fetch(capture_fps_req)
                except Exception:
                    path, actual_fps, actual_roi, timing, gains, max_raw, max_saved = _run_fetch(None)
                    used_fps_fallback = True

                if not path.exists():
                    raise RuntimeError("Box recording file missing.")
                arr = np.load(path, mmap_mode="r", allow_pickle=False)
                frames_saved = int(arr.shape[0]) if getattr(arr, "ndim", 0) >= 3 else 0
                meta = {
                    "recording_id": rec_id,
                    "created_local": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "capture_type": "manual_rotation_box_roi",
                    "file": "box_recording.npy",
                    "requested": {
                        "frames": int(capture_n_frames),
                        "fps": float(capture_fps_req),
                        "exp_ms": float(capture_exp_ms),
                        "gain_analog": float(capture_gain_analog),
                        "gain_digital": float(capture_gain_digital),
                        "roi": dict(roi_req),
                    },
                    "actual": {
                        "frames": int(frames_saved),
                        "fps": actual_fps,
                        "roi": actual_roi,
                        "timing": timing,
                        "gains": gains,
                        "max_raw_value": max_raw,
                        "max_saved_value": max_saved,
                        "used_fps_fallback": bool(used_fps_fallback),
                    },
                }
                self._write_json_atomic(rec_dir / "meta.json", meta)
                msg = (
                    f"Saved box recording: {rec_id}\n"
                    f"frames={frames_saved}, fps={actual_fps or 0.0:.1f}\n"
                    "Open Pick box spots and load saved recordings when ready."
                )
                self._ui_call(self._box_record_status_var.set, msg)
                self._ui_call(self.bottom_var.set, msg)
            except Exception as e:
                self._ui_call(self._box_record_status_var.set, f"Box recording failed: {e}")
                self._ui_call(messagebox.showerror, "Box recording", str(e))
            finally:
                self._ui_call(self._start_live_feed)
                self._ui_call(self._set_live_capture_busy, False)
                with self._live_capture_lock:
                    self._live_capture_running = False

        threading.Thread(target=_worker, daemon=True).start()

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

        if frame is not None and self._live_img_label is not None:
            try:
                self._update_live_tracking(frame)
                img = Image.fromarray(frame)
                try:
                    resample = Image.Resampling.BILINEAR
                    zoom_resample = Image.Resampling.NEAREST
                except Exception:
                    resample = Image.BILINEAR
                    zoom_resample = Image.NEAREST
                w = int(self._live_img_label.winfo_width())
                h = int(self._live_img_label.winfo_height())
                if w > 10 and h > 10:
                    src_w, src_h = img.size
                    scale = min(float(w) / float(src_w), float(h) / float(src_h))
                    disp_w = max(1, int(round(src_w * scale)))
                    disp_h = max(1, int(round(src_h * scale)))
                    img = img.resize((disp_w, disp_h), resample=resample)
                    self._live_disp_scale = scale
                    off_x = max(0, int((w - disp_w) // 2))
                    off_y = max(0, int((h - disp_h) // 2))
                    self._live_disp_offset = (off_x, off_y)

                    mag_on = bool(self._live_mag_enabled_var.get())
                    canvas = Image.new("RGB", (w, h), 0)
                    canvas.paste(img.convert("RGB"), (off_x, off_y))

                    draw = ImageDraw.Draw(canvas)
                    self._draw_protractor_overlay(draw, scale, off_x, off_y)
                    self._draw_box_record_overlay(draw, scale, off_x, off_y)

                    spot_xy = super()._get_selected_spot_center(tracked=True)
                    if spot_xy is not None:
                        try:
                            cx, cy = spot_xy
                            if 0.0 <= cx < float(src_w) and 0.0 <= cy < float(src_h):
                                ring_r = 8
                                px = int(round(off_x + cx * scale))
                                py = int(round(off_y + cy * scale))
                                draw.ellipse(
                                    [px - ring_r, py - ring_r, px + ring_r, py + ring_r],
                                    outline=(0, 255, 0),
                                    width=2,
                                )
                        except Exception:
                            pass

                    if mag_on:
                        z = self._parse_float(self._live_zoom_var.get())
                        zoom = float(z) if z and z > 0.1 else 1.0
                        zoom = max(1.0, zoom)
                        out_sz = int(self._live_zoom_output_px)
                        win = max(8, int(round(float(out_sz) / float(zoom))))
                        win = min(win, int(src_w), int(src_h))
                        win = max(1, int(win))
                        cx, cy = None, None
                        if self._live_zoom_center is not None:
                            cx, cy = self._live_zoom_center
                        if cx is None or cy is None:
                            cx = src_w / 2.0
                            cy = src_h / 2.0
                        if self._live_zoom_center is not None:
                            self._live_theta_var.set(
                                self._live_theta_text_for_center(
                                    gray=frame,
                                    center=(float(cx), float(cy)),
                                )
                            )
                        self._live_update_xy_preview(frame=frame, center=(float(cx), float(cy)))
                        half = win // 2
                        x0 = int(round(cx)) - half
                        y0 = int(round(cy)) - half
                        x0 = max(0, min(int(src_w) - win, x0))
                        y0 = max(0, min(int(src_h) - win, y0))
                        x1 = x0 + win
                        y1 = y0 + win

                        dx0 = int(round(off_x + x0 * scale))
                        dy0 = int(round(off_y + y0 * scale))
                        dx1 = int(round(off_x + x1 * scale))
                        dy1 = int(round(off_y + y1 * scale))
                        draw.rectangle([dx0, dy0, dx1, dy1], outline=(255, 0, 0), width=2)

                        if self._live_zoom_label is not None:
                            crop_arr = frame[y0:y1, x0:x1]
                            crop = Image.fromarray(crop_arr)
                            zoom_img = crop.resize((out_sz, out_sz), resample=zoom_resample)
                            try:
                                marker_x = ((float(cx) - float(x0)) / max(1.0, float(win))) * float(out_sz)
                                marker_y = ((float(cy) - float(y0)) / max(1.0, float(win))) * float(out_sz)
                                zx = int(round(marker_x))
                                zy = int(round(marker_y))
                                draw_zoom = ImageDraw.Draw(zoom_img)
                                roi_raw_f = self._parse_float(self._live_capture_roi_var.get())
                                if roi_raw_f is not None and roi_raw_f > 0.0:
                                    roi_raw = max(2, int(round(float(roi_raw_f))))
                                    if (roi_raw % 2) != 0:
                                        roi_raw -= 1
                                    roi_half_px = 0.5 * float(roi_raw)
                                    roi_left = ((float(cx) - roi_half_px) - float(x0)) / max(1.0, float(win))
                                    roi_top = ((float(cy) - roi_half_px) - float(y0)) / max(1.0, float(win))
                                    roi_right = ((float(cx) + roi_half_px) - float(x0)) / max(1.0, float(win))
                                    roi_bottom = ((float(cy) + roi_half_px) - float(y0)) / max(1.0, float(win))
                                    rx0 = int(round(roi_left * float(out_sz)))
                                    ry0 = int(round(roi_top * float(out_sz)))
                                    rx1 = int(round(roi_right * float(out_sz)))
                                    ry1 = int(round(roi_bottom * float(out_sz)))
                                    draw_zoom.rectangle([rx0, ry0, rx1, ry1], outline=(255, 0, 0), width=1)
                                cross_r = 4
                                draw_zoom.line([(zx - cross_r, zy), (zx + cross_r, zy)], fill=(255, 0, 0), width=1)
                                draw_zoom.line([(zx, zy - cross_r), (zx, zy + cross_r)], fill=(255, 0, 0), width=1)
                            except Exception:
                                pass
                            zoom_photo = ImageTk.PhotoImage(zoom_img)
                            self._live_zoom_label.configure(image=zoom_photo)
                            self._live_zoom_label.image = zoom_photo
                            if self._live_hist_label is not None:
                                hist_photo = self._render_live_hist_image(crop_arr)
                                if hist_photo is not None:
                                    self._live_hist_label.configure(image=hist_photo)
                                    self._live_hist_ref = hist_photo
                    else:
                        if self._live_zoom_label is not None and self._live_zoom_blank_ref is not None:
                            self._live_zoom_label.configure(image=self._live_zoom_blank_ref)
                        if self._live_hist_label is not None and self._live_hist_blank_ref is not None:
                            self._live_hist_label.configure(image=self._live_hist_blank_ref)
                            self._live_hist_ref = self._live_hist_blank_ref

                    self._update_protractor_text()
                    img = canvas

                photo = ImageTk.PhotoImage(img)
                self._live_img_label.configure(image=photo)
                self._live_img_ref = photo
            except Exception:
                pass

        self._live_after_id = self.root.after(50, self._live_tick)

    def _live_capture_stationary_from_magnifier(self) -> None:
        if self._spotrec_running or self._spotrec_proc is not None:
            messagebox.showerror("Live capture", "Stop Spot examine recording before capture.")
            return
        cfg = self._live_capture_parse_config()
        if cfg is None:
            return
        with self._live_capture_lock:
            if self._live_capture_running:
                messagebox.showinfo("Live capture", "A live stationary capture is already running.")
                return
            self._live_capture_running = True

        frame = self._live_last_frame
        if frame is None or getattr(frame, "ndim", 0) != 2:
            with self._live_capture_lock:
                self._live_capture_running = False
            messagebox.showerror("Live capture", "No live frame available for capture.")
            return

        if self._live_zoom_center is not None:
            cx = float(self._live_zoom_center[0])
            cy = float(self._live_zoom_center[1])
        else:
            src_h, src_w = frame.shape
            cx = float(src_w) / 2.0
            cy = float(src_h) / 2.0
        center = (float(cx), float(cy))
        protractor_info = self._protractor_angle_info()

        capture_exp_ms = float(cfg["exp_ms"])
        capture_gain_analog = float(cfg["gain_analog"])
        capture_gain_digital = float(cfg["gain_digital"])
        capture_fps_req = float(cfg["fps_req"])
        capture_n_frames = int(cfg["n_frames"])
        capture_duration_s = float(cfg["duration_s"])
        capture_roi_raw = int(cfg["roi_raw"])
        prev_live_exp = str(self._live_exp_ms_var.get())
        prev_live_gain = str(self._live_gain_var.get())
        prev_stationary_exp = str(self._stationary_capture_max_exp_ms_var.get())
        prev_stationary_dur = str(self._stationary_capture_max_duration_s_var.get())
        prev_stationary_fps = str(self._stationary_capture_max_fps_est_var.get())

        self._live_exp_ms_var.set(f"{capture_exp_ms:.3f}")
        self._live_gain_var.set(f"{capture_gain_analog:.2f}")
        self._stationary_capture_max_exp_ms_var.set(f"{capture_exp_ms:.3f}")
        self._stationary_capture_max_duration_s_var.set(f"{capture_duration_s:.3f}")
        self._stationary_capture_max_fps_est_var.set(f"{capture_fps_req:.1f}")

        self._set_live_capture_busy(True)
        self._live_capture_status_var.set(
            "Stopping live feed and recording magnifier with explicit capture settings..."
        )
        self._stop_live_feed()

        def _worker() -> None:
            try:
                _, pending_dir, _ = self._stationary_dataset_paths()
                key = self._spot_center_key(center)
                token = f"{time.strftime('%Y%m%d-%H%M%S')}_{time.time_ns()}"
                rod_id = f"rod_x{key[0]}_y{key[1]}_{token}"
                rod_dir = pending_dir / rod_id
                rod_dir.mkdir(parents=True, exist_ok=True)
                mode_path = rod_dir / "capture_maxfps_15x15.npy"

                mode_max = self._capture_stationary_mode(
                    center=center,
                    out_path=mode_path,
                    roi_raw=int(capture_roi_raw),
                    n_frames=int(capture_n_frames),
                    exp_ms=float(capture_exp_ms),
                    gain_analog=float(capture_gain_analog),
                    gain_digital=float(capture_gain_digital),
                    requested_fps=float(capture_fps_req),
                    mode_name="maxfps_15x15",
                    requested_duration_s=float(capture_duration_s),
                )
                xy_series_cap = [
                    (float(v[0]), float(v[1]))
                    for v in (mode_max.get("xy_series") or [])
                    if isinstance(v, (list, tuple)) and len(v) >= 2
                ]

                theta_est = dict(mode_max.get("theta_estimate", {}) or {})
                rod_meta = {
                    "rod_id": rod_id,
                    "created_local": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "capture_type": "stationary_rod_live_single_mode",
                    "center_px": {"x": float(center[0]), "y": float(center[1])},
                    "candidate_metrics": None,
                    "protractor": protractor_info,
                    "modes": {
                        "capture_maxfps_15x15_meta": "capture_maxfps_15x15_meta.json",
                        "capture_maxfps_15x15_summary": {
                            "actual_fps": mode_max.get("actual", {}).get("fps"),
                            "frames": mode_max.get("actual", {}).get("frames"),
                            "duration_s": mode_max.get("actual", {}).get("duration_s"),
                            "fov_raw_px": mode_max.get("requested", {}).get("fov_raw_px"),
                            "r_mean": mode_max.get("xy_metrics", {}).get("r_mean"),
                            "range_x": mode_max.get("xy_metrics", {}).get("range_x"),
                            "range_y": mode_max.get("xy_metrics", {}).get("range_y"),
                        },
                    },
                    "theta_active_medium": theta_est.get("theta_active_label"),
                    "theta_active_deg": theta_est.get("theta_active_deg"),
                    "theta_water_deg": theta_est.get("theta_water_deg"),
                    "theta_glycerol50_deg": theta_est.get("theta_glycerol50_deg"),
                }
                self._write_json_atomic(rod_dir / "meta.json", rod_meta)
                self._ui_call(self._live_set_captured_xy_freeze, xy_series_cap, center)

                status_lines = [f"Saved to pending: {rod_id}"]
                if protractor_info is not None:
                    status_lines.append(
                        f"angle={float(protractor_info['angle_from_vertical_deg']):+.2f} deg"
                    )
                status_txt = "\n".join(status_lines)
                self._ui_call(self._live_capture_status_var.set, status_txt)
                self._ui_call(self.bottom_var.set, status_txt)
                self._ui_call(self._stationary_review_refresh, True)
            except Exception as e:
                self._ui_call(self._live_capture_status_var.set, f"Live capture failed: {e}")
                self._ui_call(messagebox.showerror, "Live capture", str(e))
            finally:
                self._ui_call(self._live_exp_ms_var.set, prev_live_exp)
                self._ui_call(self._live_gain_var.set, prev_live_gain)
                self._ui_call(self._stationary_capture_max_exp_ms_var.set, prev_stationary_exp)
                self._ui_call(self._stationary_capture_max_duration_s_var.set, prev_stationary_dur)
                self._ui_call(self._stationary_capture_max_fps_est_var.set, prev_stationary_fps)
                self._ui_call(self._start_live_feed)
                self._ui_call(self._set_live_capture_busy, False)
                with self._live_capture_lock:
                    self._live_capture_running = False

        threading.Thread(target=_worker, daemon=True).start()


def main() -> None:
    root = tk.Tk()
    root.title("Manual Rotation")
    root.geometry("1500x950")
    ManualRotationApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
