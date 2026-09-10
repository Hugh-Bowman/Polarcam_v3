from __future__ import annotations

import re
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import matplotlib
import numpy as np
from PIL import Image
import cv2

matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

import Detection_alg_offline as detect_spinners
from pol_reconstruction import make_qu_reconstructor


def resolve_default_widefield_dir() -> Path:
    script_path = Path(__file__).resolve().parent / "Spinners_gui_live.py"
    widefield_dirname = "widefield_frames"
    try:
        text = script_path.read_text(encoding="utf-8")
        m = re.search(r'RECORDINGS_WIDEFIELD_DIRNAME\s*=\s*"([^"]+)"', text)
        if m:
            widefield_dirname = m.group(1)
    except Exception:
        pass
    today_str = np.datetime64("today").astype(str)
    return Path(__file__).resolve().parent / "recordings" / today_str / widefield_dirname


def stretch_to_u8(frame: np.ndarray, low: float, high: float) -> np.ndarray:
    arr = np.asarray(frame, dtype=np.float32)
    if not np.isfinite(low):
        low = float(np.nanmin(arr))
    if not np.isfinite(high):
        high = float(np.nanmax(arr))
    if high <= low:
        high = low + 1.0
    scaled = (arr - low) / (high - low)
    scaled = np.clip(scaled, 0.0, 1.0)
    return (scaled * 255.0).astype(np.uint8)


class HistogramStretchControls:
    def __init__(
        self,
        parent: tk.Widget,
        on_change,
        title: str,
    ) -> None:
        self.on_change = on_change
        self._updating = False
        self.data_min = 0.0
        self.data_max = 1.0

        frame = ttk.LabelFrame(parent, text=title)
        frame.pack(fill=tk.X, padx=8, pady=6)
        self.frame = frame

        row1 = ttk.Frame(frame)
        row1.pack(fill=tk.X, padx=8, pady=(6, 2))
        row2 = ttk.Frame(frame)
        row2.pack(fill=tk.X, padx=8, pady=(0, 6))

        self.low_var = tk.DoubleVar(value=0.0)
        self.high_var = tk.DoubleVar(value=1.0)
        self.low_label_var = tk.StringVar(value="Low: 0")
        self.high_label_var = tk.StringVar(value="High: 1")

        ttk.Label(row1, textvariable=self.low_label_var, width=16).pack(side=tk.LEFT)
        self.low_scale = tk.Scale(
            row1,
            orient=tk.HORIZONTAL,
            from_=0.0,
            to=1.0,
            resolution=1.0,
            showvalue=False,
            variable=self.low_var,
            command=self._on_scale_change,
        )
        self.low_scale.pack(side=tk.LEFT, fill=tk.X, expand=True)

        ttk.Label(row2, textvariable=self.high_label_var, width=16).pack(side=tk.LEFT)
        self.high_scale = tk.Scale(
            row2,
            orient=tk.HORIZONTAL,
            from_=0.0,
            to=1.0,
            resolution=1.0,
            showvalue=False,
            variable=self.high_var,
            command=self._on_scale_change,
        )
        self.high_scale.pack(side=tk.LEFT, fill=tk.X, expand=True)

    def set_data_range(self, data_min: float, data_max: float, reset: bool = False) -> None:
        self.data_min = float(data_min)
        self.data_max = float(data_max)
        if self.data_max <= self.data_min:
            self.data_max = self.data_min + 1.0
        self._updating = True
        try:
            self.low_scale.configure(from_=self.data_min, to=self.data_max, resolution=1.0)
            self.high_scale.configure(from_=self.data_min, to=self.data_max, resolution=1.0)
            if reset:
                self.low_var.set(self.data_min)
                self.high_var.set(self.data_max)
            else:
                self.low_var.set(min(max(self.low_var.get(), self.data_min), self.data_max))
                self.high_var.set(min(max(self.high_var.get(), self.data_min), self.data_max))
        finally:
            self._updating = False
        self._update_labels()

    def _update_labels(self) -> None:
        self.low_label_var.set(f"Low: {self.low_var.get():.0f}")
        self.high_label_var.set(f"High: {self.high_var.get():.0f}")

    def _on_scale_change(self, _value) -> None:
        if self._updating:
            return
        low = float(self.low_var.get())
        high = float(self.high_var.get())
        if low >= high:
            if self.low_scale == self.frame.focus_get():
                high = low + 1.0
                self.high_var.set(high)
            else:
                low = high - 1.0
                self.low_var.set(low)
        low = max(low, self.data_min)
        high = min(high, self.data_max)
        if low >= high:
            high = low + 1.0
            self.high_var.set(high)
        self._update_labels()
        self.on_change()

    def values(self) -> tuple[float, float]:
        low = float(self.low_var.get())
        high = float(self.high_var.get())
        if high <= low:
            high = low + 1.0
        return low, high


class FrameAverageReviewer:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Manual Frame Average Reviewer")
        self.default_dir = resolve_default_widefield_dir()

        self.frames: np.ndarray | None = None
        self.frame_status: list[bool | None] = []
        self.current_index = 0
        self.current_path: Path | None = None
        self.current_average: np.ndarray | None = None
        self.current_average_count = 0
        self.current_s_map: np.ndarray | None = None

        self.status_var = tk.StringVar(value="Load an NPY stack to begin.")
        self.frame_info_var = tk.StringVar(value="No file loaded")
        self.good_count_var = tk.StringVar(value="Good: 0 / 0")

        self._build_ui()
        self.root.bind("<Left>", lambda _e: self.prev_frame())
        self.root.bind("<Right>", lambda _e: self.next_frame())
        self.root.bind("y", lambda _e: self.mark_current(True))
        self.root.bind("n", lambda _e: self.mark_current(False))

    def _build_ui(self) -> None:
        top = ttk.Frame(self.root)
        top.pack(fill=tk.X, padx=8, pady=8)

        ttk.Button(top, text="Load NPY", command=self.load_npy).pack(side=tk.LEFT)
        ttk.Label(top, textvariable=self.frame_info_var).pack(side=tk.LEFT, padx=(12, 0))
        ttk.Label(top, textvariable=self.good_count_var).pack(side=tk.RIGHT)

        notebook = ttk.Notebook(self.root)
        notebook.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0, 8))

        review_tab = ttk.Frame(notebook)
        avg_tab = ttk.Frame(notebook)
        smap_tab = ttk.Frame(notebook)
        notebook.add(review_tab, text="Frame Review")
        notebook.add(avg_tab, text="Average Image")
        notebook.add(smap_tab, text="S_map")

        self._build_review_tab(review_tab)
        self._build_average_tab(avg_tab)
        self._build_smap_tab(smap_tab)

        bottom = ttk.Label(self.root, textvariable=self.status_var, anchor="w")
        bottom.pack(fill=tk.X, padx=8, pady=(0, 8))

    def _build_review_tab(self, parent: ttk.Frame) -> None:
        left = ttk.Frame(parent)
        right = ttk.Frame(parent)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        right.pack(side=tk.RIGHT, fill=tk.Y)

        self.review_fig = Figure(figsize=(6.4, 5.8), dpi=100)
        self.review_ax_img = self.review_fig.add_subplot(211)
        self.review_ax_hist = self.review_fig.add_subplot(212)
        self.review_canvas = FigureCanvasTkAgg(self.review_fig, master=left)
        self.review_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        nav = ttk.Frame(right)
        nav.pack(fill=tk.X, padx=8, pady=(8, 4))
        ttk.Button(nav, text="Prev", command=self.prev_frame).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(nav, text="Next", command=self.next_frame).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(6, 0))

        yn = ttk.Frame(right)
        yn.pack(fill=tk.X, padx=8, pady=4)
        ttk.Button(yn, text="Yes / keep", command=lambda: self.mark_current(True)).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(yn, text="No / exclude", command=lambda: self.mark_current(False)).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(6, 0))

        ttk.Label(
            right,
            text="Frames start as good. Keyboard: left/right to move, y to keep, n to exclude",
            wraplength=220,
            justify=tk.LEFT,
        ).pack(fill=tk.X, padx=8, pady=(2, 8))

        self.review_controls = HistogramStretchControls(
            right,
            on_change=self.refresh_review_display,
            title="Display stretch",
        )

    def _build_average_tab(self, parent: ttk.Frame) -> None:
        left = ttk.Frame(parent)
        right = ttk.Frame(parent)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        right.pack(side=tk.RIGHT, fill=tk.Y)

        self.avg_fig = Figure(figsize=(6.4, 5.8), dpi=100)
        self.avg_ax_img = self.avg_fig.add_subplot(211)
        self.avg_ax_hist = self.avg_fig.add_subplot(212)
        self.avg_canvas = FigureCanvasTkAgg(self.avg_fig, master=left)
        self.avg_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        ttk.Button(right, text="Save average image", command=self.save_average_image).pack(
            fill=tk.X, padx=8, pady=(8, 4)
        )
        ttk.Label(
            right,
            text="The average updates as frames are moved into or out of the good pile.",
            wraplength=220,
            justify=tk.LEFT,
        ).pack(fill=tk.X, padx=8, pady=(2, 8))

        self.avg_controls = HistogramStretchControls(
            right,
            on_change=self.refresh_average_display,
            title="Display stretch",
        )

    def _build_smap_tab(self, parent: ttk.Frame) -> None:
        left = ttk.Frame(parent)
        right = ttk.Frame(parent)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        right.pack(side=tk.RIGHT, fill=tk.Y)

        self.smap_fig = Figure(figsize=(6.4, 5.8), dpi=100)
        self.smap_ax_img = self.smap_fig.add_subplot(211)
        self.smap_ax_hist = self.smap_fig.add_subplot(212)
        self.smap_canvas = FigureCanvasTkAgg(self.smap_fig, master=left)
        self.smap_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        ttk.Label(
            right,
            text="S_map is recomputed from the currently included frames using the same range metric as Spinners_gui_live.",
            wraplength=220,
            justify=tk.LEFT,
        ).pack(fill=tk.X, padx=8, pady=(8, 8))

        self.smap_controls = HistogramStretchControls(
            right,
            on_change=self.refresh_smap_display,
            title="Display stretch",
        )

    def load_npy(self) -> None:
        start_dir = self.default_dir if self.default_dir.exists() else Path.cwd()
        path = filedialog.askopenfilename(
            parent=self.root,
            title="Choose NPY stack",
            initialdir=str(start_dir),
            filetypes=[("NumPy stack", "*.npy"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            arr = np.load(path, allow_pickle=False)
            if arr.ndim == 2:
                arr = arr[np.newaxis, ...]
            if arr.ndim == 4 and arr.shape[-1] == 1:
                arr = arr[..., 0]
            if arr.ndim != 3:
                raise ValueError(f"Expected 3D stack, got shape {arr.shape}")
            self.frames = np.asarray(arr)
        except Exception as exc:
            messagebox.showerror("Load failed", f"Could not load NPY file.\n\n{exc}")
            return

        self.current_path = Path(path)
        self.current_index = 0
        self.frame_status = [True] * int(self.frames.shape[0])
        self.current_average = None
        self.current_average_count = 0
        self.current_s_map = None

        data_min = float(np.min(self.frames))
        data_max = float(np.max(self.frames))
        self.review_controls.set_data_range(data_min, data_max, reset=True)
        self.avg_controls.set_data_range(data_min, data_max, reset=True)
        self.recompute_average()
        self.recompute_s_map()
        self.refresh_all()
        self.status_var.set(f"Loaded {self.current_path.name}")

    def current_frame(self) -> np.ndarray | None:
        if self.frames is None:
            return None
        return self.frames[self.current_index]

    def refresh_all(self) -> None:
        self.refresh_review_display()
        self.refresh_average_display()
        self.refresh_smap_display()
        self.update_labels()

    def update_labels(self) -> None:
        total = 0 if self.frames is None else int(self.frames.shape[0])
        good = sum(1 for v in self.frame_status if v is True)
        self.good_count_var.set(f"Good: {good} / {total}")
        if self.frames is None:
            self.frame_info_var.set("No file loaded")
            return
        state = self.frame_status[self.current_index]
        state_txt = "GOOD"
        if state is True:
            state_txt = "GOOD"
        elif state is False:
            state_txt = "BAD"
        self.frame_info_var.set(
            f"{self.current_path.name} | frame {self.current_index + 1}/{total} | {state_txt}"
        )

    def refresh_review_display(self) -> None:
        self.review_ax_img.clear()
        self.review_ax_hist.clear()
        frame = self.current_frame()
        if frame is None:
            self.review_ax_img.set_title("No frame loaded")
            self.review_canvas.draw_idle()
            return

        low, high = self.review_controls.values()
        frame_u8 = stretch_to_u8(frame, low, high)
        self.review_ax_img.imshow(frame_u8, cmap="gray", vmin=0, vmax=255)
        self.review_ax_img.set_title("Current frame")
        self.review_ax_img.set_axis_off()

        flat = np.asarray(frame, dtype=np.float32).ravel()
        self.review_ax_hist.hist(flat, bins=128, color="0.3")
        self.review_ax_hist.axvline(low, color="tab:orange", linestyle="--", linewidth=1.5)
        self.review_ax_hist.axvline(high, color="tab:orange", linestyle="--", linewidth=1.5)
        self.review_ax_hist.set_title("Brightness histogram for display stretch")
        self.review_ax_hist.set_xlabel("Pixel value")
        self.review_ax_hist.set_ylabel("Count")
        self.review_fig.tight_layout()
        self.review_canvas.draw_idle()
        self.update_labels()

    def refresh_average_display(self) -> None:
        self.avg_ax_img.clear()
        self.avg_ax_hist.clear()

        if self.current_average is None or self.current_average_count <= 0:
            self.avg_ax_img.set_title("No good frames selected")
            self.avg_ax_img.set_axis_off()
            self.avg_canvas.draw_idle()
            return

        low, high = self.avg_controls.values()
        avg_u8 = stretch_to_u8(self.current_average, low, high)
        self.avg_ax_img.imshow(avg_u8, cmap="gray", vmin=0, vmax=255)
        self.avg_ax_img.set_title(f"Average image from {self.current_average_count} good frames")
        self.avg_ax_img.set_axis_off()

        flat = np.asarray(self.current_average, dtype=np.float32).ravel()
        self.avg_ax_hist.hist(flat, bins=128, color="0.3")
        self.avg_ax_hist.axvline(low, color="tab:orange", linestyle="--", linewidth=1.5)
        self.avg_ax_hist.axvline(high, color="tab:orange", linestyle="--", linewidth=1.5)
        self.avg_ax_hist.set_title("Average image histogram for display stretch")
        self.avg_ax_hist.set_xlabel("Pixel value")
        self.avg_ax_hist.set_ylabel("Count")
        self.avg_fig.tight_layout()
        self.avg_canvas.draw_idle()

    def refresh_smap_display(self) -> None:
        self.smap_ax_img.clear()
        self.smap_ax_hist.clear()

        if self.current_s_map is None:
            self.smap_ax_img.set_title("No S_map available")
            self.smap_ax_img.set_axis_off()
            self.smap_canvas.draw_idle()
            return

        low, high = self.smap_controls.values()
        smap_u8 = stretch_to_u8(self.current_s_map, low, high)
        self.smap_ax_img.imshow(smap_u8, cmap="gray", vmin=0, vmax=255)
        self.smap_ax_img.set_title("S_map from included frames")
        self.smap_ax_img.set_axis_off()

        flat = np.asarray(self.current_s_map, dtype=np.float32).ravel()
        self.smap_ax_hist.hist(flat, bins=128, color="0.3")
        self.smap_ax_hist.axvline(low, color="tab:orange", linestyle="--", linewidth=1.5)
        self.smap_ax_hist.axvline(high, color="tab:orange", linestyle="--", linewidth=1.5)
        self.smap_ax_hist.set_title("S_map histogram for display stretch")
        self.smap_ax_hist.set_xlabel("S value")
        self.smap_ax_hist.set_ylabel("Count")
        self.smap_fig.tight_layout()
        self.smap_canvas.draw_idle()

    def prev_frame(self) -> None:
        if self.frames is None:
            return
        self.current_index = (self.current_index - 1) % int(self.frames.shape[0])
        self.refresh_review_display()

    def next_frame(self) -> None:
        if self.frames is None:
            return
        self.current_index = (self.current_index + 1) % int(self.frames.shape[0])
        self.refresh_review_display()

    def mark_current(self, is_good: bool) -> None:
        if self.frames is None:
            return
        self.frame_status[self.current_index] = is_good
        self.recompute_average()
        self.recompute_s_map()
        self.refresh_all()
        self.next_frame()

    def recompute_average(self) -> None:
        if self.frames is None:
            self.current_average = None
            self.current_average_count = 0
            return
        idx = [i for i, good in enumerate(self.frame_status) if good is True]
        self.current_average_count = len(idx)
        if not idx:
            self.current_average = None
            return
        self.current_average = np.mean(self.frames[idx], axis=0)

    def recompute_s_map(self) -> None:
        if self.frames is None:
            self.current_s_map = None
            return
        idx = [i for i, good in enumerate(self.frame_status) if good is True]
        if len(idx) < 2:
            self.current_s_map = None
            return

        selected = self.frames[idx[: max(2, int(detect_spinners.S_MAP_FRAMES))]]
        shape = tuple(int(v) for v in selected[0].shape)
        recon = make_qu_reconstructor(shape, out_dtype=np.float32)
        min_x = max_x = None
        min_y = max_y = None
        x_sm = y_sm = None

        for frame in selected:
            x_raw, y_raw = recon(frame)
            if x_sm is None:
                x_sm = np.empty_like(x_raw)
                y_sm = np.empty_like(y_raw)
            cv2.boxFilter(
                x_raw,
                ddepth=-1,
                ksize=(detect_spinners.S_MAP_SMOOTH_K, detect_spinners.S_MAP_SMOOTH_K),
                dst=x_sm,
                normalize=True,
                borderType=cv2.BORDER_REPLICATE,
            )
            cv2.boxFilter(
                y_raw,
                ddepth=-1,
                ksize=(detect_spinners.S_MAP_SMOOTH_K, detect_spinners.S_MAP_SMOOTH_K),
                dst=y_sm,
                normalize=True,
                borderType=cv2.BORDER_REPLICATE,
            )

            if min_x is None:
                min_x = x_sm.copy()
                max_x = x_sm.copy()
                min_y = y_sm.copy()
                max_y = y_sm.copy()
            else:
                np.minimum(min_x, x_sm, out=min_x)
                np.maximum(max_x, x_sm, out=max_x)
                np.minimum(min_y, y_sm, out=min_y)
                np.maximum(max_y, y_sm, out=max_y)

        self.current_s_map = self._anisotropy_range_s_map(min_x, max_x, min_y, max_y, shape)[0]
        smap_min = float(np.min(self.current_s_map))
        smap_max = float(np.max(self.current_s_map))
        self.smap_controls.set_data_range(smap_min, smap_max, reset=True)

    @staticmethod
    def _anisotropy_range_s_map(
        min_x: np.ndarray,
        max_x: np.ndarray,
        min_y: np.ndarray,
        max_y: np.ndarray,
        raw_shape: tuple[int, int],
    ) -> tuple[np.ndarray, np.ndarray]:
        rx = (max_x.astype(np.float32) - min_x.astype(np.float32))
        ry = (max_y.astype(np.float32) - min_y.astype(np.float32))
        s_int = (rx * rx) + (ry * ry)

        h, w = raw_shape
        si_h, si_w = s_int.shape
        if (si_h, si_w) == (h, w):
            s_full = s_int
        elif (si_h, si_w) == (h - 1, w - 1):
            s_full = np.pad(s_int, ((0, 1), (0, 1)), mode="edge")
        else:
            s_full = cv2.resize(s_int, (w, h), interpolation=cv2.INTER_LINEAR)
        return (s_full.astype(np.float32, copy=False), s_int.astype(np.float32, copy=False))

    def save_average_image(self) -> None:
        if self.current_average is None or self.current_average_count <= 0:
            messagebox.showinfo("Nothing to save", "No good frames have been selected yet.")
            return
        initial_dir = str(self.current_path.parent if self.current_path is not None else self.default_dir)
        initial_name = (
            f"{self.current_path.stem}_average_good.npy" if self.current_path is not None else "average_good.npy"
        )
        path = filedialog.asksaveasfilename(
            parent=self.root,
            title="Save average image",
            initialdir=initial_dir,
            initialfile=initial_name,
            defaultextension=".npy",
            filetypes=[
                ("NumPy array", "*.npy"),
                ("PNG image", "*.png"),
                ("TIFF image", "*.tif *.tiff"),
                ("All files", "*.*"),
            ],
        )
        if not path:
            return
        out_path = Path(path)
        suffix = out_path.suffix.lower()
        if suffix == ".npy":
            np.save(out_path, self.current_average)
        elif suffix in {".png", ".tif", ".tiff"}:
            low, high = self.avg_controls.values()
            img = stretch_to_u8(self.current_average, low, high)
            Image.fromarray(img).save(out_path)
        else:
            np.save(out_path.with_suffix(".npy"), self.current_average)
            out_path = out_path.with_suffix(".npy")
        self.status_var.set(f"Saved average image to {out_path}")


def main() -> None:
    root = tk.Tk()
    root.geometry("1180x860")
    FrameAverageReviewer(root)
    root.mainloop()


if __name__ == "__main__":
    main()
