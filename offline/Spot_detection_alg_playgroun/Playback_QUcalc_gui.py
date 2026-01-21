# pol_basic_player_min_throttled_display_with_qu_single_decode_process_all.py
import time
import threading
import queue
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import cv2
import numpy as np
from PIL import Image, ImageTk

import detect_spinners
from pol_reconstruction import make_qu_reconstructor


def _to_gray_u8(frame: np.ndarray) -> np.ndarray:
    """
    OpenCV decodes your AVI into BGR with replicated grayscale.
    extractChannel gives a contiguous (H,W) uint8 array.
    """
    if frame is None:
        return None
    if frame.ndim == 3:
        return cv2.extractChannel(frame, 0)  # contiguous
    return frame.astype(np.uint8, copy=False)


class BasicVideoPlayer:
    def _show_st2_popup(self, st2_med: np.ndarray):
        """
        Pop up a window showing the median S_t^2 image.
        """
        try:
            u8 = detect_spinners.to_u8_preview(st2_med, lo_pct=0.0, hi_pct=100.0)
        except Exception as e:
            messagebox.showerror("Spinner detect error", str(e))
            return

        win = tk.Toplevel(self.root)
        win.title("S_map from first 10 frames")

        img_rgb = Image.fromarray(u8, mode="L").convert("RGB")
        if centers:
            draw = ImageDraw.Draw(img_rgb)
            radius = 6
            for cx, cy in centers:
                x0, y0 = cx - radius, cy - radius
                x1, y1 = cx + radius, cy + radius
                draw.ellipse([x0, y0, x1, y1], outline=(255, 0, 0), width=2)

        try:
            out_path = Path.cwd() / "S_map_spots.png"
            img_rgb.save(out_path)
        except Exception as e:
            messagebox.showwarning("Save image warning", f"Could not save S_map image: {e}")

        if self._overlay_base_frame is not None:
            try:
                raw_rgb = Image.fromarray(self._overlay_base_frame, mode="L").convert("RGB")
                if centers:
                    draw_raw = ImageDraw.Draw(raw_rgb)
                    radius = 6
                    for cx, cy in centers:
                        x0, y0 = cx - radius, cy - radius
                        x1, y1 = cx + radius, cy + radius
                        draw_raw.ellipse([x0, y0, x1, y1], outline=(255, 0, 0), width=2)
                raw_out_path = Path.cwd() / "frame1_spots.png"
                raw_rgb.save(raw_out_path)
            except Exception as e:
                messagebox.showwarning(
                    "Save image warning", f"Could not save raw frame image: {e}"
                )

        img = ImageTk.PhotoImage(Image.fromarray(u8, mode="L"))
        lbl = ttk.Label(win, image=img)
        lbl.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Keep references so the image doesn't get garbage-collected
        self._st_popup_img_ref = img
        self._st_popup_label = lbl

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("AVI Player (single decode + process ALL frames + target display FPS)")

        # Single capture (decoded once)
        self.cap = None
        self.video_path = None

        # Spinner detection (first 10 Q/U frames)
        self._q_buf = []
        self._u_buf = []
        self._st_popup_done = False
        self._st_popup_img_ref = None  # keep PhotoImage alive
        self._st_popup_label = None

        self.frame_count = 0
        self.source_fps = 30.0  # metadata only; display is independent
        self.current_idx = 0

        # GUI image
        self.last_frame_gray = None
        self.tk_img = None

        # Display control
        self.playing = False
        self.target_display_fps = 15.0
        self._display_interval = 1.0 / self.target_display_fps
        self._last_display_t = 0.0

        # Threads + coordination
        self.decode_thread = None
        self.recon_thread = None
        self.stop_event = threading.Event()

        # EOF indicator for decoder
        self.decode_done = False

        # Processing progress
        self.proc_done = 0  # frames processed by recon thread

        # Queue: bounded to avoid RAM blow-up, BLOCKING put => no frame drops
        self.frame_q = queue.Queue(maxsize=16)

        self._build_ui()

    def _build_ui(self):
        top = ttk.Frame(self.root, padding=8)
        top.pack(side=tk.TOP, fill=tk.X)

        ttk.Button(top, text="Select AVI", command=self.open_video).pack(side=tk.LEFT)
        self.play_btn = ttk.Button(top, text="Play", command=self.toggle_play, state=tk.DISABLED)
        self.play_btn.pack(side=tk.LEFT, padx=(8, 0))
        ttk.Button(top, text="Return to start", command=self.return_to_start, state=tk.DISABLED).pack(
            side=tk.LEFT, padx=(8, 0)
        )
        self.restart_btn = top.winfo_children()[-1]

        self.status_var = tk.StringVar(value="No video loaded")
        ttk.Label(top, textvariable=self.status_var).pack(side=tk.LEFT, padx=(12, 0))

        main = ttk.Frame(self.root, padding=8)
        main.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.img_label = ttk.Label(main)
        self.img_label.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.bottom_var = tk.StringVar(value="")
        ttk.Label(main, textvariable=self.bottom_var).pack(side=tk.BOTTOM, anchor="w")

    def _show_finished(self, show: bool):
        self.bottom_var.set("Playback finished" if show else "")

    def open_video(self):
        path = filedialog.askopenfilename(
            title="Select AVI file",
            filetypes=[("AVI files", "*.avi"), ("All files", "*.*")],
        )
        if not path:
            return

        self._close_video()

        cap = cv2.VideoCapture(path, cv2.CAP_AVFOUNDATION)
        if not cap.isOpened():
            cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            messagebox.showerror("Error", f"Could not open: {path}")
            return

        ok, frame0 = cap.read()
        if not ok or frame0 is None:
            cap.release()
            messagebox.showerror("Error", "Could not read first frame.")
            return

        gray0 = _to_gray_u8(frame0)
        if gray0 is None:
            cap.release()
            messagebox.showerror("Error", "Could not convert first frame to grayscale.")
            return

        # rewind for decode thread
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        self.cap = cap
        self.video_path = path
        self._show_finished(False)

        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        fps = float(self.cap.get(cv2.CAP_PROP_FPS)) or 0.0
        self.source_fps = fps if fps > 1.0 else 30.0

        self.current_idx = 0
        self.last_frame_gray = gray0
        self._display(gray0)

        # Reset state
        self.proc_done = 0
        self.decode_done = False
        self._clear_queue()
        self.stop_event.clear()

        # Reset spinner popup buffers/state (ADDED)
        self._q_buf = []
        self._u_buf = []
        self._st_popup_done = False
        self._st_popup_img_ref = None
        self._st_popup_label = None

        # Start recon thread (consumes queue, processes EVERY frame)
        H, W = gray0.shape
        self.recon_thread = threading.Thread(target=self._recon_worker, args=((H, W),), daemon=True)
        self.recon_thread.start()

        # Start decode thread (produces queue, BLOCKING puts, no drops)
        self.decode_thread = threading.Thread(target=self._decode_worker, daemon=True)
        self.decode_thread.start()

        name = path.split("/")[-1]
        self.status_var.set(
            f"Loaded: {name}  src≈{self.source_fps:.2f}fps  "
            f"disp≈{self.target_display_fps:.1f}fps  Q/U: 0%"
        )

        self.play_btn.configure(state=tk.NORMAL, text="Play")
        self.restart_btn.configure(state=tk.NORMAL)

        self.playing = False
        self._last_display_t = 0.0

        self._status_tick()

    def toggle_play(self):
        if self.cap is None:
            return

        self.playing = not self.playing
        self.play_btn.configure(text="Pause" if self.playing else "Play")

        if self.playing:
            self._show_finished(False)
            self._display_tick()

    def _display_tick(self):
        """
        Displays latest decoded frame at target_display_fps.
        Decoding/processing are background threads.
        """
        if not self.playing:
            return

        if self.last_frame_gray is not None:
            now = time.perf_counter()
            if (now - self._last_display_t) >= self._display_interval:
                self._display(self.last_frame_gray)
                self._last_display_t = now

        # Consider playback finished only when decoding is done
        if self.decode_done:
            self.playing = False
            self.play_btn.configure(text="Play")
            self._show_finished(True)
            return

        self.root.after(max(1, int(self._display_interval * 1000)), self._display_tick)

    def return_to_start(self):
        if self.video_path is None:
            return

        self._show_finished(False)
        path = self.video_path

        # stop everything and close
        self._close_video()

        # reopen same file
        cap = cv2.VideoCapture(path, cv2.CAP_AVFOUNDATION)
        if not cap.isOpened():
            cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            messagebox.showerror("Error", f"Could not reopen: {path}")
            return

        ok, frame0 = cap.read()
        if not ok or frame0 is None:
            cap.release()
            messagebox.showerror("Error", "Could not read first frame.")
            return

        gray0 = _to_gray_u8(frame0)
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        self.cap = cap
        self.video_path = path
        self._show_finished(False)

        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        fps = float(self.cap.get(cv2.CAP_PROP_FPS)) or 0.0
        self.source_fps = fps if fps > 1.0 else 30.0

        self.current_idx = 0
        self.last_frame_gray = gray0
        self._display(gray0)

        self.proc_done = 0
        self.decode_done = False
        self._clear_queue()
        self.stop_event.clear()

        H, W = gray0.shape
        self.recon_thread = threading.Thread(target=self._recon_worker, args=((H, W),), daemon=True)
        self.recon_thread.start()
        self.decode_thread = threading.Thread(target=self._decode_worker, daemon=True)
        self.decode_thread.start()

        name = path.split("/")[-1]
        self.status_var.set(
            f"Loaded: {name}  src≈{self.source_fps:.2f}fps  "
            f"disp≈{self.target_display_fps:.1f}fps  Q/U: 0%"
        )

        self.playing = False
        self.play_btn.configure(text="Play", state=tk.NORMAL)
        self.restart_btn.configure(state=tk.NORMAL)
        self._last_display_t = 0.0

        # Reset popup state
        self._q_buf = []
        self._u_buf = []
        self._st_popup_done = False
        self._st_popup_img_ref = None
        self._st_popup_label = None

        self._status_tick()

    def _status_tick(self):
        if not self.video_path:
            return

        if self.frame_count > 0:
            pct = 100.0 * (self.proc_done / float(self.frame_count))
            pct = max(10.0, min(100.0, pct))
        else:
            pct = 0.0

        name = self.video_path.split("/")[-1]
        self.status_var.set(
            f"Loaded: {name}  src≈{self.source_fps:.2f}fps  "
            f"disp≈{self.target_display_fps:.1f}fps  Q/U: {pct:.1f}%"
        )
        self.root.after(200, self._status_tick)

    def _display(self, gray_u8: np.ndarray):
        if gray_u8 is None:
            return
        img = ImageTk.PhotoImage(Image.fromarray(gray_u8, mode="L"))
        self.tk_img = img
        self.img_label.configure(image=img)

    def _decode_worker(self):
        idx = 0
        try:
            while not self.stop_event.is_set() and self.cap is not None:
                ok, frame = self.cap.read()
                if not ok or frame is None:
                    break

                gray = _to_gray_u8(frame)
                if gray is None:
                    break

                idx += 1
                self.current_idx = idx
                self.last_frame_gray = gray

                # BLOCK until recon consumes enough space -> guarantees no dropped frames
                while not self.stop_event.is_set():
                    try:
                        self.frame_q.put(gray, timeout=0.1)
                        break
                    except queue.Full:
                        continue
        finally:
            self.decode_done = True

    def _recon_worker(self, shape: tuple[int, int]):
        recon = make_qu_reconstructor(shape, out_dtype=np.int16)
        processed = 0

        while not self.stop_event.is_set():
            # exit condition: decoding finished and queue drained
            if self.decode_done and self.frame_q.empty():
                break

            try:
                gray = self.frame_q.get(timeout=0.1)
            except queue.Empty:
                continue

            Q, U = recon(gray)

            # Buffer first 10 Q/U frames, then compute median(S_t^2) and pop up once
            if (not self._st_popup_done) and (len(self._q_buf) < 10):
                self._q_buf.append(Q.copy())
                self._u_buf.append(U.copy())

                if len(self._q_buf) == 10:
                    try:
                        Q10 = np.stack(self._q_buf, axis=0)
                        U10 = np.stack(self._u_buf, axis=0)
                        st2_med = detect_spinners.median_st2(Q10, U10)
                    except Exception as e:
                        self.root.after(0, lambda: messagebox.showerror("Spinner detect error", str(e)))
                    else:
                        self._st_popup_done = True
                        # schedule UI update on main thread
                        self.root.after(0, lambda arr=st2_med: self._show_st2_popup(arr))

            processed += 1
            self.proc_done = processed

    def _clear_queue(self):
        try:
            while True:
                self.frame_q.get_nowait()
        except queue.Empty:
            pass

    def _stop_workers(self):
        self.stop_event.set()
        self.decode_done = True

        if self.decode_thread and self.decode_thread.is_alive():
            self.decode_thread.join(timeout=1.0)
        self.decode_thread = None

        if self.recon_thread and self.recon_thread.is_alive():
            self.recon_thread.join(timeout=1.0)
        self.recon_thread = None

        self._clear_queue()
        self.stop_event.clear()

    def _close_video(self):
        self.playing = False
        self._show_finished(False)

        self.play_btn.configure(text="Play", state=tk.DISABLED)
        self.restart_btn.configure(state=tk.DISABLED)

        self._stop_workers()

        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None

        self.video_path = None
        self.frame_count = 0
        self.source_fps = 30.0
        self.current_idx = 0
        self.last_frame_gray = None
        self.tk_img = None
        self.proc_done = 0
        self.decode_done = False

        self.status_var.set("No video loaded")
        self.bottom_var.set("")
        self.img_label.configure(image="")

        self._q_buf = []
        self._u_buf = []
        self._st_popup_done = False
        self._st_popup_img_ref = None
        self._st_popup_label = None

    def on_close(self):
        self._close_video()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = BasicVideoPlayer(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()
