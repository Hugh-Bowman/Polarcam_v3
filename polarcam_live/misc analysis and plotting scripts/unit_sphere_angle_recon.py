from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
import tkinter as tk
from tkinter import filedialog

import matplotlib.pyplot as plt
import numpy as np


@dataclass
class ReconResult:
    path: Path
    mode: str
    fps: float
    i0: np.ndarray
    i45: np.ndarray
    i90: np.ndarray
    i135: np.ndarray
    x: np.ndarray
    y: np.ndarray
    u: np.ndarray
    axis_k: np.ndarray
    e1: np.ndarray
    e2: np.ndarray
    alpha_deg: float
    beta_deg: float
    gamma_deg: float
    phi_wrapped_deg: np.ndarray
    phi_unwrapped_deg: np.ndarray


class UnitSphereAngleRecon:
    def __init__(
        self,
        fps: float = 1600.0,
        na: float = 1.40,
        n_medium: float = 1.518,
        bin_deg: float = 9.0,
        max_phi_step_deg: float = 3.0,
        block_inner_na: bool = False,
        inner_fraction: float = 0.0,
    ) -> None:
        self.fps = float(max(1e-9, fps))
        self.na = float(max(0.01, min(na, n_medium - 1e-6)))
        self.n_medium = float(max(1.0, n_medium))
        self.bin_deg = float(max(1.0, min(180.0, bin_deg)))
        self.max_phi_step_deg = float(max(0.1, min(45.0, max_phi_step_deg)))
        self.block_inner_na = bool(block_inner_na)
        self.inner_fraction = float(max(0.0, min(0.95, inner_fraction)))

        self.ray_theta_samples = 28
        self.ray_psi_samples = 96
        self.ray_dirs = np.zeros((1, 3), dtype=np.float64)
        self.ray_w = np.ones((1,), dtype=np.float64)
        self._build_collection_rays()

    @staticmethod
    def _unit(v: np.ndarray) -> np.ndarray:
        n = float(np.linalg.norm(v))
        if n <= 0.0:
            return np.array([0.0, 0.0, 1.0], dtype=np.float64)
        return v / n

    @staticmethod
    def _to_gray_u8(frame: np.ndarray) -> np.ndarray | None:
        if frame is None:
            return None
        x = np.asarray(frame)
        if x.ndim == 3:
            x = x[..., 0]
        if x.ndim != 2:
            return None
        if x.dtype == np.uint8:
            return np.ascontiguousarray(x)
        if np.issubdtype(x.dtype, np.integer):
            maxv = int(x.max()) if x.size else 0
            if maxv <= 255:
                out = x.astype(np.uint8, copy=False)
            elif maxv <= 4095:
                out = (x.astype(np.uint16, copy=False) >> 4).astype(np.uint8, copy=False)
            else:
                out = (x.astype(np.uint32, copy=False) >> 8).astype(np.uint8, copy=False)
            return np.ascontiguousarray(out)
        if np.issubdtype(x.dtype, np.floating):
            maxv = float(np.nanmax(x)) if x.size else 0.0
            if not np.isfinite(maxv):
                maxv = 0.0
            if maxv <= 1.0:
                y = x * 255.0
            elif maxv <= 255.0:
                y = x
            elif maxv <= 4095.0:
                y = x / 16.0
            else:
                y = x / 256.0
            return np.ascontiguousarray(np.clip(y, 0.0, 255.0).astype(np.uint8))
        return np.ascontiguousarray(x.astype(np.uint8))

    def _parse_npy_source(self, arr: np.ndarray) -> tuple[bool, int, np.ndarray]:
        if arr.dtype == object:
            raise RuntimeError("Unsupported NPY: object arrays are not supported.")
        if arr.ndim < 2 or arr.ndim > 4:
            raise RuntimeError(f"Unsupported NPY shape: {arr.shape}")

        if arr.ndim == 2:
            frame0 = arr
            frame_count = 1
            has_frames_dim = False
        elif arr.ndim == 3:
            if int(arr.shape[-1]) in (1, 3, 4) and int(arr.shape[0]) > 4 and int(arr.shape[1]) > 4:
                frame0 = arr
                frame_count = 1
                has_frames_dim = False
            else:
                frame0 = arr[0]
                frame_count = int(arr.shape[0])
                has_frames_dim = True
        else:
            frame0 = arr[0]
            frame_count = int(arr.shape[0])
            has_frames_dim = True

        g0 = self._to_gray_u8(frame0)
        if g0 is None:
            raise RuntimeError("Could not convert first frame to grayscale.")
        if (g0.shape[0] % 2) != 0 or (g0.shape[1] % 2) != 0:
            raise RuntimeError(f"Frame shape must be even (polar mosaic). Got {g0.shape}.")
        return has_frames_dim, frame_count, g0

    def _iter_gray_frames(self, arr: np.ndarray, has_frames_dim: bool, frame_count: int):
        if has_frames_dim:
            for i in range(frame_count):
                g = self._to_gray_u8(arr[i])
                if g is not None:
                    yield g
        else:
            g = self._to_gray_u8(arr)
            if g is not None:
                yield g

    @staticmethod
    def _extract_pol_means(g: np.ndarray) -> tuple[float, float, float, float]:
        i0 = g[0::2, 0::2]
        i45 = g[0::2, 1::2]
        i135 = g[1::2, 0::2]
        i90 = g[1::2, 1::2]
        return float(i0.mean()), float(i45.mean()), float(i90.mean()), float(i135.mean())

    def _load_polar_timeseries(self, path: Path) -> tuple[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        arr = np.load(path, mmap_mode="r", allow_pickle=True)
        has_frames_dim, frame_count, g0 = self._parse_npy_source(arr)

        h, w = g0.shape
        small = min(h, w)
        large = max(h, w)
        inspection = bool(large == 256 and small < 50)
        side = int(small) if inspection else None

        i0, i45, i90, i135 = [], [], [], []
        for g in self._iter_gray_frames(arr, has_frames_dim, frame_count):
            g2 = g[:side, :side] if (inspection and side is not None) else g
            a, b, c, d = self._extract_pol_means(g2)
            i0.append(a)
            i45.append(b)
            i90.append(c)
            i135.append(d)

        if not i0:
            raise RuntimeError("No valid frames found in NPY.")

        mode = "inspection" if inspection else "full-frame"
        return (
            mode,
            np.asarray(i0, dtype=np.float64),
            np.asarray(i45, dtype=np.float64),
            np.asarray(i90, dtype=np.float64),
            np.asarray(i135, dtype=np.float64),
        )

    def _build_collection_rays(self) -> None:
        ratio = float(self.na) / float(self.n_medium)
        ratio = max(0.0, min(0.999999, ratio))
        theta_max = math.asin(ratio)

        if theta_max <= 1e-9:
            self.ray_dirs = np.array([[0.0, 0.0, 1.0]], dtype=np.float64)
            self.ray_w = np.array([1.0], dtype=np.float64)
            return

        theta = np.linspace(0.0, theta_max, int(max(4, self.ray_theta_samples)))
        psi = np.linspace(0.0, 2.0 * np.pi, int(max(8, self.ray_psi_samples)), endpoint=False)
        th, ps = np.meshgrid(theta, psi, indexing="ij")
        sin_th = np.sin(th)

        dirs_all = np.stack([sin_th * np.cos(ps), sin_th * np.sin(ps), np.cos(th)], axis=-1).reshape(-1, 3)
        w_all = (sin_th * (theta[1] - theta[0] if theta.size > 1 else theta_max) * (2.0 * np.pi / psi.size)).reshape(-1)

        if self.block_inner_na:
            sin_th_max = max(1e-12, math.sin(theta_max))
            rho = sin_th.reshape(-1) / sin_th_max
            keep = rho >= float(self.inner_fraction)
            if not np.any(keep):
                keep = np.ones_like(rho, dtype=bool)
            dirs = dirs_all[keep]
            w = w_all[keep]
        else:
            dirs = dirs_all
            w = w_all

        sw = float(np.sum(w))
        if sw <= 0.0 or (not np.isfinite(sw)):
            w = np.ones_like(w) / float(w.size)
        else:
            w = w / sw

        self.ray_dirs = dirs.astype(np.float64, copy=False)
        self.ray_w = w.astype(np.float64, copy=False)

    def _simulate_components(self, dirs: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        rays = self.ray_dirs
        w = self.ray_w

        dot = dirs @ rays.T
        ex = dirs[:, 0:1] - (dot * rays[None, :, 0])
        ey = dirs[:, 1:2] - (dot * rays[None, :, 1])

        def intensity(psi_deg: float) -> np.ndarray:
            pr = math.radians(float(psi_deg))
            ax = math.cos(pr)
            ay = math.sin(pr)
            comp = (ax * ex) + (ay * ey)
            return (comp * comp) @ w

        i0 = intensity(0.0)
        i90 = intensity(90.0)
        i45 = intensity(45.0)
        i135 = intensity(135.0)
        return i0, i90, i45, i135

    def _invert_to_unit_sphere(
        self, i0: np.ndarray, i45: np.ndarray, i90: np.ndarray, i135: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        eps = 1e-12
        s0 = i0 + i90
        s45 = i45 + i135

        x = (i0 - i90) / (s0 + eps)
        y = (i45 - i135) / (s45 + eps)

        meas = np.column_stack((i0, i45, i90, i135))
        meas_sum = np.sum(meas, axis=1, keepdims=True)
        meas_sum = np.where(meas_sum <= eps, 1.0, meas_sum)
        q_meas = meas / meas_sum

        n_theta = 45
        n_psi = 144
        theta = np.linspace(0.0, np.pi, n_theta)
        psi = np.linspace(0.0, 2.0 * np.pi, n_psi, endpoint=False)
        th, ps = np.meshgrid(theta, psi, indexing="ij")
        dirs = np.stack([np.sin(th) * np.cos(ps), np.sin(th) * np.sin(ps), np.cos(th)], axis=-1).reshape(-1, 3)

        g0, g90, g45, g135 = self._simulate_components(dirs)
        grid = np.column_stack((g0, g45, g90, g135))
        grid_sum = np.sum(grid, axis=1, keepdims=True)
        grid_sum = np.where(grid_sum <= eps, 1.0, grid_sum)
        q_grid = grid / grid_sum

        n = int(q_meas.shape[0])
        idx = np.zeros((n,), dtype=np.int32)
        chunk = 500
        for i0c in range(0, n, chunk):
            i1c = min(n, i0c + chunk)
            part = q_meas[i0c:i1c]
            d2 = np.sum((part[:, None, :] - q_grid[None, :, :]) ** 2, axis=2)
            idx[i0c:i1c] = np.argmin(d2, axis=1)

        u = dirs[idx]
        return x, y, u

    @staticmethod
    def _fit_axis_from_u(u: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float, float]:
        m = (u.T @ u) / max(1, int(u.shape[0]))
        vals, vecs = np.linalg.eigh(m)
        k = vecs[:, int(np.argmax(vals))]
        if k[2] < 0.0:
            k = -k
        k = k / max(1e-12, np.linalg.norm(k))

        z = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        ref = z if abs(float(np.dot(k, z))) < 0.95 else np.array([1.0, 0.0, 0.0], dtype=np.float64)
        e1 = np.cross(k, ref)
        e1 = e1 / max(1e-12, np.linalg.norm(e1))
        e2 = np.cross(k, e1)
        e2 = e2 / max(1e-12, np.linalg.norm(e2))

        cos_g = float(np.mean(np.abs(u @ k)))
        cos_g = max(-1.0, min(1.0, cos_g))
        gamma_deg = math.degrees(math.acos(cos_g))

        alpha_deg = math.degrees(math.acos(max(-1.0, min(1.0, float(k[2])))))
        beta_deg = math.degrees(math.atan2(float(k[1]), float(k[0]))) % 360.0
        return k, e1, e2, alpha_deg, beta_deg, gamma_deg

    @staticmethod
    def _wrap_pi(x: float) -> float:
        return float((x + math.pi) % (2.0 * math.pi) - math.pi)

    def _unwrap_double_cover_phi(self, raw_wrapped_rad: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Resolve the phi vs (phi + pi) ambiguity frame-to-frame with continuity.
        """
        raw = np.asarray(raw_wrapped_rad, dtype=np.float64)
        n = int(raw.size)
        if n == 0:
            return np.zeros((0,), dtype=np.float64), np.zeros((0,), dtype=np.float64)

        cand = np.column_stack((raw, np.mod(raw + np.pi, 2.0 * np.pi)))
        cost = np.full((n, 2), np.inf, dtype=np.float64)
        unw = np.zeros((n, 2), dtype=np.float64)
        back = np.zeros((n, 2), dtype=np.int8)

        cost[0, :] = 0.0
        unw[0, 0] = cand[0, 0]
        unw[0, 1] = cand[0, 1]

        max_step = math.radians(self.max_phi_step_deg)
        rough = np.unwrap(raw.copy())
        d_rough = np.diff(rough) if n > 1 else np.zeros((0,), dtype=np.float64)
        med = float(np.median(d_rough)) if d_rough.size else 0.0
        pref_dir = 1 if med > 1e-6 else (-1 if med < -1e-6 else 0)

        for t in range(1, n):
            for b in (0, 1):
                best_c = np.inf
                best_u = 0.0
                best_prev = 0
                for pb in (0, 1):
                    prev_u = unw[t - 1, pb]
                    d = self._wrap_pi(float(cand[t, b] - prev_u))
                    cur_u = prev_u + d

                    c = cost[t - 1, pb] + (d / max(1e-12, max_step)) ** 2
                    if abs(d) > max_step:
                        over = (abs(d) - max_step) / max(1e-12, max_step)
                        c += 250.0 + 500.0 * (over * over)
                    if pref_dir > 0 and d < -0.25 * max_step:
                        c += 30.0 * ((-d) / max(1e-12, max_step)) ** 2
                    elif pref_dir < 0 and d > 0.25 * max_step:
                        c += 30.0 * (d / max(1e-12, max_step)) ** 2

                    if c < best_c:
                        best_c = c
                        best_u = cur_u
                        best_prev = pb

                cost[t, b] = best_c
                unw[t, b] = best_u
                back[t, b] = np.int8(best_prev)

        state = int(np.argmin(cost[-1]))
        out_unw = np.zeros((n,), dtype=np.float64)
        for t in range(n - 1, -1, -1):
            out_unw[t] = unw[t, state]
            state = int(back[t, state]) if t > 0 else 0

        out_wrapped = np.mod(out_unw, 2.0 * np.pi)
        return out_wrapped, out_unw

    def run(self, path: Path) -> ReconResult:
        mode, i0, i45, i90, i135 = self._load_polar_timeseries(path)
        x, y, u = self._invert_to_unit_sphere(i0, i45, i90, i135)

        k, e1, e2, alpha_deg, beta_deg, gamma_deg = self._fit_axis_from_u(u)
        proj = u - (u @ k)[:, None] * k[None, :]
        pn = np.linalg.norm(proj, axis=1)

        raw = np.zeros((u.shape[0],), dtype=np.float64)
        valid = pn > 1e-12
        raw[valid] = np.mod(np.arctan2(proj[valid] @ e2, proj[valid] @ e1), 2.0 * np.pi)
        if np.any(~valid):
            for i in np.where(~valid)[0]:
                raw[i] = raw[i - 1] if i > 0 else 0.0

        phi_wrapped, phi_unwrapped = self._unwrap_double_cover_phi(raw)

        return ReconResult(
            path=path,
            mode=mode,
            fps=self.fps,
            i0=i0,
            i45=i45,
            i90=i90,
            i135=i135,
            x=x,
            y=y,
            u=u,
            axis_k=k,
            e1=e1,
            e2=e2,
            alpha_deg=alpha_deg,
            beta_deg=beta_deg,
            gamma_deg=gamma_deg,
            phi_wrapped_deg=np.degrees(phi_wrapped),
            phi_unwrapped_deg=np.degrees(phi_unwrapped),
        )

    def plot(self, res: ReconResult) -> None:
        t = np.arange(res.x.size, dtype=np.float64) / max(1e-12, res.fps)

        fig = plt.figure(figsize=(16, 9))
        gs = fig.add_gridspec(2, 3, hspace=0.30, wspace=0.28)

        ax_xy = fig.add_subplot(gs[0, 0])
        ax_sphere = fig.add_subplot(gs[0, 1], projection="3d")
        ax_plane = fig.add_subplot(gs[0, 2])
        ax_phi_t = fig.add_subplot(gs[1, 0:2])
        ax_hist = fig.add_subplot(gs[1, 2])

        # XY plot
        ax_xy.plot(res.x, res.y, color="0.55", lw=0.8, alpha=0.8)
        sc_xy = ax_xy.scatter(res.x, res.y, c=t, s=4, cmap="viridis", alpha=0.85)
        ax_xy.axhline(0.0, color="0.85", lw=1)
        ax_xy.axvline(0.0, color="0.85", lw=1)
        ax_xy.set_aspect("equal", adjustable="box")
        ax_xy.set_xlabel("X = (I0-I90)/(I0+I90)")
        ax_xy.set_ylabel("Y = (I45-I135)/(I45+I135)")
        ax_xy.set_title("Data XY Trace")
        ax_xy.grid(alpha=0.2)
        cb = fig.colorbar(sc_xy, ax=ax_xy, fraction=0.046, pad=0.04)
        cb.set_label("time (s)")

        # Unit sphere + fitted axis/circle
        uu, vv = np.mgrid[0 : 2 * np.pi : 40j, 0 : np.pi : 20j]
        xs = np.cos(uu) * np.sin(vv)
        ys = np.sin(uu) * np.sin(vv)
        zs = np.cos(vv)
        ax_sphere.plot_wireframe(xs, ys, zs, color="0.8", linewidth=0.4, alpha=0.5)

        sc3 = ax_sphere.scatter(res.u[:, 0], res.u[:, 1], res.u[:, 2], c=t, s=5, cmap="viridis", alpha=0.9)
        k = res.axis_k
        ax_sphere.plot([-k[0], k[0]], [-k[1], k[1]], [-k[2], k[2]], color="tab:red", lw=2.0, label="rotation axis")

        gamma = math.radians(res.gamma_deg)
        th = np.linspace(0.0, 2.0 * np.pi, 361)
        circ = (math.cos(gamma) * k[None, :]) + (
            math.sin(gamma) * (np.cos(th)[:, None] * res.e1[None, :] + np.sin(th)[:, None] * res.e2[None, :])
        )
        ax_sphere.plot(circ[:, 0], circ[:, 1], circ[:, 2], color="tab:orange", lw=2.0, label="fitted small circle")
        ax_sphere.set_box_aspect((1, 1, 1))
        ax_sphere.set_xlim(-1.05, 1.05)
        ax_sphere.set_ylim(-1.05, 1.05)
        ax_sphere.set_zlim(-1.05, 1.05)
        ax_sphere.set_xlabel("x")
        ax_sphere.set_ylabel("y")
        ax_sphere.set_zlabel("z")
        ax_sphere.set_title("Reconstructed Unit-Sphere Points")
        ax_sphere.legend(loc="upper left", fontsize=8)
        fig.colorbar(sc3, ax=ax_sphere, fraction=0.046, pad=0.04, label="time (s)")

        # Plane perpendicular to axis
        proj = res.u - (res.u @ k)[:, None] * k[None, :]
        p1 = proj @ res.e1
        p2 = proj @ res.e2
        ax_plane.scatter(p1, p2, c=t, s=5, cmap="viridis", alpha=0.9)
        r = np.sqrt(p1 * p1 + p2 * p2)
        r0 = float(np.median(r)) if r.size else 0.0
        th2 = np.linspace(0.0, 2.0 * np.pi, 361)
        ax_plane.plot(r0 * np.cos(th2), r0 * np.sin(th2), color="tab:orange", lw=1.8, label="median radius")
        ax_plane.axhline(0.0, color="0.85", lw=1)
        ax_plane.axvline(0.0, color="0.85", lw=1)
        ax_plane.set_aspect("equal", adjustable="box")
        ax_plane.set_xlabel("component along e1")
        ax_plane.set_ylabel("component along e2")
        ax_plane.set_title("Projection Into Plane Perpendicular To Axis")
        ax_plane.grid(alpha=0.2)
        ax_plane.legend(loc="best", fontsize=8)

        # phi time trace (continuous, axis-referenced)
        ax_phi_t.plot(t, res.phi_unwrapped_deg, color="tab:blue", lw=1.2)
        ax_phi_t.set_xlabel("time (s)")
        ax_phi_t.set_ylabel("phi around rotation axis (deg, unwrapped)")
        ax_phi_t.set_title("Continuous phi(t) Around Fitted Rotation Axis")
        ax_phi_t.grid(alpha=0.25)

        # phi histogram
        bw = self.bin_deg
        edges = np.arange(0.0, 360.0 + 1e-9, bw)
        wrapped = np.mod(res.phi_wrapped_deg, 360.0)
        counts, _ = np.histogram(wrapped, bins=edges)
        centers = 0.5 * (edges[:-1] + edges[1:])
        ax_hist.bar(centers, counts, width=0.9 * bw, color="tab:purple", alpha=0.85)
        ax_hist.set_xlim(0.0, 360.0)
        ax_hist.set_xlabel("phi around axis (deg)")
        ax_hist.set_ylabel("count")
        ax_hist.set_title("phi Distribution")
        ax_hist.grid(alpha=0.2)

        fig.suptitle(
            (
                f"{res.path.name} [{res.mode}]   "
                f"alpha={res.alpha_deg:.2f} deg, beta={res.beta_deg:.2f} deg, gamma={res.gamma_deg:.2f} deg   "
                f"FPS={res.fps:.2f}, max-step={self.max_phi_step_deg:.2f} deg"
            ),
            fontsize=12,
        )
        plt.show()


def _pick_npy_file() -> Path | None:
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    path = filedialog.askopenfilename(
        title="Open NPY",
        filetypes=[("NumPy", "*.npy"), ("All files", "*.*")],
        initialdir=str(Path.cwd()),
    )
    try:
        root.destroy()
    except Exception:
        pass
    if not path:
        return None
    return Path(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Unit-sphere axis-angle reconstruction from polar NPY data.")
    parser.add_argument("npy", nargs="?", default=None, help="Path to .npy file (optional; dialog opens if omitted).")
    parser.add_argument("--fps", type=float, default=1600.0, help="Frame rate in Hz.")
    parser.add_argument("--na", type=float, default=1.40, help="Objective NA used for forward model inversion.")
    parser.add_argument("--n-medium", type=float, default=1.518, help="Immersion medium refractive index.")
    parser.add_argument("--bin-deg", type=float, default=9.0, help="Histogram bin size in degrees.")
    parser.add_argument(
        "--max-step-deg",
        type=float,
        default=3.0,
        help="Max preferred per-frame phi step for continuity tracking (degrees).",
    )
    parser.add_argument(
        "--inner-cut",
        type=float,
        default=0.0,
        help="Optional inner NA cutout fraction (0..0.95). Set >0 to enable center block.",
    )
    args = parser.parse_args()

    path = Path(args.npy) if args.npy else _pick_npy_file()
    if path is None:
        print("No file selected.")
        return
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    recon = UnitSphereAngleRecon(
        fps=float(args.fps),
        na=float(args.na),
        n_medium=float(args.n_medium),
        bin_deg=float(args.bin_deg),
        max_phi_step_deg=float(args.max_step_deg),
        block_inner_na=bool(float(args.inner_cut) > 0.0),
        inner_fraction=float(max(0.0, min(0.95, args.inner_cut))),
    )
    result = recon.run(path)
    recon.plot(result)


if __name__ == "__main__":
    main()
