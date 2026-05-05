from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
import tkinter as tk
from tkinter import filedialog

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button, CheckButtons, Slider, TextBox
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


@dataclass
class CurveResult:
    phi_deg: np.ndarray
    x: np.ndarray
    y: np.ndarray


class RodDipoleSimulator:
    """
    Forward + data-fit simulator for a rotating rod dipole.
    """

    def __init__(self) -> None:
        self.alpha_deg = 25.0
        self.beta_deg = 35.0
        self.gamma_deg = 25.0
        self.phi_deg = 0.0

        self.na = 1.40
        self.n_medium = 1.518

        self.block_inner_na = False
        self.block_inner_fraction = 0.2

        self.phi_samples = 361
        self.fit_phi_samples = 721
        self.ray_theta_samples = 28
        self.ray_psi_samples = 96

        self.ray_dirs = np.zeros((1, 3), dtype=np.float64)
        self.ray_w = np.ones((1,), dtype=np.float64)

        self.forward_curve: CurveResult | None = None
        self.forward_curve_ref_no_block: CurveResult | None = None

        self.view_mode = "sim"
        self.data_loaded = False
        self.data_name = ""
        self.data_mode = ""
        self.data_fps = 1600.0
        self.data_xy = np.zeros((0, 2), dtype=np.float64)
        self.data_t = np.zeros((0,), dtype=np.float64)

        self.fit_done = False
        self.fit_alpha = 0.0
        self.fit_beta = 0.0
        self.fit_gamma = 0.0
        self.fit_cost = np.nan
        self.meas_phi_wrapped = np.zeros((0,), dtype=np.float64)
        self.meas_phi_unwrapped = np.zeros((0,), dtype=np.float64)
        self.fit_phi_wrapped = np.zeros((0,), dtype=np.float64)
        self.fit_phi_unwrapped = np.zeros((0,), dtype=np.float64)
        self.fit_loop_phi = np.zeros((0,), dtype=np.float64)
        self.fit_loop_x = np.zeros((0,), dtype=np.float64)
        self.fit_loop_y = np.zeros((0,), dtype=np.float64)
        self.fit_x = np.zeros((0,), dtype=np.float64)
        self.fit_y = np.zeros((0,), dtype=np.float64)

        self.data_phi_hist_edges = np.linspace(0.0, 360.0, 41)
        self.data_phi_hist_counts = np.zeros((40,), dtype=np.float64)
        self.data_phi_peak_center = None

        self.data_idx = 0
        self.play_sim = False
        self.play_speed_deg = 2.0

        self.fig = None
        self.ax3d = None
        self.ax_xy = None
        self.ax_phi = None
        self.status_text = None

        self.s_alpha = None
        self.s_beta = None
        self.s_gamma = None
        self.s_phi = None
        self.s_na = None

        self.tb_cutout = None
        self.tb_fps = None
        self.tb_bin = None

        self.chk_block = None

        self.btn_play_sim = None
        self.btn_recompute = None
        self.btn_load = None
        self.btn_fit = None
        self.btn_tab_sim = None
        self.btn_tab_data = None

        self.timer = None

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

    @staticmethod
    def _rotation_axis(alpha_deg: float, beta_deg: float) -> np.ndarray:
        a = math.radians(float(alpha_deg))
        b = math.radians(float(beta_deg))
        return np.array([math.sin(a) * math.cos(b), math.sin(a) * math.sin(b), math.cos(a)], dtype=np.float64)

    def _cone_basis(self, k: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        z = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        ref = z if abs(float(np.dot(k, z))) < 0.95 else np.array([1.0, 0.0, 0.0], dtype=np.float64)
        e1 = self._unit(np.cross(k, ref))
        e2 = self._unit(np.cross(k, e1))
        return e1, e2

    def _rod_directions(self, alpha_deg: float, beta_deg: float, gamma_deg: float, phi_rad: np.ndarray) -> np.ndarray:
        k = self._rotation_axis(alpha_deg, beta_deg)
        e1, e2 = self._cone_basis(k)
        g = math.radians(float(gamma_deg))
        cg, sg = math.cos(g), math.sin(g)
        c = np.cos(phi_rad)[:, None]
        s = np.sin(phi_rad)[:, None]
        dirs = (cg * k[None, :]) + (sg * (c * e1[None, :] + s * e2[None, :]))
        norms = np.linalg.norm(dirs, axis=1, keepdims=True)
        norms = np.where(norms <= 0.0, 1.0, norms)
        return dirs / norms

    def _build_collection_rays(self, exclude_inner: bool | None = None) -> None:
        if exclude_inner is None:
            exclude_inner = bool(self.block_inner_na)

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

        if exclude_inner:
            sin_th_max = max(1e-12, math.sin(theta_max))
            rho = sin_th.reshape(-1) / sin_th_max
            keep = rho >= float(self.block_inner_fraction)
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

    def _simulate_xy_for_params(self, alpha_deg: float, beta_deg: float, gamma_deg: float, phi_deg: np.ndarray) -> CurveResult:
        dirs = self._rod_directions(alpha_deg, beta_deg, gamma_deg, np.deg2rad(phi_deg))
        i0, i90, i45, i135 = self._simulate_components(dirs)
        eps = 1e-15
        x = (i0 - i90) / (i0 + i90 + eps)
        y = (i45 - i135) / (i45 + i135 + eps)
        return CurveResult(phi_deg=phi_deg.astype(np.float64, copy=False), x=x, y=y)
    def _compute_forward_curve(self) -> None:
        phi_deg = np.linspace(0.0, 360.0, int(max(90, self.phi_samples)))
        self.forward_curve = self._simulate_xy_for_params(self.alpha_deg, self.beta_deg, self.gamma_deg, phi_deg)

        if self.block_inner_na:
            self._build_collection_rays(exclude_inner=False)
            self.forward_curve_ref_no_block = self._simulate_xy_for_params(self.alpha_deg, self.beta_deg, self.gamma_deg, phi_deg)
            self._build_collection_rays(exclude_inner=True)
        else:
            self.forward_curve_ref_no_block = None

    def _parse_npy_source(self, arr: np.ndarray) -> tuple[bool, int, np.ndarray]:
        if arr.dtype == object:
            raise RuntimeError("Unsupported NPY: object arrays are not supported.")
        if arr.ndim < 2 or arr.ndim > 4:
            raise RuntimeError("Unsupported NPY shape.")

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

    def _xy_from_gray_region(self, g: np.ndarray) -> tuple[float, float]:
        I0 = g[0::2, 0::2]
        I45 = g[0::2, 1::2]
        I135 = g[1::2, 0::2]
        I90 = g[1::2, 1::2]

        eps = 1e-6
        m0 = float(I0.mean()) if I0.size else 0.0
        m90 = float(I90.mean()) if I90.size else 0.0
        m45 = float(I45.mean()) if I45.size else 0.0
        m135 = float(I135.mean()) if I135.size else 0.0
        x = (m0 - m90) / (m0 + m90 + eps)
        y = (m45 - m135) / (m45 + m135 + eps)
        return (float(x), float(y))

    def _load_npy_xy(self, path: Path) -> tuple[np.ndarray, str]:
        arr = np.load(path, mmap_mode="r", allow_pickle=True)
        has_frames_dim, frame_count, g0 = self._parse_npy_source(arr)

        h, w = g0.shape
        small = min(h, w)
        large = max(h, w)
        inspection = bool(large == 256 and small < 50)
        side = int(small) if inspection else None

        xy: list[tuple[float, float]] = []
        for g in self._iter_gray_frames(arr, has_frames_dim, frame_count):
            g2 = g[:side, :side] if (inspection and side is not None) else g
            xy.append(self._xy_from_gray_region(g2))

        if not xy:
            raise RuntimeError("No valid frames found in NPY.")

        mode = "inspection" if inspection else "full-frame"
        return np.asarray(xy, dtype=np.float64), mode

    @staticmethod
    def _nearest_loop_cost(xy: np.ndarray, mx: np.ndarray, my: np.ndarray) -> float:
        d2 = (xy[:, 0:1] - mx[None, :]) ** 2 + (xy[:, 1:2] - my[None, :]) ** 2
        return float(np.mean(np.min(d2, axis=1)))

    def _fit_geometry_to_xy(self, xy: np.ndarray) -> tuple[float, float, float, float]:
        rng = np.random.default_rng(1234)

        n_fit = min(500, int(xy.shape[0]))
        if xy.shape[0] > n_fit:
            idx = np.linspace(0, xy.shape[0] - 1, n_fit).astype(int)
            xy_fit = xy[idx]
        else:
            xy_fit = xy

        phi_coarse = np.linspace(0.0, 360.0, 181)

        candidates = [(self.alpha_deg, self.beta_deg, self.gamma_deg)]
        for _ in range(220):
            candidates.append((
                float(rng.uniform(0.0, 90.0)),
                float(rng.uniform(0.0, 360.0)),
                float(rng.uniform(2.0, 80.0)),
            ))

        best = (np.inf, self.alpha_deg, self.beta_deg, self.gamma_deg)
        for a, b, g in candidates:
            c = self._simulate_xy_for_params(a, b, g, phi_coarse)
            cost = self._nearest_loop_cost(xy_fit, c.x, c.y)
            if cost < best[0]:
                best = (cost, a, b, g)

        _, a0, b0, g0 = best
        for da, db, dg in ((14.0, 30.0, 14.0), (6.0, 12.0, 6.0)):
            al = np.linspace(max(0.0, a0 - da), min(90.0, a0 + da), 6)
            bl = np.linspace((b0 - db) % 360.0, (b0 + db) % 360.0, 8)
            gl = np.linspace(max(1.0, g0 - dg), min(89.0, g0 + dg), 6)
            for a in al:
                for b in bl:
                    for g in gl:
                        c = self._simulate_xy_for_params(float(a), float(b), float(g), phi_coarse)
                        cost = self._nearest_loop_cost(xy_fit, c.x, c.y)
                        if cost < best[0]:
                            best = (cost, float(a), float(b % 360.0), float(g))
            _, a0, b0, g0 = best

        return best

    @staticmethod
    def _project_phi_nearest(xy: np.ndarray, phi_grid: np.ndarray, x_grid: np.ndarray, y_grid: np.ndarray) -> np.ndarray:
        n = int(xy.shape[0])
        out = np.zeros((n,), dtype=np.float64)
        chunk = 2500
        for i0 in range(0, n, chunk):
            i1 = min(n, i0 + chunk)
            part = xy[i0:i1]
            d2 = (part[:, 0:1] - x_grid[None, :]) ** 2 + (part[:, 1:2] - y_grid[None, :]) ** 2
            j = np.argmin(d2, axis=1)
            out[i0:i1] = phi_grid[j]
        return out

    def _track_phi_continuous(
        self,
        xy: np.ndarray,
        phi_grid: np.ndarray,
        x_grid: np.ndarray,
        y_grid: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Strict continuity-aware phi tracking on a closed XY loop.
        Enforces a hard maximum per-frame phi jump to prevent branch switching
        at self-intersections.
        """
        n = int(xy.shape[0])
        m = int(phi_grid.size)
        if n == 0 or m == 0:
            return np.zeros((0,), dtype=np.float64), np.zeros((0,), dtype=np.float64)

        # Hard continuity limit in degrees per frame.
        max_step_deg = 2.5
        step_idx = max(1, int(round((max_step_deg / 360.0) * max(1, m - 1))))

        idx_path = np.zeros((n,), dtype=np.int32)
        phi_unwrapped = np.zeros((n,), dtype=np.float64)

        # Frame 0: global nearest.
        d2_0 = (xy[0, 0] - x_grid) ** 2 + (xy[0, 1] - y_grid) ** 2
        i0 = int(np.argmin(d2_0))
        idx_path[0] = i0
        phi_unwrapped[0] = float(phi_grid[i0])

        offsets = np.arange(-step_idx, step_idx + 1, dtype=np.int32)

        for t in range(1, n):
            prev = int(idx_path[t - 1])
            cand = (prev + offsets) % m
            d2 = (xy[t, 0] - x_grid[cand]) ** 2 + (xy[t, 1] - y_grid[cand]) ** 2
            j = int(np.argmin(d2))
            cur = int(cand[j])
            idx_path[t] = cur

            d_idx = cur - prev
            d_idx = int(((d_idx + (m // 2)) % m) - (m // 2))
            phi_unwrapped[t] = phi_unwrapped[t - 1] + (360.0 * d_idx / max(1, (m - 1)))

        return np.mod(phi_unwrapped, 360.0), phi_unwrapped

    def _estimate_u_from_xy(self, xy: np.ndarray) -> np.ndarray:
        """
        Invert XY -> physical direction by nearest match over a dense sphere grid.
        This does not constrain points to the fitted rotation circle.
        """
        theta = np.linspace(0.0, np.pi, 61)
        psi = np.linspace(0.0, 2.0 * np.pi, 180, endpoint=False)
        th, ps = np.meshgrid(theta, psi, indexing="ij")

        sin_th = np.sin(th)
        u_grid = np.stack(
            [sin_th * np.cos(ps), sin_th * np.sin(ps), np.cos(th)],
            axis=-1,
        ).reshape(-1, 3)

        i0, i90, i45, i135 = self._simulate_components(u_grid)
        eps = 1e-15
        xg = (i0 - i90) / (i0 + i90 + eps)
        yg = (i45 - i135) / (i45 + i135 + eps)

        n = int(xy.shape[0])
        out = np.zeros((n, 3), dtype=np.float64)
        chunk = 2000
        for j0 in range(0, n, chunk):
            j1 = min(n, j0 + chunk)
            part = xy[j0:j1]
            d2 = (part[:, 0:1] - xg[None, :]) ** 2 + (part[:, 1:2] - yg[None, :]) ** 2
            idx = np.argmin(d2, axis=1)
            out[j0:j1] = u_grid[idx]
        return out

    def _phi_from_projected_plane(self, u_est: np.ndarray, alpha_deg: float, beta_deg: float) -> np.ndarray:
        """
        Project estimated physical directions into the fitted circle plane and
        compute phi from the angular coordinate in that plane.
        """
        k = self._rotation_axis(alpha_deg, beta_deg)
        e1, e2 = self._cone_basis(k)

        dotk = u_est @ k
        v = u_est - dotk[:, None] * k[None, :]
        vn = np.linalg.norm(v, axis=1)
        safe = vn > 1e-12
        v_unit = np.zeros_like(v)
        v_unit[safe] = v[safe] / vn[safe, None]
        v_unit[~safe] = e1[None, :]

        phi = np.degrees(np.arctan2(v_unit @ e2, v_unit @ e1))
        return np.mod(phi, 360.0)

    def _phi_distribution_from_xy(self, xy: np.ndarray, bin_deg: float) -> tuple[np.ndarray, np.ndarray, float | None]:
        arr = np.asarray(xy, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[1] != 2 or arr.shape[0] < 3:
            edges = np.arange(0.0, 360.0 + 1e-9, max(1.0, float(bin_deg)))
            return edges, np.zeros((edges.size - 1,), dtype=np.float64), None

        x = arr[:, 0]
        y = arr[:, 1]
        A = np.column_stack((2.0 * x, 2.0 * y, np.ones_like(x)))
        b = (x * x) + (y * y)
        sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        cx, cy = float(sol[0]), float(sol[1])

        x_shift = x - cx
        y_shift = y - cy
        phi = 0.5 * np.unwrap(np.arctan2(y_shift, x_shift))
        phi_deg = np.mod(np.degrees(phi), 360.0)

        bw = max(1.0, float(bin_deg))
        edges = np.arange(0.0, 360.0 + 1e-9, bw)
        counts, _ = np.histogram(phi_deg, bins=edges)
        peak = None
        if counts.size > 0 and int(np.max(counts)) > 0:
            i = int(np.argmax(counts))
            peak = 0.5 * (float(edges[i]) + float(edges[i + 1]))
        return edges, counts.astype(np.float64), peak

    def _fit_loaded_data(self) -> None:
        if not self.data_loaded:
            self._set_status("No data loaded.")
            return

        self._set_status("Fitting rod parameters to loaded XY data...")
        self._build_collection_rays(exclude_inner=self.block_inner_na)

        cost, a, b, g = self._fit_geometry_to_xy(self.data_xy)
        self.fit_alpha, self.fit_beta, self.fit_gamma, self.fit_cost = a, b, g, cost

        phi_grid = np.linspace(0.0, 360.0, int(max(360, self.fit_phi_samples)))
        curve = self._simulate_xy_for_params(a, b, g, phi_grid)
        self.fit_loop_phi = curve.phi_deg.copy()
        self.fit_loop_x = curve.x.copy()
        self.fit_loop_y = curve.y.copy()

        self._set_status("Fitting done. Tracking continuous phi on fitted loop...")
        phi_data_wrapped, phi_unwrapped = self._track_phi_continuous(self.data_xy, curve.phi_deg, curve.x, curve.y)

        phi_ext = np.append(curve.phi_deg, 360.0)
        x_ext = np.append(curve.x, curve.x[0])
        y_ext = np.append(curve.y, curve.y[0])
        fit_x = np.interp(phi_data_wrapped, phi_ext, x_ext)
        fit_y = np.interp(phi_data_wrapped, phi_ext, y_ext)

        self.meas_phi_wrapped = phi_data_wrapped
        self.meas_phi_unwrapped = phi_unwrapped
        self.fit_phi_wrapped = phi_data_wrapped
        self.fit_phi_unwrapped = phi_unwrapped
        self.fit_x = fit_x
        self.fit_y = fit_y
        self.data_idx = max(0, int(phi_data_wrapped.size) - 1)

        self.fit_done = True
        self.data_phi_hist_edges, self.data_phi_hist_counts, self.data_phi_peak_center = self._phi_distribution_from_xy(
            np.column_stack((fit_x, fit_y)),
            self._get_bin_deg(),
        )

        self._set_status(
            f"Fit complete (single loop): axis alpha={a:.2f}, beta={b:.2f}, cone gamma={g:.2f}, cost={cost:.6f}"
        )
    def _set_status(self, msg: str) -> None:
        if self.status_text is not None:
            self.status_text.set_text(msg)
        if self.fig is not None:
            self.fig.canvas.draw_idle()

    def _read_controls(self) -> None:
        self.alpha_deg = float(self.s_alpha.val)
        self.beta_deg = float(self.s_beta.val)
        self.gamma_deg = float(self.s_gamma.val)
        self.phi_deg = float(self.s_phi.val)
        self.na = float(self.s_na.val)
        self.na = max(0.01, min(self.na, self.n_medium - 1e-6))

    def _get_bin_deg(self) -> float:
        try:
            b = float(self.tb_bin.text)
        except Exception:
            b = 9.0
        if not np.isfinite(b) or b <= 0.0:
            b = 9.0
        return float(min(180.0, max(1.0, b)))

    def _on_geometry_slider(self, _val: float) -> None:
        self._read_controls()
        self._recompute_all()

    def _on_phi_slider(self, _val: float) -> None:
        self._read_controls()
        self._draw_scene()

    def _on_toggle_play_sim(self, _event) -> None:
        self.play_sim = not bool(self.play_sim)
        if self.btn_play_sim is not None:
            self.btn_play_sim.label.set_text("Pause Sim" if self.play_sim else "Play Sim")

    def _on_recompute_button(self, _event) -> None:
        self._read_controls()
        self._recompute_all()

    def _on_toggle_block(self, _label: str) -> None:
        if self.chk_block is not None:
            self.block_inner_na = bool(self.chk_block.get_status()[0])
        self._read_controls()
        self._recompute_all()

    def _on_cutout_submit(self, text: str) -> None:
        try:
            v = float(text)
        except Exception:
            if self.tb_cutout is not None:
                prev = self.tb_cutout.eventson
                self.tb_cutout.eventson = False
                self.tb_cutout.set_val(f"{self.block_inner_fraction:.3f}")
                self.tb_cutout.eventson = prev
            return

        self.block_inner_fraction = float(max(0.0, min(0.95, v)))
        if self.tb_cutout is not None:
            prev = self.tb_cutout.eventson
            self.tb_cutout.eventson = False
            self.tb_cutout.set_val(f"{self.block_inner_fraction:.3f}")
            self.tb_cutout.eventson = prev
        self._set_status(f"Inner pupil cutout set to {self.block_inner_fraction:.3f} of pupil radius")
        self._recompute_all()

    def _on_fps_submit(self, text: str) -> None:
        try:
            f = float(text)
            if f > 0.0:
                self.data_fps = float(f)
                if self.data_loaded:
                    n = int(self.data_xy.shape[0])
                    self.data_t = np.arange(n, dtype=np.float64) / max(1e-9, self.data_fps)
                self._set_status(f"Data FPS set to {self.data_fps:.3f}")
                self._draw_scene()
        except Exception:
            pass

    def _on_bin_submit(self, text: str) -> None:
        _ = text
        if self.fit_done:
            self.data_phi_hist_edges, self.data_phi_hist_counts, self.data_phi_peak_center = self._phi_distribution_from_xy(
                np.column_stack((self.fit_x, self.fit_y)),
                self._get_bin_deg(),
            )
            self._draw_scene()

    def _on_load_npy(self, _event) -> None:
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        path = filedialog.askopenfilename(
            title="Open NPY",
            filetypes=[("NumPy", "*.npy"), ("All", "*.*")],
            initialdir=str(Path.cwd()),
        )
        try:
            root.destroy()
        except Exception:
            pass
        if not path:
            return

        p = Path(path)
        try:
            xy, mode = self._load_npy_xy(p)
        except Exception as e:
            self._set_status(f"Load failed: {e}")
            return

        self.data_loaded = True
        self.fit_done = False
        self.data_name = p.name
        self.data_mode = mode
        self.data_xy = xy
        self.data_idx = 0

        self.data_fps = 1600.0 if mode == "inspection" else 77.0
        if self.tb_fps is not None:
            self.tb_fps.set_val(f"{self.data_fps:.3f}")
        self.data_t = np.arange(xy.shape[0], dtype=np.float64) / max(1e-9, self.data_fps)

        self._set_status(f"Loaded {p.name} ({mode}), frames={xy.shape[0]}, FPS={self.data_fps:.2f}")
        self.view_mode = "data"
        self._draw_scene()

    def _on_fit_data(self, _event) -> None:
        self._read_controls()
        self._fit_loaded_data()
        self.view_mode = "data"
        self._draw_scene()

    def _on_tab_sim(self, _event) -> None:
        self.view_mode = "sim"
        self._draw_scene()

    def _on_tab_data(self, _event) -> None:
        self.view_mode = "data"
        self._draw_scene()

    def _on_timer(self) -> None:
        if self.play_sim and self.s_phi is not None:
            nxt = (float(self.s_phi.val) + float(self.play_speed_deg)) % 360.0
            self.s_phi.set_val(nxt)

    def _draw_geometry_3d(self, alpha_deg: float, beta_deg: float, gamma_deg: float, phi_deg: float) -> None:
        self.ax3d.cla()

        k = self._rotation_axis(alpha_deg, beta_deg)
        cone = self._rod_directions(alpha_deg, beta_deg, gamma_deg, np.deg2rad(np.linspace(0.0, 360.0, 240)))
        u = self._rod_directions(alpha_deg, beta_deg, gamma_deg, np.array([math.radians(float(phi_deg))]))[0]

        plane = np.linspace(-0.9, 0.9, 2)
        px, py = np.meshgrid(plane, plane)
        pz = np.zeros_like(px)
        self.ax3d.plot_surface(px, py, pz, alpha=0.15, color="gray", linewidth=0)
        self.ax3d.quiver(0, 0, 0, 0, 0, 1, length=1.05, color="black", linewidth=2, arrow_length_ratio=0.08)
        self.ax3d.quiver(0, 0, 0, k[0], k[1], k[2], length=1.0, color="tab:red", linewidth=2, arrow_length_ratio=0.08)
        self.ax3d.plot(cone[:, 0], cone[:, 1], cone[:, 2], color="tab:orange", lw=1.5)
        self.ax3d.plot([0.0, u[0]], [0.0, u[1]], [0.0, u[2]], color="tab:blue", lw=3)
        self.ax3d.scatter([u[0]], [u[1]], [u[2]], color="tab:blue", s=36)

        self.ax3d.set_title("Rod / Cone Geometry")
        self.ax3d.set_xlabel("x")
        self.ax3d.set_ylabel("y")
        self.ax3d.set_zlabel("z")
        self.ax3d.set_xlim(-1.1, 1.1)
        self.ax3d.set_ylim(-1.1, 1.1)
        self.ax3d.set_zlim(-1.1, 1.1)
        self.ax3d.set_box_aspect((1, 1, 1))
    def _draw_sim_tab(self) -> None:
        self.ax_xy.cla()
        self.ax_phi.cla()

        if self.forward_curve is None:
            self._compute_forward_curve()

        self.ax_xy.plot(self.forward_curve.x, self.forward_curve.y, color="tab:blue", lw=2, label="XY(phi)")
        if self.block_inner_na and self.forward_curve_ref_no_block is not None:
            self.ax_xy.plot(
                self.forward_curve_ref_no_block.x,
                self.forward_curve_ref_no_block.y,
                color="0.45",
                ls="--",
                lw=1.5,
                label="XY(phi), no block",
            )
            d = np.sqrt(
                (self.forward_curve.x - self.forward_curve_ref_no_block.x) ** 2
                + (self.forward_curve.y - self.forward_curve_ref_no_block.y) ** 2
            )
            self.ax_xy.text(
                0.03,
                0.97,
                f"Inner block={self.block_inner_fraction:.3f}\nRMS dXY={float(np.sqrt(np.mean(d*d))):.4f}\nmax dXY={float(np.max(d)):.4f}",
                transform=self.ax_xy.transAxes,
                va="top",
                ha="left",
                fontsize=8,
                bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "0.8"},
            )

        cur = self._simulate_xy_for_params(
            self.alpha_deg,
            self.beta_deg,
            self.gamma_deg,
            np.array([self.phi_deg]),
        )
        self.ax_xy.scatter([cur.x[0]], [cur.y[0]], color="tab:red", s=40, zorder=5, label=f"phi={self.phi_deg:.1f} deg")
        self.ax_xy.axhline(0.0, color="0.85", lw=1)
        self.ax_xy.axvline(0.0, color="0.85", lw=1)
        self.ax_xy.set_aspect("equal", adjustable="box")
        self.ax_xy.set_xlabel("X")
        self.ax_xy.set_ylabel("Y")
        self.ax_xy.set_title("Simulation XY Loop")
        self.ax_xy.grid(alpha=0.2)
        self.ax_xy.legend(loc="best", fontsize=8)

        self.ax_phi.plot(self.forward_curve.phi_deg, self.forward_curve.x, label="X(phi)", color="tab:green", lw=1.6)
        self.ax_phi.plot(self.forward_curve.phi_deg, self.forward_curve.y, label="Y(phi)", color="tab:purple", lw=1.6)
        if self.block_inner_na and self.forward_curve_ref_no_block is not None:
            self.ax_phi.plot(
                self.forward_curve_ref_no_block.phi_deg,
                self.forward_curve_ref_no_block.x,
                label="X(phi), no block",
                color="tab:green",
                lw=1.3,
                ls=":",
            )
            self.ax_phi.plot(
                self.forward_curve_ref_no_block.phi_deg,
                self.forward_curve_ref_no_block.y,
                label="Y(phi), no block",
                color="tab:purple",
                lw=1.3,
                ls=":",
            )
        self.ax_phi.axvline(float(self.phi_deg), color="tab:red", linestyle="--", lw=1.2)
        self.ax_phi.set_xlim(0.0, 360.0)
        self.ax_phi.set_xlabel("phi (deg)")
        self.ax_phi.set_ylabel("value")
        self.ax_phi.set_title(
            f"Simulation: NA={self.na:.2f}, n={self.n_medium:.3f}, alpha={self.alpha_deg:.1f}, beta={self.beta_deg:.1f}, gamma={self.gamma_deg:.1f}, inner-block={'ON' if self.block_inner_na else 'OFF'}"
        )
        self.ax_phi.grid(alpha=0.25)
        self.ax_phi.legend(loc="best", fontsize=8)

        self._draw_geometry_3d(self.alpha_deg, self.beta_deg, self.gamma_deg, self.phi_deg)

    def _draw_data_tab(self) -> None:
        self.ax_xy.cla()
        self.ax_phi.cla()

        if not self.data_loaded:
            self.ax_xy.text(0.5, 0.5, "Load an NPY file first", ha="center", va="center", transform=self.ax_xy.transAxes)
            self.ax_phi.text(0.5, 0.5, "Then click Fit", ha="center", va="center", transform=self.ax_phi.transAxes)
            self._draw_geometry_3d(self.alpha_deg, self.beta_deg, self.gamma_deg, self.phi_deg)
            return

        n = int(self.data_xy.shape[0])
        if n > 8000:
            step = max(1, n // 8000)
            disp_idx = np.arange(0, n, step)
        else:
            disp_idx = np.arange(0, n)

        self.ax_xy.scatter(self.data_xy[disp_idx, 0], self.data_xy[disp_idx, 1], s=3, alpha=0.25, color="0.5", label="data XY")

        if self.fit_done:
            self.ax_xy.plot(self.fit_loop_x, self.fit_loop_y, color="tab:blue", lw=2.0, label="best-fit simulated XY loop")
            i = max(0, min(int(self.data_idx), n - 1))
            self.ax_xy.scatter([self.data_xy[i, 0]], [self.data_xy[i, 1]], color="tab:red", s=35, label="data now")
            self.ax_xy.scatter([self.fit_x[i]], [self.fit_y[i]], color="tab:orange", s=35, label="nearest on loop")

        self.ax_xy.axhline(0.0, color="0.85", lw=1)
        self.ax_xy.axvline(0.0, color="0.85", lw=1)
        self.ax_xy.set_aspect("equal", adjustable="box")
        self.ax_xy.set_xlabel("X")
        self.ax_xy.set_ylabel("Y")
        self.ax_xy.set_title(f"Data XY ({self.data_name}, {self.data_mode})")
        self.ax_xy.grid(alpha=0.2)
        self.ax_xy.legend(loc="best", fontsize=8)

        if self.fit_done and self.fit_phi_wrapped.size > 0:
            t_plot = self.data_t.copy()
            phi_plot = self.fit_phi_wrapped.copy()
            if phi_plot.size >= 2:
                jumps = np.abs(np.diff(phi_plot))
                jump_idx = np.where(jumps > 180.0)[0]
                if jump_idx.size > 0:
                    t_plot = t_plot.astype(np.float64, copy=True)
                    phi_plot = phi_plot.astype(np.float64, copy=True)
                    phi_plot[jump_idx + 1] = np.nan

            self.ax_phi.plot(t_plot, phi_plot, color="tab:blue", lw=1.2, label="phi(t) from data + fitted geometry")
            i = max(0, min(int(self.data_idx), int(self.fit_phi_wrapped.size) - 1))
            self.ax_phi.axvline(float(self.data_t[i]), color="tab:red", linestyle="--", lw=1.0)
            self.ax_phi.scatter([self.data_t[i]], [self.fit_phi_wrapped[i]], color="tab:red", s=20)
            self.ax_phi.set_xlabel("time (s)")
            self.ax_phi.set_ylabel("phi (deg, wrapped)")
            self.ax_phi.set_ylim(0.0, 360.0)
            self.ax_phi.grid(alpha=0.25)
            self.ax_phi.legend(loc="upper left", fontsize=8)
            self.ax_phi.set_title(
                f"phi(t) from loaded data   alpha={self.fit_alpha:.2f}, beta={self.fit_beta:.2f}, gamma={self.fit_gamma:.2f}, cost={self.fit_cost:.6f}"
            )

            ax_hist = inset_axes(self.ax_phi, width="33%", height="60%", loc="upper right", borderpad=1.0)
            edges = self.data_phi_hist_edges
            counts = self.data_phi_hist_counts
            centers = 0.5 * (edges[:-1] + edges[1:])
            bw = float(edges[1] - edges[0]) if edges.size > 1 else 9.0
            ax_hist.bar(centers, counts, width=0.9 * bw, color="tab:purple", alpha=0.8)
            if self.data_phi_peak_center is not None:
                peak = float(self.data_phi_peak_center)
                ax_hist.axvline(peak, color="black", linestyle=":", lw=1.3)
                for k in range(10):
                    ax_hist.axvline((peak + 36.0 * k) % 360.0, color="0.4", linestyle=":", lw=0.9)
            ax_hist.set_xlim(0.0, 360.0)
            ax_hist.set_title("phi dist", fontsize=8)
            ax_hist.tick_params(labelsize=7)
        else:
            self.ax_phi.text(0.5, 0.5, "Click Fit to estimate axis and phi(t)", ha="center", va="center", transform=self.ax_phi.transAxes)
            self.ax_phi.set_title("Data Fit")

        if self.fit_done and self.fit_phi_wrapped.size > 0:
            i = max(0, min(int(self.data_idx), int(self.fit_phi_wrapped.size) - 1))
            phi_now = float(self.fit_phi_wrapped[i])
            self._draw_geometry_3d(self.fit_alpha, self.fit_beta, self.fit_gamma, phi_now)
        else:
            self._draw_geometry_3d(self.alpha_deg, self.beta_deg, self.gamma_deg, self.phi_deg)

    def _draw_scene(self) -> None:
        if self.fig is None:
            return
        if self.view_mode == "data":
            self._draw_data_tab()
        else:
            self._draw_sim_tab()
        self.fig.canvas.draw_idle()

    def _recompute_all(self) -> None:
        self._build_collection_rays(exclude_inner=self.block_inner_na)
        self._compute_forward_curve()
        if self.fit_done and self.data_loaded:
            self._fit_loaded_data()
        self._draw_scene()

    def build_ui(self) -> None:
        self.fig = plt.figure(figsize=(14.2, 8.8))
        self.fig.suptitle("Rod Dipole Simulator + Data Fit", fontsize=14)
        self.status_text = self.fig.text(0.05, 0.935, "Ready", fontsize=9)

        gs = self.fig.add_gridspec(
            2,
            2,
            left=0.05,
            right=0.98,
            bottom=0.27,
            top=0.90,
            width_ratios=[1.15, 1.15],
            height_ratios=[1.0, 1.0],
            hspace=0.28,
            wspace=0.2,
        )
        self.ax3d = self.fig.add_subplot(gs[:, 0], projection="3d")
        self.ax_xy = self.fig.add_subplot(gs[0, 1])
        self.ax_phi = self.fig.add_subplot(gs[1, 1])

        ax_alpha = self.fig.add_axes([0.08, 0.19, 0.32, 0.03])
        ax_beta = self.fig.add_axes([0.08, 0.15, 0.32, 0.03])
        ax_gamma = self.fig.add_axes([0.08, 0.11, 0.32, 0.03])
        ax_phi = self.fig.add_axes([0.50, 0.19, 0.32, 0.03])
        ax_na = self.fig.add_axes([0.50, 0.15, 0.32, 0.03])

        self.s_alpha = Slider(ax_alpha, "alpha (deg)", 0.0, 90.0, valinit=self.alpha_deg, valstep=0.1)
        self.s_beta = Slider(ax_beta, "beta (deg)", 0.0, 360.0, valinit=self.beta_deg, valstep=0.1)
        self.s_gamma = Slider(ax_gamma, "gamma (deg)", 0.0, 89.0, valinit=self.gamma_deg, valstep=0.1)
        self.s_phi = Slider(ax_phi, "phi (deg)", 0.0, 360.0, valinit=self.phi_deg, valstep=0.1)
        self.s_na = Slider(ax_na, "NA", 0.05, self.n_medium - 0.001, valinit=self.na, valstep=0.001)

        self.s_alpha.on_changed(self._on_geometry_slider)
        self.s_beta.on_changed(self._on_geometry_slider)
        self.s_gamma.on_changed(self._on_geometry_slider)
        self.s_na.on_changed(self._on_geometry_slider)
        self.s_phi.on_changed(self._on_phi_slider)

        ax_play_sim = self.fig.add_axes([0.50, 0.10, 0.10, 0.04])
        ax_recompute = self.fig.add_axes([0.62, 0.10, 0.13, 0.04])
        ax_load = self.fig.add_axes([0.77, 0.10, 0.10, 0.04])
        ax_fit = self.fig.add_axes([0.89, 0.10, 0.08, 0.04])
        ax_tab_sim = self.fig.add_axes([0.50, 0.05, 0.10, 0.04])
        ax_tab_data = self.fig.add_axes([0.62, 0.05, 0.10, 0.04])

        self.btn_play_sim = Button(ax_play_sim, "Play Sim")
        self.btn_recompute = Button(ax_recompute, "Recompute")
        self.btn_load = Button(ax_load, "Load NPY")
        self.btn_fit = Button(ax_fit, "Fit")
        self.btn_tab_sim = Button(ax_tab_sim, "Sim Tab")
        self.btn_tab_data = Button(ax_tab_data, "Data Tab")

        self.btn_play_sim.on_clicked(self._on_toggle_play_sim)
        self.btn_recompute.on_clicked(self._on_recompute_button)
        self.btn_load.on_clicked(self._on_load_npy)
        self.btn_fit.on_clicked(self._on_fit_data)
        self.btn_tab_sim.on_clicked(self._on_tab_sim)
        self.btn_tab_data.on_clicked(self._on_tab_data)

        ax_block = self.fig.add_axes([0.08, 0.05, 0.28, 0.07])
        ax_cut = self.fig.add_axes([0.37, 0.05, 0.10, 0.04])
        ax_fps = self.fig.add_axes([0.86, 0.05, 0.06, 0.04])
        ax_bin = self.fig.add_axes([0.93, 0.05, 0.05, 0.04])

        self.chk_block = CheckButtons(ax_block, ["Block inner pupil"], [self.block_inner_na])
        self.tb_cutout = TextBox(ax_cut, "cutout", initial=f"{self.block_inner_fraction:.3f}")
        self.tb_fps = TextBox(ax_fps, "fps", initial=f"{self.data_fps:.1f}")
        self.tb_bin = TextBox(ax_bin, "bin", initial="9")

        self.chk_block.on_clicked(self._on_toggle_block)
        self.tb_cutout.on_submit(self._on_cutout_submit)
        self.tb_fps.on_submit(self._on_fps_submit)
        self.tb_bin.on_submit(self._on_bin_submit)

        self.timer = self.fig.canvas.new_timer(interval=40)
        self.timer.add_callback(self._on_timer)
        self.timer.start()

        self._recompute_all()

    def show(self) -> None:
        self.build_ui()
        plt.show()


def main() -> None:
    RodDipoleSimulator().show()


if __name__ == "__main__":
    main()
