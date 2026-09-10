from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from skimage import io
from torch.utils.data import Dataset
try:
    import cv2
except Exception:  # pragma: no cover - runtime guard
    cv2 = None


def _load_image(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        arr = np.load(path)
        if arr.ndim == 3:
            arr = arr.mean(axis=0)
        return arr.astype(np.float32)
    arr = io.imread(path).astype(np.float32)
    if arr.ndim == 3:
        arr = arr[..., 0]
    return arr


def gather_files(root: Path, exts: Sequence[str]) -> list[Path]:
    exts_lower = {e.lower() for e in exts}
    files = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts_lower:
            files.append(p)
    return sorted(files)


def load_single_image(path: Path) -> np.ndarray:
    return _load_image(path)


def build_triplets(
    target_dir: Path,
    masked_dir: Path,
    mask_dir: Path,
) -> list[tuple[Path, Path, Path]]:
    triplets: list[tuple[Path, Path, Path]] = []
    targets = sorted(target_dir.glob("*.npy"))
    for t in targets:
        stem = t.stem
        m_in = masked_dir / f"{stem}_masked.npy"
        m = mask_dir / f"{stem}_mask.npy"
        if m_in.exists() and m.exists():
            triplets.append((t, m_in, m))
    return triplets


class PrecomputedMaskedDataset(Dataset):
    """
    Dataset for pre-generated pairs:
    - target full background
    - masked input (target with masked pixels removed)
    - mask (1=hidden)
    """

    def __init__(
        self,
        triplets: Sequence[tuple[Path, Path, Path]],
        normalize_percentile: float = 99.5,
        normalization_mode: str = "fixed_global",
        fixed_scale: float = 1.0,
        patch_size: int | None = None,
        samples_per_image: int = 1,
        prefer_mask_center: bool = True,
        target_mode: str = "smooth_background",
        smooth_sigma: float = 32.0,
        smooth_mask_aware: bool = True,
        input_mode: str = "raw_masked",
    ) -> None:
        self.triplets = list(triplets)
        self.normalize_percentile = float(normalize_percentile)
        self.normalization_mode = str(normalization_mode).strip().lower()
        self.fixed_scale = float(fixed_scale)
        self.patch_size = int(patch_size) if patch_size is not None and int(patch_size) > 0 else None
        self.samples_per_image = max(1, int(samples_per_image))
        self.prefer_mask_center = bool(prefer_mask_center)
        self.target_mode = str(target_mode).strip().lower()
        self.smooth_sigma = float(smooth_sigma)
        self.smooth_mask_aware = bool(smooth_mask_aware)
        self.input_mode = str(input_mode).strip().lower()
        self._printed_norm_debug = False
        self._printed_mask_debug = False
        if not self.triplets:
            raise ValueError("No precomputed triplets were provided.")
        if self.target_mode not in {"raw", "smooth_background"}:
            raise ValueError(f"Unsupported target_mode: {self.target_mode}")
        if self.input_mode not in {"raw_masked", "smooth_masked"}:
            raise ValueError(f"Unsupported input_mode: {self.input_mode}")
        if self.normalization_mode not in {"fixed_global", "per_image_percentile"}:
            raise ValueError(f"Unsupported normalization_mode: {self.normalization_mode}")
        if self.target_mode == "smooth_background" and cv2 is None:
            raise RuntimeError("OpenCV is required for smooth_background target mode.")

    def __len__(self) -> int:
        return len(self.triplets) * self.samples_per_image

    def _sample_patch_coords(self, h: int, w: int, mask: np.ndarray, ps: int) -> tuple[int, int]:
        if h <= ps:
            y0 = 0
        else:
            y0 = np.random.randint(0, h - ps + 1)
        if w <= ps:
            x0 = 0
        else:
            x0 = np.random.randint(0, w - ps + 1)

        if not self.prefer_mask_center:
            return y0, x0

        ys, xs = np.where(mask > 0.5)
        if ys.size == 0:
            return y0, x0
        k = np.random.randint(0, ys.size)
        cy = int(ys[k])
        cx = int(xs[k])
        y0 = max(0, min(cy - ps // 2, max(0, h - ps)))
        x0 = max(0, min(cx - ps // 2, max(0, w - ps)))
        return y0, x0

    def _smooth_background(self, target_raw: np.ndarray, mask: np.ndarray) -> np.ndarray:
        if cv2 is None:
            return target_raw
        sigma = max(0.1, float(self.smooth_sigma))
        known = 1.0 - mask
        if self.smooth_mask_aware:
            numerator = cv2.GaussianBlur(
                (target_raw * known).astype(np.float32),
                (0, 0),
                sigmaX=sigma,
                sigmaY=sigma,
                borderType=cv2.BORDER_REFLECT,
            )
            denominator = cv2.GaussianBlur(
                known.astype(np.float32),
                (0, 0),
                sigmaX=sigma,
                sigmaY=sigma,
                borderType=cv2.BORDER_REFLECT,
            )
            out = numerator / (denominator + 1e-8)
            return out.astype(np.float32)

        return cv2.GaussianBlur(
            target_raw.astype(np.float32),
            (0, 0),
            sigmaX=sigma,
            sigmaY=sigma,
            borderType=cv2.BORDER_REFLECT,
        ).astype(np.float32)

    @staticmethod
    def _canonicalize_hole_mask(mask_raw: np.ndarray, masked_raw: np.ndarray) -> tuple[np.ndarray, bool]:
        mask_bin = (mask_raw > 0.5).astype(np.float32)
        ones_sel = mask_bin > 0.5
        zeros_sel = ~ones_sel

        ones_abs = float(np.mean(np.abs(masked_raw[ones_sel]))) if np.any(ones_sel) else np.inf
        zeros_abs = float(np.mean(np.abs(masked_raw[zeros_sel]))) if np.any(zeros_sel) else np.inf

        if np.isfinite(ones_abs) and np.isfinite(zeros_abs):
            if zeros_abs + 1e-8 < ones_abs:
                ones_are_holes = False
            elif ones_abs + 1e-8 < zeros_abs:
                ones_are_holes = True
            else:
                ones_are_holes = float(mask_bin.mean()) <= 0.5
        else:
            ones_are_holes = float(mask_bin.mean()) <= 0.5

        hole_mask = mask_bin if ones_are_holes else (1.0 - mask_bin)
        return hole_mask.astype(np.float32, copy=False), bool(ones_are_holes)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        t_path, mi_path, m_path = self.triplets[idx % len(self.triplets)]
        target_raw = np.load(t_path, mmap_mode="r").astype(np.float32)
        masked_raw = np.load(mi_path, mmap_mode="r").astype(np.float32)
        mask_raw = np.load(m_path, mmap_mode="r").astype(np.float32)
        if target_raw.shape != masked_raw.shape:
            raise ValueError(
                f"Masked input shape {masked_raw.shape} does not match target shape {target_raw.shape} for {t_path.name}"
            )
        if target_raw.shape != mask_raw.shape:
            raise ValueError(f"Mask shape {mask_raw.shape} does not match target shape {target_raw.shape} for {t_path.name}")
        mask, ones_are_holes = self._canonicalize_hole_mask(mask_raw, masked_raw)
        target_smooth = self._smooth_background(target_raw, mask)
        target = target_smooth if self.target_mode == "smooth_background" else target_raw

        if self.patch_size is not None:
            h, w = target.shape
            ps = self.patch_size
            if h < ps or w < ps:
                pad_h = max(0, ps - h)
                pad_w = max(0, ps - w)
                target_raw = np.pad(target_raw, ((0, pad_h), (0, pad_w)), mode="reflect")
                masked_raw = np.pad(masked_raw, ((0, pad_h), (0, pad_w)), mode="reflect")
                target_smooth = np.pad(target_smooth, ((0, pad_h), (0, pad_w)), mode="reflect")
                target = np.pad(target, ((0, pad_h), (0, pad_w)), mode="reflect")
                mask = np.pad(mask, ((0, pad_h), (0, pad_w)), mode="constant")
                h, w = target.shape
            y0, x0 = self._sample_patch_coords(h, w, mask, ps)
            target_raw = target_raw[y0 : y0 + ps, x0 : x0 + ps]
            masked_raw = masked_raw[y0 : y0 + ps, x0 : x0 + ps]
            target_smooth = target_smooth[y0 : y0 + ps, x0 : x0 + ps]
            target = target[y0 : y0 + ps, x0 : x0 + ps]
            mask = mask[y0 : y0 + ps, x0 : x0 + ps]

        if self.normalization_mode == "fixed_global":
            scale = float(max(self.fixed_scale, 1e-6))
            raw_max = float(np.max(target_raw))
            # Safety guard: if data already appears float-normalized, avoid collapsing values
            # by dividing by a large integer-like fixed scale.
            if scale > 1.0 and raw_max <= 2.0:
                scale = 1.0
                if not self._printed_norm_debug:
                    print(
                        "[dataset] fixed_global guard activated: "
                        f"raw_max={raw_max:.6f} with fixed_scale={self.fixed_scale}; using scale=1.0"
                    )
        else:
            scale = float(np.percentile(target_raw, self.normalize_percentile))
            scale = max(scale, 1e-6)
        target_raw_n = target_raw / scale
        masked_raw_n = masked_raw / scale
        target_smooth_n = target_smooth / scale
        target_n = target / scale
        if self.input_mode == "smooth_masked":
            masked_n = target_smooth_n * (1.0 - mask)
        else:
            masked_n = masked_raw_n

        if not self._printed_norm_debug:
            print(
                "[dataset] pre-norm target_raw stats: "
                f"min={float(np.min(target_raw)):.6f} max={float(np.max(target_raw)):.6f} mean={float(np.mean(target_raw)):.6f}"
            )
            print(
                "[dataset] post-norm target stats: "
                f"min={float(np.min(target_n)):.6f} max={float(np.max(target_n)):.6f} mean={float(np.mean(target_n)):.6f} "
                f"(scale={scale:.6f}, mode={self.normalization_mode})"
            )
            self._printed_norm_debug = True
        if not self._printed_mask_debug:
            print(
                "[dataset] inferred hole-mask polarity: "
                f"{'1=hole' if ones_are_holes else '0=hole'} for {t_path.name}"
            )
            self._printed_mask_debug = True

        inp = np.stack([masked_n, mask], axis=0)
        return {
            "input": torch.from_numpy(inp.astype(np.float32)),
            "target": torch.from_numpy(target_n[None, ...].astype(np.float32)),
            "mask": torch.from_numpy(mask[None, ...].astype(np.float32)),
            "target_raw": torch.from_numpy(target_raw_n[None, ...].astype(np.float32)),
            "target_smooth": torch.from_numpy(target_smooth_n[None, ...].astype(np.float32)),
            "scale": torch.tensor([scale], dtype=torch.float32),
            "target_path": str(t_path),
            "masked_path": str(mi_path),
            "mask_path": str(m_path),
        }
