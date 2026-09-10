#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import re
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

try:
    import cv2
except Exception:  # pragma: no cover - handled at runtime
    cv2 = None

try:
    import imageio.v3 as iio
except Exception:  # pragma: no cover - handled at runtime
    iio = None

try:
    from skimage.metrics import structural_similarity
except Exception:  # pragma: no cover - handled at runtime
    structural_similarity = None

try:
    from skimage.restoration import inpaint_biharmonic
except Exception:  # pragma: no cover - handled at runtime
    inpaint_biharmonic = None

try:
    import torch
    import torch.nn.functional as torch_F
except Exception:  # pragma: no cover - handled at runtime
    torch = None
    torch_F = None


SUPPORTED_EXTS = {".npy", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


@dataclass
class Sample:
    sample_id: str
    target: np.ndarray
    mask: np.ndarray
    target_min: float
    target_max: float
    target_norm: np.ndarray
    masked_input_norm: np.ndarray


@dataclass
class MethodAggregate:
    method: str
    masked_mae: float
    masked_rmse: float
    masked_psnr: float
    masked_ssim: float
    mean_runtime_s: float


class QuitController:
    def __init__(self, enabled: bool) -> None:
        self.enabled = bool(enabled) and sys.stdin is not None and sys.stdin.isatty()
        self._event = threading.Event()
        self._thread: threading.Thread | None = None
        if self.enabled:
            self._thread = threading.Thread(target=self._listen_for_quit, daemon=True)
            self._thread.start()
            print("Press 'q' then Enter at any time to stop after the current image.")

    def _listen_for_quit(self) -> None:
        while not self._event.is_set():
            try:
                line = input()
            except EOFError:
                return
            except Exception:
                return
            if line.strip().lower() == "q":
                self._event.set()
                print("[STOP] Quit requested by user.")
                return

    def should_stop(self) -> bool:
        return self._event.is_set()


class InpaintMethod:
    def __init__(self, name: str) -> None:
        self.name = name
        self.skip_reason: str | None = None

    def predict(self, sample: Sample) -> np.ndarray:
        raise NotImplementedError


class TeleaMethod(InpaintMethod):
    def __init__(self, radius: float) -> None:
        super().__init__("OpenCV Telea")
        self.radius = float(radius)
        if cv2 is None:
            self.skip_reason = "OpenCV is not available."

    def predict(self, sample: Sample) -> np.ndarray:
        if cv2 is None:
            raise RuntimeError("OpenCV not available")
        mask_u8 = (sample.mask * 255.0).astype(np.uint8)
        src_u8 = np.clip(np.round(sample.masked_input_norm * 255.0), 0, 255).astype(np.uint8)
        pred = cv2.inpaint(
            src_u8,
            mask_u8,
            self.radius,
            cv2.INPAINT_TELEA,
        )
        return np.clip(pred.astype(np.float32) / 255.0, 0.0, 1.0)


class NavierStokesMethod(InpaintMethod):
    def __init__(self, radius: float) -> None:
        super().__init__("OpenCV Navier-Stokes")
        self.radius = float(radius)
        if cv2 is None:
            self.skip_reason = "OpenCV is not available."

    def predict(self, sample: Sample) -> np.ndarray:
        if cv2 is None:
            raise RuntimeError("OpenCV not available")
        mask_u8 = (sample.mask * 255.0).astype(np.uint8)
        src_u8 = np.clip(np.round(sample.masked_input_norm * 255.0), 0, 255).astype(np.uint8)
        pred = cv2.inpaint(
            src_u8,
            mask_u8,
            self.radius,
            cv2.INPAINT_NS,
        )
        return np.clip(pred.astype(np.float32) / 255.0, 0.0, 1.0)


class BiharmonicMethod(InpaintMethod):
    def __init__(self) -> None:
        super().__init__("scikit-image biharmonic")
        if inpaint_biharmonic is None:
            self.skip_reason = "scikit-image inpaint_biharmonic is not available."

    def predict(self, sample: Sample) -> np.ndarray:
        if inpaint_biharmonic is None:
            raise RuntimeError("inpaint_biharmonic unavailable")
        pred = inpaint_biharmonic(sample.masked_input_norm.astype(np.float32), sample.mask > 0.5)
        return np.clip(pred.astype(np.float32), 0.0, 1.0)


class ExternalCommandMethod(InpaintMethod):
    def __init__(
        self,
        name: str,
        command_template: str | None,
        repo_dir: Path | None = None,
        extra_context: dict[str, str] | None = None,
    ) -> None:
        super().__init__(name)
        self.command_template = command_template.strip() if command_template else ""
        self.repo_dir = repo_dir
        self.extra_context = dict(extra_context or {})
        if iio is None:
            self.skip_reason = "imageio is not available for file exchange with external model."
        elif not self.command_template:
            self.skip_reason = "No command template configured."
        elif repo_dir is not None and not repo_dir.exists():
            self.skip_reason = f"Repository path does not exist: {repo_dir}"

    def _find_output(self, output_path: Path, output_dir: Path, sample_id: str) -> Path:
        if output_path.exists():
            return output_path
        if not output_dir.exists():
            raise FileNotFoundError(f"Output not found: {output_path}")
        image_candidates: list[Path] = []
        for p in output_dir.rglob("*"):
            if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS and p.suffix.lower() != ".npy":
                image_candidates.append(p)
        if not image_candidates:
            raise FileNotFoundError(f"No image output produced in {output_dir}")
        score_sorted = sorted(
            image_candidates,
            key=lambda p: (
                0 if sample_id in p.stem else 1,
                1 if "mask" in p.stem.lower() else 0,
                len(str(p)),
            ),
        )
        return score_sorted[0]

    def predict(self, sample: Sample) -> np.ndarray:
        if self.skip_reason is not None:
            raise RuntimeError(self.skip_reason)
        assert iio is not None

        with tempfile.TemporaryDirectory(prefix=f"bench_{sanitize_name(self.name)}_") as td:
            root = Path(td)
            input_dir = root / "input"
            mask_dir = root / "masks"
            output_dir = root / "output"
            input_dir.mkdir(parents=True, exist_ok=True)
            mask_dir.mkdir(parents=True, exist_ok=True)
            output_dir.mkdir(parents=True, exist_ok=True)

            stem = sample.sample_id
            image_path = input_dir / f"{stem}.png"
            mask_path = mask_dir / f"{stem}_mask001.png"
            mask_path_in_input = input_dir / f"{stem}_mask001.png"
            output_path = output_dir / f"{stem}.png"

            gray_u8 = np.clip(np.round(sample.masked_input_norm * 255.0), 0, 255).astype(np.uint8)
            gray_rgb = np.repeat(gray_u8[..., None], 3, axis=-1)
            mask_u8 = (sample.mask * 255.0).astype(np.uint8)
            iio.imwrite(image_path, gray_rgb)
            iio.imwrite(mask_path, mask_u8)
            iio.imwrite(mask_path_in_input, mask_u8)

            command = self.command_template.format(
                python=sys.executable,
                image=image_path,
                mask=mask_path,
                output=output_path,
                output_dir=output_dir,
                input_dir=input_dir,
                mask_dir=mask_dir,
                stem=stem,
                repo=str(self.repo_dir) if self.repo_dir else "",
                **self.extra_context,
            )
            run_cwd = self.repo_dir if self.repo_dir else Path.cwd()
            proc = subprocess.run(
                command,
                shell=True,
                cwd=run_cwd,
                capture_output=True,
                text=True,
            )
            if proc.returncode != 0:
                raise RuntimeError(
                    f"Command failed ({proc.returncode})\n"
                    f"Command: {command}\n"
                    f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
                )

            pred_path = self._find_output(output_path, output_dir, stem)
            pred = load_image_any(pred_path)
            if pred.ndim == 3:
                pred = pred.mean(axis=-1)
            pred = pred.astype(np.float32)
            if pred.max() > 1.0 + 1e-6:
                pred = pred / 255.0
            return np.clip(pred, 0.0, 1.0)


class PartialConvNaotoMethod(InpaintMethod):
    def __init__(self, repo_dir: Path, checkpoint_path: Path, force_cpu: bool = True) -> None:
        super().__init__("NVIDIA Partial Convolution")
        self.repo_dir = repo_dir
        self.checkpoint_path = checkpoint_path
        self.force_cpu = bool(force_cpu)
        self._model = None
        self._opt_module = None
        self._device = None

        if torch is None or torch_F is None:
            self.skip_reason = "PyTorch is not available."
            return
        if not self.repo_dir.exists():
            self.skip_reason = f"PartialConv repo not found: {self.repo_dir}"
            return
        if not self.checkpoint_path.exists():
            self.skip_reason = f"PartialConv checkpoint not found: {self.checkpoint_path}"
            return

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        assert torch is not None
        if str(self.repo_dir) not in sys.path:
            sys.path.insert(0, str(self.repo_dir))
        try:
            from net import PConvUNet  # type: ignore[import-not-found]
            import opt as pconv_opt  # type: ignore[import-not-found]
        except Exception as exc:
            raise RuntimeError(f"Failed importing PartialConv modules from {self.repo_dir}: {exc}") from exc

        device = torch.device("cpu" if self.force_cpu or not torch.cuda.is_available() else "cuda")
        model = PConvUNet()
        ckpt = torch.load(self.checkpoint_path, map_location=device, weights_only=False)
        if not isinstance(ckpt, dict) or "model" not in ckpt:
            raise RuntimeError("Unexpected PartialConv checkpoint format; expected key 'model'.")
        model.load_state_dict(ckpt["model"], strict=False)
        model.to(device)
        model.eval()
        self._model = model
        self._opt_module = pconv_opt
        self._device = device

    def predict(self, sample: Sample) -> np.ndarray:
        self._ensure_model()
        assert torch is not None and torch_F is not None
        assert self._model is not None and self._opt_module is not None and self._device is not None

        masked = sample.masked_input_norm.astype(np.float32)
        valid = (1.0 - sample.mask).astype(np.float32)  # PartialConv expects 1=valid, 0=hole.
        rgb = np.repeat(masked[..., None], 3, axis=-1)
        valid3 = np.repeat(valid[..., None], 3, axis=-1)

        x = torch.from_numpy(rgb).permute(2, 0, 1).float()
        m = torch.from_numpy(valid3).permute(2, 0, 1).float()
        for c, (mu, sd) in enumerate(zip(self._opt_module.MEAN, self._opt_module.STD)):
            x[c] = (x[c] - float(mu)) / float(sd)
        x = x * m

        h, w = int(x.shape[1]), int(x.shape[2])
        stride = 128
        hp = ((h + stride - 1) // stride) * stride
        wp = ((w + stride - 1) // stride) * stride
        pad_h = hp - h
        pad_w = wp - w
        x = x.unsqueeze(0).to(self._device)
        m = m.unsqueeze(0).to(self._device)
        if pad_h or pad_w:
            x = torch_F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
            m = torch_F.pad(m, (0, pad_w, 0, pad_h), mode="constant", value=1.0)

        with torch.no_grad():
            out, _ = self._model(x, m)
        out = out[:, :, :h, :w].squeeze(0).detach().cpu()
        for c, (mu, sd) in enumerate(zip(self._opt_module.MEAN, self._opt_module.STD)):
            out[c] = out[c] * float(sd) + float(mu)
        out = out.clamp(0.0, 1.0)
        out_gray = out.mean(dim=0).numpy().astype(np.float32)
        return np.clip(out_gray, 0.0, 1.0)


def sanitize_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name.strip())


def load_image_any(path: Path) -> np.ndarray:
    ext = path.suffix.lower()
    if ext == ".npy":
        arr = np.load(path)
    else:
        if iio is None:
            raise RuntimeError("imageio is required for non-NPY image files.")
        arr = iio.imread(path)
    arr = np.asarray(arr)
    if arr.ndim == 2:
        out = arr.astype(np.float32)
    elif arr.ndim == 3:
        if arr.shape[-1] in (1, 3, 4):
            out = arr[..., 0].astype(np.float32)
        elif arr.shape[0] in (1, 3, 4):
            out = arr[0].astype(np.float32)
        else:
            out = arr.mean(axis=0).astype(np.float32)
    else:
        raise ValueError(f"Unsupported image shape {arr.shape} for {path}")
    return out


def normalize_image(img: np.ndarray) -> tuple[np.ndarray, float, float]:
    mn = float(np.nanmin(img))
    mx = float(np.nanmax(img))
    if not np.isfinite(mn) or not np.isfinite(mx):
        raise ValueError("Image contains non-finite values.")
    if mx <= mn + 1e-12:
        return np.zeros_like(img, dtype=np.float32), mn, mx
    norm = (img - mn) / (mx - mn)
    return norm.astype(np.float32), mn, mx


def denormalize_image(img_norm: np.ndarray, mn: float, mx: float) -> np.ndarray:
    if mx <= mn + 1e-12:
        return np.full_like(img_norm, fill_value=mn, dtype=np.float32)
    out = img_norm * (mx - mn) + mn
    return out.astype(np.float32)


def load_samples(target_dir: Path, mask_dir: Path, max_images: int | None = None) -> list[Sample]:
    if not target_dir.exists():
        raise FileNotFoundError(f"Target directory not found: {target_dir}")
    if not mask_dir.exists():
        raise FileNotFoundError(f"Mask directory not found: {mask_dir}")

    target_files = sorted(
        p for p in target_dir.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS and "_mask" not in p.stem
    )
    if max_images is not None:
        target_files = target_files[: max(0, int(max_images))]
    samples: list[Sample] = []

    for t_path in target_files:
        stem = t_path.stem
        mask_candidates = [mask_dir / f"{stem}_mask.npy", mask_dir / f"{stem}_mask.png", mask_dir / f"{stem}.npy", mask_dir / f"{stem}.png"]
        m_path = next((p for p in mask_candidates if p.exists()), None)
        if m_path is None:
            continue

        target = load_image_any(t_path).astype(np.float32)
        mask_raw = load_image_any(m_path).astype(np.float32)
        mask = (mask_raw > 0.5).astype(np.float32)
        if target.shape != mask.shape:
            raise ValueError(f"Shape mismatch for {t_path.name}: target {target.shape}, mask {mask.shape}")
        if float(mask.sum()) <= 0.0:
            continue

        target_norm, mn, mx = normalize_image(target)
        masked_input_norm = target_norm * (1.0 - mask)
        samples.append(
            Sample(
                sample_id=stem,
                target=target,
                mask=mask,
                target_min=mn,
                target_max=mx,
                target_norm=target_norm,
                masked_input_norm=masked_input_norm.astype(np.float32),
            )
        )

    if not samples:
        raise RuntimeError("No valid target/mask pairs were found.")
    return samples


def masked_metrics(pred: np.ndarray, target: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    pred = pred.astype(np.float32)
    target = target.astype(np.float32)
    mask = mask.astype(np.float32)
    denom = float(mask.sum())
    if denom <= 0.0:
        return {
            "masked_mae": math.nan,
            "masked_rmse": math.nan,
            "masked_psnr": math.nan,
            "masked_ssim": math.nan,
        }

    err = np.abs(pred - target)
    mae = float((err * mask).sum() / denom)
    rmse = float(np.sqrt((((pred - target) ** 2) * mask).sum() / denom))
    data_range = float(max(target.max() - target.min(), 1e-8))
    psnr = float("inf") if rmse <= 1e-12 else float(20.0 * np.log10(data_range / rmse))

    ssim_val = math.nan
    if structural_similarity is not None:
        ys, xs = np.where(mask > 0.5)
        if ys.size > 0 and xs.size > 0:
            y0 = max(0, int(ys.min()) - 4)
            y1 = min(target.shape[0], int(ys.max()) + 5)
            x0 = max(0, int(xs.min()) - 4)
            x1 = min(target.shape[1], int(xs.max()) + 5)
            target_roi = target[y0:y1, x0:x1]
            pred_roi = pred[y0:y1, x0:x1]
            mask_roi = mask[y0:y1, x0:x1]
            pred_composite = target_roi * (1.0 - mask_roi) + pred_roi * mask_roi
            try:
                ssim_val = float(
                    structural_similarity(
                        target_roi,
                        pred_composite,
                        data_range=max(float(target_roi.max() - target_roi.min()), 1e-8),
                    )
                )
            except Exception:
                ssim_val = math.nan

    return {
        "masked_mae": mae,
        "masked_rmse": rmse,
        "masked_psnr": psnr,
        "masked_ssim": ssim_val,
    }


def save_png(path: Path, img_01: np.ndarray) -> None:
    if iio is None:
        return
    arr = np.clip(np.round(img_01 * 255.0), 0, 255).astype(np.uint8)
    iio.imwrite(path, arr)


def save_sample_visuals(sample: Sample, out_root: Path) -> None:
    sample_dir = out_root / sanitize_name(sample.sample_id)
    sample_dir.mkdir(parents=True, exist_ok=True)
    save_png(sample_dir / "original.png", sample.target_norm)
    save_png(sample_dir / "mask.png", sample.mask)
    save_png(sample_dir / "masked_input.png", sample.masked_input_norm)


def save_method_visuals(sample: Sample, method_name: str, pred: np.ndarray, out_root: Path) -> None:
    sample_dir = out_root / sanitize_name(sample.sample_id) / sanitize_name(method_name)
    sample_dir.mkdir(parents=True, exist_ok=True)
    pred_norm, _, _ = normalize_image(pred)
    save_png(sample_dir / "method_output.png", pred_norm)

    abs_err = np.abs(pred - sample.target) * sample.mask
    max_err = float(abs_err.max())
    if max_err <= 1e-12:
        err_viz = np.zeros_like(abs_err, dtype=np.float32)
    else:
        err_viz = abs_err / max_err
    save_png(sample_dir / "absolute_error_inside_mask.png", err_viz)


def aggregate(values: list[float]) -> float:
    finite = [v for v in values if np.isfinite(v)]
    if not finite:
        return math.nan
    return float(np.mean(finite))


def resize_to_shape(img: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    if img.shape == (out_h, out_w):
        return img.astype(np.float32)
    if cv2 is None:
        raise RuntimeError(f"Output shape {img.shape} does not match target {(out_h, out_w)} and OpenCV is unavailable.")
    return cv2.resize(img.astype(np.float32), (out_w, out_h), interpolation=cv2.INTER_LINEAR).astype(np.float32)


def run_benchmark(
    samples: list[Sample],
    methods: list[InpaintMethod],
    output_dir: Path,
    quit_controller: QuitController | None = None,
) -> list[MethodAggregate]:
    example_out = output_dir / "example_outputs"
    example_out.mkdir(parents=True, exist_ok=True)
    for sample in samples:
        save_sample_visuals(sample, example_out)

    aggregates: list[MethodAggregate] = []
    for method in methods:
        if quit_controller is not None and quit_controller.should_stop():
            print(f"[STOP] Ending benchmark before method '{method.name}'.")
            break
        if method.skip_reason is not None:
            print(f"[SKIP] {method.name}: {method.skip_reason}")
            aggregates.append(
                MethodAggregate(
                    method=method.name,
                    masked_mae=math.nan,
                    masked_rmse=math.nan,
                    masked_psnr=math.nan,
                    masked_ssim=math.nan,
                    mean_runtime_s=math.nan,
                )
            )
            continue

        maes: list[float] = []
        rmses: list[float] = []
        psnrs: list[float] = []
        ssims: list[float] = []
        runtimes: list[float] = []

        print(f"[RUN ] {method.name}")
        for idx, sample in enumerate(samples, start=1):
            if quit_controller is not None and quit_controller.should_stop():
                print(f"  [STOP] User requested stop before sample {sample.sample_id}.")
                break
            t0 = time.perf_counter()
            try:
                pred_norm = method.predict(sample)
                pred_norm = np.clip(pred_norm.astype(np.float32), 0.0, 1.0)
                pred_norm = resize_to_shape(pred_norm, sample.target.shape[0], sample.target.shape[1])
                pred = denormalize_image(pred_norm, sample.target_min, sample.target_max)
            except Exception as exc:
                print(f"  [WARN] {method.name} failed on {sample.sample_id}: {exc}")
                continue
            dt = time.perf_counter() - t0
            m = masked_metrics(pred=pred, target=sample.target, mask=sample.mask)
            maes.append(m["masked_mae"])
            rmses.append(m["masked_rmse"])
            psnrs.append(m["masked_psnr"])
            ssims.append(m["masked_ssim"])
            runtimes.append(dt)
            save_method_visuals(sample, method.name, pred, example_out)
            print(f"  [{idx:03d}/{len(samples):03d}] {sample.sample_id} | MAE={m['masked_mae']:.6g} | {dt:.4f}s")

        aggregates.append(
            MethodAggregate(
                method=method.name,
                masked_mae=aggregate(maes),
                masked_rmse=aggregate(rmses),
                masked_psnr=aggregate(psnrs),
                masked_ssim=aggregate(ssims),
                mean_runtime_s=aggregate(runtimes),
            )
        )
    return aggregates


def write_csv(path: Path, rows: list[MethodAggregate]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "method",
                "masked_mae",
                "masked_rmse",
                "masked_psnr",
                "masked_ssim",
                "mean_runtime_s",
            ],
        )
        writer.writeheader()
        for r in rows:
            writer.writerow(
                {
                    "method": r.method,
                    "masked_mae": r.masked_mae,
                    "masked_rmse": r.masked_rmse,
                    "masked_psnr": r.masked_psnr,
                    "masked_ssim": r.masked_ssim,
                    "mean_runtime_s": r.mean_runtime_s,
                }
            )


def leaderboard(rows: list[MethodAggregate]) -> list[MethodAggregate]:
    return sorted(
        rows,
        key=lambda r: (not np.isfinite(r.masked_mae), r.masked_mae if np.isfinite(r.masked_mae) else float("inf")),
    )


def fmt_num(v: float) -> str:
    if not np.isfinite(v):
        return "nan"
    return f"{v:.6g}"


def print_leaderboard(rows: list[MethodAggregate]) -> None:
    ranked = leaderboard(rows)
    print("\nRank | Method | Masked MAE | Masked RMSE | Masked PSNR | Runtime")
    for i, r in enumerate(ranked, start=1):
        print(
            f"{i:>4d} | {r.method} | {fmt_num(r.masked_mae)} | {fmt_num(r.masked_rmse)} | "
            f"{fmt_num(r.masked_psnr)} | {fmt_num(r.mean_runtime_s)}"
        )


def _default_path_or_none(path_str: str) -> Path | None:
    p = Path(path_str)
    return p if p.exists() else None


def build_methods(args: argparse.Namespace) -> list[InpaintMethod]:
    if args.lama_repo is None:
        args.lama_repo = _default_path_or_none("third_party/lama")
    if args.lama_model_path is None:
        args.lama_model_path = _default_path_or_none("third_party/models/big-lama")
    if args.zits_repo is None:
        args.zits_repo = _default_path_or_none("third_party/ZITS-PlusPlus")
    if args.zits_ckpt_path is None:
        args.zits_ckpt_path = _default_path_or_none("third_party/ZITS-PlusPlus/ckpts/model_512/models/last.ckpt")
    if args.zits_wf_ckpt_path is None:
        args.zits_wf_ckpt_path = _default_path_or_none("third_party/ZITS-PlusPlus/ckpts/best_lsm_hawp.pth")
    if args.partialconv_repo is None:
        args.partialconv_repo = _default_path_or_none("third_party/partialconv_naoto")
    if args.partialconv_checkpoint_path is None:
        args.partialconv_checkpoint_path = _default_path_or_none("third_party/models/partialconv/iter_1000000.pth")

    for attr in (
        "lama_repo",
        "lama_model_path",
        "zits_repo",
        "zits_ckpt_path",
        "zits_wf_ckpt_path",
        "partialconv_repo",
        "partialconv_checkpoint_path",
    ):
        p = getattr(args, attr, None)
        if isinstance(p, Path):
            setattr(args, attr, p.resolve())

    lama_template = args.lama_command_template
    if (not lama_template.strip()) and args.lama_repo is not None and args.lama_model_path is not None:
        lama_template = (
            'set "PYTHONPATH={repo}" && '
            '"{python}" "bin/predict.py" '
            'model.path="{lama_model_path}" '
            'indir="{input_dir}" '
            'outdir="{output_dir}" '
            'device=cpu'
        )

    zits_template = args.zits_command_template
    if (
        (not zits_template.strip())
        and args.zits_repo is not None
        and args.zits_ckpt_path is not None
        and args.zits_wf_ckpt_path is not None
    ):
        zits_template = (
            '"{python}" "test.py" '
            '--config "configs/config_zitspp_finetune.yml" '
            '--exp_name model_512 '
            '--ckpt_resume "{zits_ckpt_path}" '
            '--save_path "{output_dir}" '
            '--img_dir "{input_dir}" '
            '--mask_dir "{mask_dir}" '
            '--wf_ckpt "{zits_wf_ckpt_path}" '
            '--use_ema --test_size 512 --obj_removal --save_image_only'
        )

    key_to_method_factory: dict[str, Callable[[], InpaintMethod]] = {
        "telea": lambda: TeleaMethod(radius=args.inpaint_radius),
        "ns": lambda: NavierStokesMethod(radius=args.inpaint_radius),
        "biharmonic": BiharmonicMethod,
        "lama": lambda: ExternalCommandMethod(
            name="LaMa pretrained",
            command_template=lama_template,
            repo_dir=args.lama_repo,
            extra_context={"lama_model_path": str(args.lama_model_path) if args.lama_model_path is not None else ""},
        ),
        "zitspp": lambda: ExternalCommandMethod(
            name="ZITS++ pretrained",
            command_template=zits_template,
            repo_dir=args.zits_repo,
            extra_context={
                "zits_ckpt_path": str(args.zits_ckpt_path) if args.zits_ckpt_path is not None else "",
                "zits_wf_ckpt_path": str(args.zits_wf_ckpt_path) if args.zits_wf_ckpt_path is not None else "",
            },
        ),
        "partialconv": lambda: PartialConvNaotoMethod(
            repo_dir=args.partialconv_repo if args.partialconv_repo is not None else Path("__missing_partialconv_repo__"),
            checkpoint_path=(
                args.partialconv_checkpoint_path
                if args.partialconv_checkpoint_path is not None
                else Path("__missing_partialconv_checkpoint__.pth")
            ),
            force_cpu=bool(args.force_cpu_models),
        ),
    }

    requested = [m.strip().lower() for m in args.methods.split(",") if m.strip()]
    methods: list[InpaintMethod] = []
    unknown = [m for m in requested if m not in key_to_method_factory]
    if unknown:
        raise ValueError(f"Unknown methods requested: {unknown}")
    for key in requested:
        methods.append(key_to_method_factory[key]())
    return methods


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark pretrained and classical inpainting/background-completion methods on grayscale microscopy validation data "
            "using masked-only metrics."
        )
    )
    parser.add_argument(
        "--target-dir",
        type=Path,
        default=Path("misc analysis and plotting scripts/gpt prompted exploration/Background ml recon/data/val_backgrounds"),
        help="Validation target directory containing original unmasked microscopy images.",
    )
    parser.add_argument(
        "--mask-dir",
        type=Path,
        default=Path("misc analysis and plotting scripts/gpt prompted exploration/Background ml recon/data/val_masks"),
        help="Validation mask directory where 1 indicates pixels to remove/fill.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results"), help="Output root directory.")
    parser.add_argument("--max-images", type=int, default=None, help="Optional cap on number of validation samples.")
    parser.add_argument(
        "--no-quit-listener",
        action="store_true",
        help="Disable 'q + Enter' graceful-stop listener.",
    )
    parser.add_argument(
        "--methods",
        type=str,
        default="telea,ns,biharmonic,lama,zitspp,partialconv",
        help="Comma-separated method keys: telea,ns,biharmonic,lama,zitspp,partialconv",
    )
    parser.add_argument("--inpaint-radius", type=float, default=3.0, help="Inpaint radius for OpenCV classical methods.")

    parser.add_argument("--lama-repo", type=Path, default=None, help="Optional LaMa repository directory (command cwd).")
    parser.add_argument(
        "--lama-model-path",
        type=Path,
        default=None,
        help="LaMa pretrained model directory for default command mode (typically repo-local big-lama).",
    )
    parser.add_argument(
        "--lama-command-template",
        type=str,
        default="",
        help=(
            "Shell command template for LaMa inference. Placeholders: {python},{image},{mask},{output},{output_dir},{input_dir},{stem},{repo}."
        ),
    )
    parser.add_argument("--zits-repo", type=Path, default=None, help="Optional ZITS++ repository directory (command cwd).")
    parser.add_argument(
        "--zits-command-template",
        type=str,
        default="",
        help=(
            "Shell command template for ZITS++ inference. Placeholders: {python},{image},{mask},{output},{output_dir},{input_dir},{mask_dir},{stem},{repo}."
        ),
    )
    parser.add_argument(
        "--zits-ckpt-path",
        type=Path,
        default=None,
        help="Path to ZITS++ generator checkpoint (default auto-detected).",
    )
    parser.add_argument(
        "--zits-wf-ckpt-path",
        type=Path,
        default=None,
        help="Path to ZITS++ wireframe checkpoint (default auto-detected).",
    )
    parser.add_argument(
        "--partialconv-repo",
        type=Path,
        default=None,
        help="PartialConv (naoto0804 implementation) repository directory.",
    )
    parser.add_argument(
        "--partialconv-checkpoint-path",
        type=Path,
        default=None,
        help="Path to PartialConv checkpoint file (default auto-detected).",
    )
    parser.add_argument(
        "--force-cpu-models",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Force ML pretrained methods to run on CPU when their implementation supports it.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    quit_controller = QuitController(enabled=not args.no_quit_listener)
    methods = build_methods(args)
    samples = load_samples(args.target_dir, args.mask_dir, args.max_images)
    print(f"Loaded {len(samples)} validation samples.")
    rows = run_benchmark(samples=samples, methods=methods, output_dir=args.output_dir, quit_controller=quit_controller)
    csv_path = args.output_dir / "benchmark_scores.csv"
    write_csv(csv_path, rows)
    print(f"\nSaved scores: {csv_path}")
    print(f"Saved example outputs: {args.output_dir / 'example_outputs'}")
    print_leaderboard(rows)


if __name__ == "__main__":
    main()
