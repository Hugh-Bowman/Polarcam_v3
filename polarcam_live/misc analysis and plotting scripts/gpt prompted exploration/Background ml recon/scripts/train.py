from __future__ import annotations

import argparse
import csv
import math
import threading
import time
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from background_ml_recon.dataset import PrecomputedMaskedDataset, build_triplets
from background_ml_recon.losses import masked_reconstruction_loss
from background_ml_recon.model import build_model
from background_ml_recon.utils import load_yaml, save_checkpoint, save_json, set_seed


def start_stop_listener(stop_event: threading.Event) -> list[threading.Thread]:
    threads: list[threading.Thread] = []
    try:
        import msvcrt  # type: ignore

        def _poll_keys() -> None:
            while not stop_event.is_set():
                try:
                    if msvcrt.kbhit():
                        ch = msvcrt.getwch().lower()
                        if ch == "q":
                            print("\n[stop] 'q' pressed. Stopping after current batch.")
                            stop_event.set()
                            return
                    time.sleep(0.05)
                except Exception:
                    return

        t = threading.Thread(target=_poll_keys, daemon=True)
        t.start()
        threads.append(t)
    except Exception:
        pass

    def _stdin_lines() -> None:
        while not stop_event.is_set():
            try:
                line = sys.stdin.readline()
            except Exception:
                return
            if not line:
                return
            if line.strip().lower() == "q":
                print("\n[stop] 'q' received on stdin. Stopping after current batch.")
                stop_event.set()
                return

    t2 = threading.Thread(target=_stdin_lines, daemon=True)
    t2.start()
    threads.append(t2)
    return threads


def _parse_optional_patch_size(value) -> int | None:
    if value is None:
        return None
    if isinstance(value, str):
        v = value.strip().lower()
        if v in {"none", "null", ""}:
            return None
    iv = int(value)
    return iv if iv > 0 else None


def _parse_tokens(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        t = value.strip().lower()
        return [t] if t else []
    if isinstance(value, (list, tuple)):
        out = []
        for v in value:
            s = str(v).strip().lower()
            if s:
                out.append(s)
        return out
    s = str(value).strip().lower()
    return [s] if s else []


def filter_triplets_by_tokens(
    triplets: list[tuple[Path, Path, Path]],
    tokens: list[str],
) -> list[tuple[Path, Path, Path]]:
    if not tokens:
        return triplets
    out = []
    for t in triplets:
        stem = t[0].stem.lower()
        if any(tok in stem for tok in tokens):
            out.append(t)
    return out


def compose_full_prediction(inp: torch.Tensor, mask: torch.Tensor, pred_delta: torch.Tensor) -> torch.Tensor:
    # Enforce exact identity on known pixels; model only predicts masked residual.
    hole_mask = canonical_hole_mask(mask, inp[:, 0:1])
    return inp[:, 0:1] + hole_mask * pred_delta


def canonical_hole_mask(mask: torch.Tensor, masked_input: torch.Tensor | None = None) -> torch.Tensor:
    mask_bin = (mask > 0.5).to(mask.dtype)
    if masked_input is not None:
        ones_sel = mask_bin > 0.5
        zeros_sel = ~ones_sel
        if bool(ones_sel.any()) and bool(zeros_sel.any()):
            ones_abs = torch.abs(masked_input[ones_sel]).mean()
            zeros_abs = torch.abs(masked_input[zeros_sel]).mean()
            if torch.isfinite(ones_abs) and torch.isfinite(zeros_abs):
                if zeros_abs + 1e-8 < ones_abs:
                    return 1.0 - mask_bin
                if ones_abs + 1e-8 < zeros_abs:
                    return mask_bin
    if float(mask_bin.mean().item()) > 0.5:
        return 1.0 - mask_bin
    return mask_bin


def scale_pred_delta(model: torch.nn.Module, pred_delta_raw: torch.Tensor) -> torch.Tensor:
    if hasattr(model, "output_scale") and isinstance(model.output_scale, torch.Tensor):
        return model.output_scale * pred_delta_raw
    return pred_delta_raw


def save_loss_plot(path: Path, history: dict[str, list[float]]) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[warn] matplotlib unavailable; skipping loss plot ({exc})")
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    tr = history.get("train_loss", [])
    va = history.get("val_loss", [])
    n = max(len(tr), len(va))
    if n == 0:
        return
    epochs = list(range(1, n + 1))
    plt.figure(figsize=(7.5, 4.5))
    if tr:
        plt.plot(epochs[: len(tr)], tr, label="train_loss", linewidth=1.8)
    if va:
        plt.plot(epochs[: len(va)], va, label="val_loss", linewidth=1.8)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss vs Epoch")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()


def run_epoch(
    model,
    loader,
    device,
    optimizer,
    masked_weight,
    visible_weight,
    loss_space,
    loss_type,
    lama_weight_known,
    lama_weight_missing,
    huber_beta,
    train_mode: bool,
    epoch: int,
    epochs_total: int,
    stop_event: threading.Event,
    bad_batch_loss_threshold: float | None = None,
    log_scale_stats_once: dict | None = None,
):
    if train_mode:
        model.train()
    else:
        model.eval()

    total = 0.0
    n = 0
    masked_count_total = 0.0
    abs_model_sum = 0.0
    abs_zero_sum = 0.0
    target_sum = 0.0
    target_sq_sum = 0.0
    pred_sum = 0.0
    pred_sq_sum = 0.0
    mask_sum = 0.0
    pix_count = 0.0

    mode_label = "train" if train_mode else "val"
    bar = tqdm(
        loader,
        total=len(loader),
        leave=True,
        dynamic_ncols=True,
        mininterval=0.4,
        desc=f"{mode_label} epoch {epoch:03d}/{epochs_total:03d}",
    )
    for batch_idx, batch in enumerate(bar):
        if stop_event.is_set():
            break
        inp = batch["input"].to(device)
        target = batch["target"].to(device)
        mask = batch["mask"].to(device)

        if train_mode:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(train_mode):
            pred_delta_raw = model(inp)
            pred_delta = scale_pred_delta(model, pred_delta_raw)
            pred = compose_full_prediction(inp, mask, pred_delta)
            loss, logs = masked_reconstruction_loss(
                pred,
                target,
                mask,
                masked_weight=masked_weight,
                visible_weight=visible_weight,
                loss_space=loss_space,
                loss_type=loss_type,
                lama_weight_known=lama_weight_known,
                lama_weight_missing=lama_weight_missing,
                huber_beta=huber_beta,
            )
            if log_scale_stats_once is not None and not bool(log_scale_stats_once.get("done", False)):
                with torch.no_grad():
                    sel = mask > 0.5
                    raw_vals = pred_delta_raw[sel] if sel.any() else pred_delta_raw.reshape(-1)
                    scaled_vals = pred_delta[sel] if sel.any() else pred_delta.reshape(-1)
                    pred_vals = pred[sel] if sel.any() else pred.reshape(-1)
                    print(
                        "[scale] raw pred_delta min/max/mean/std="
                        f"{float(raw_vals.min().item()):.6f}/{float(raw_vals.max().item()):.6f}/"
                        f"{float(raw_vals.mean().item()):.6f}/{float(raw_vals.std(unbiased=False).item()):.6f}"
                    )
                    print(
                        "[scale] scaled pred_delta min/max/mean/std="
                        f"{float(scaled_vals.min().item()):.6f}/{float(scaled_vals.max().item()):.6f}/"
                        f"{float(scaled_vals.mean().item()):.6f}/{float(scaled_vals.std(unbiased=False).item()):.6f}"
                    )
                    print(
                        "[scale] final pred masked min/max/mean/std="
                        f"{float(pred_vals.min().item()):.6f}/{float(pred_vals.max().item()):.6f}/"
                        f"{float(pred_vals.mean().item()):.6f}/{float(pred_vals.std(unbiased=False).item()):.6f}"
                    )
                log_scale_stats_once["done"] = True

            if train_mode:
                loss.backward()
                optimizer.step()

        with torch.no_grad():
            sel = mask > 0.5
            cnt = int(sel.sum().item())
            if cnt > 0:
                pred_masked = pred[sel]
                target_masked = target[sel]
                zero_masked = inp[:, 0:1][sel]
                abs_model_sum += float(torch.abs(pred_masked - target_masked).sum().item())
                abs_zero_sum += float(torch.abs(zero_masked - target_masked).sum().item())
                target_sum += float(target_masked.sum().item())
                target_sq_sum += float((target_masked * target_masked).sum().item())
                pred_sum += float(pred_masked.sum().item())
                pred_sq_sum += float((pred_masked * pred_masked).sum().item())
                masked_count_total += float(cnt)
            mask_sum += float(mask.sum().item())
            pix_count += float(mask.numel())

            thr = bad_batch_loss_threshold
            if thr is not None and np.isfinite(float(thr)) and float(logs["loss_total"]) > float(thr):
                masked_target_vals = target[sel] if cnt > 0 else torch.empty(0, device=target.device)
                masked_pred_vals = pred[sel] if cnt > 0 else torch.empty(0, device=pred.device)
                file_list = batch.get("target_path", [])
                if isinstance(file_list, str):
                    file_list = [file_list]
                print(
                    f"[bad-batch] epoch={epoch:03d} mode={'train' if train_mode else 'val'} batch={batch_idx} "
                    f"loss={float(logs['loss_total']):.6f}"
                )
                print(
                    f"[bad-batch] input min/max/mean="
                    f"{float(inp[:,0:1].min().item()):.6f}/{float(inp[:,0:1].max().item()):.6f}/{float(inp[:,0:1].mean().item()):.6f}"
                )
                print(
                    f"[bad-batch] target min/max/mean="
                    f"{float(target.min().item()):.6f}/{float(target.max().item()):.6f}/{float(target.mean().item()):.6f}"
                )
                if cnt > 0:
                    print(
                        f"[bad-batch] masked target min/max/mean/std="
                        f"{float(masked_target_vals.min().item()):.6f}/{float(masked_target_vals.max().item()):.6f}/"
                        f"{float(masked_target_vals.mean().item()):.6f}/{float(masked_target_vals.std(unbiased=False).item()):.6f}"
                    )
                    print(
                        f"[bad-batch] masked pred min/max/mean/std="
                        f"{float(masked_pred_vals.min().item()):.6f}/{float(masked_pred_vals.max().item()):.6f}/"
                        f"{float(masked_pred_vals.mean().item()):.6f}/{float(masked_pred_vals.std(unbiased=False).item()):.6f}"
                    )
                print(f"[bad-batch] mask_fraction={float(mask.mean().item()):.6f}")
                if file_list:
                    preview_files = ", ".join([str(x) for x in list(file_list)[:4]])
                    print(f"[bad-batch] filenames={preview_files}")

        bs = inp.shape[0]
        total += logs["loss_total"] * bs
        n += bs
        bar.set_postfix(loss=f"{(total / max(n, 1)):.5f}")

    bar.close()
    mean_loss = total / max(n, 1)
    masked_denom = max(masked_count_total, 1.0)
    masked_target_mean = target_sum / masked_denom
    masked_pred_mean = pred_sum / masked_denom
    masked_target_var = max(target_sq_sum / masked_denom - masked_target_mean * masked_target_mean, 0.0)
    masked_pred_var = max(pred_sq_sum / masked_denom - masked_pred_mean * masked_pred_mean, 0.0)
    return {
        "loss": float(mean_loss),
        "masked_loss_model": float(abs_model_sum / masked_denom),
        "masked_loss_zero_fill": float(abs_zero_sum / masked_denom),
        "masked_target_mean": float(masked_target_mean),
        "masked_target_std": float(math.sqrt(masked_target_var)),
        "masked_prediction_mean": float(masked_pred_mean),
        "masked_prediction_std": float(math.sqrt(masked_pred_var)),
        "mask_fraction": float(mask_sum / max(pix_count, 1.0)),
    }


def evaluate_zero_fill_baseline(
    loader,
    device,
    masked_weight: float,
    visible_weight: float,
    loss_space: str,
    loss_type: str,
    lama_weight_known: float,
    lama_weight_missing: float,
    huber_beta: float,
) -> float:
    total = 0.0
    n = 0
    for batch in loader:
        inp = batch["input"].to(device)
        target = batch["target"].to(device)
        mask = batch["mask"].to(device)
        pred = inp[:, 0:1]  # Zero-fill baseline because masked pixels in input are zero.
        _, logs = masked_reconstruction_loss(
            pred,
            target,
            mask,
            masked_weight=masked_weight,
            visible_weight=visible_weight,
            loss_space=loss_space,
            loss_type=loss_type,
            lama_weight_known=lama_weight_known,
            lama_weight_missing=lama_weight_missing,
            huber_beta=huber_beta,
        )
        bs = inp.shape[0]
        total += logs["loss_total"] * bs
        n += bs
    return total / max(n, 1)


def evaluate_lama_backbone_baseline(
    model: torch.nn.Module,
    loader,
    device,
    masked_weight: float,
    visible_weight: float,
    loss_space: str,
    loss_type: str,
    lama_weight_known: float,
    lama_weight_missing: float,
    huber_beta: float,
) -> float | None:
    if not hasattr(model, "predict_lama_full"):
        return None
    total = 0.0
    n = 0
    model.eval()
    with torch.no_grad():
        for batch in loader:
            inp = batch["input"].to(device)
            target = batch["target"].to(device)
            mask = batch["mask"].to(device)
            masked_input = inp[:, 0:1]
            pred_full = model.predict_lama_full(masked_input, mask)
            pred_delta = pred_full - masked_input
            pred = compose_full_prediction(inp, mask, pred_delta)
            _, logs = masked_reconstruction_loss(
                pred,
                target,
                mask,
                masked_weight=masked_weight,
                visible_weight=visible_weight,
                loss_space=loss_space,
                loss_type=loss_type,
                lama_weight_known=lama_weight_known,
                lama_weight_missing=lama_weight_missing,
                huber_beta=huber_beta,
            )
            bs = inp.shape[0]
            total += logs["loss_total"] * bs
            n += bs
    return total / max(n, 1)


def save_fixed_val_preview(
    model: torch.nn.Module,
    val_ds: PrecomputedMaskedDataset,
    device: torch.device,
    out_path: Path,
    sample_index: int = 0,
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    if len(val_ds) == 0:
        return
    sample_index = int(max(0, min(sample_index, len(val_ds) - 1)))
    batch = val_ds[sample_index]
    inp = batch["input"][None, ...].to(device)
    target = batch["target"][None, ...].to(device)
    mask = batch["mask"][None, ...].to(device)
    target_raw = batch.get("target_raw", batch["target"])[None, ...].to(device)
    target_smooth = batch.get("target_smooth", batch["target"])[None, ...].to(device)
    with torch.no_grad():
        pred_delta = scale_pred_delta(model, model(inp))
        pred = compose_full_prediction(inp, mask, pred_delta)

    raw_np = target_raw[0, 0].detach().cpu().numpy().astype(np.float32)
    smooth_np = target_smooth[0, 0].detach().cpu().numpy().astype(np.float32)
    masked_np = inp[0, 0].detach().cpu().numpy().astype(np.float32)
    mask_np = mask[0, 0].detach().cpu().numpy().astype(np.float32)
    pred_np = pred[0, 0].detach().cpu().numpy().astype(np.float32)
    err_np = np.abs(pred_np - smooth_np) * mask_np

    stack = np.concatenate([raw_np.ravel(), smooth_np.ravel(), masked_np.ravel(), pred_np.ravel()])
    lo = float(np.percentile(stack, 1.0))
    hi = float(np.percentile(stack, 99.0))
    if hi <= lo:
        lo, hi = float(np.min(stack)), float(np.max(stack))
    if hi <= lo:
        hi = lo + 1e-6
    e_hi = float(np.percentile(err_np, 99.0))
    if e_hi <= 0:
        e_hi = float(np.max(err_np))
    if e_hi <= 0:
        e_hi = 1e-6

    fig, axes = plt.subplots(1, 6, figsize=(20, 4))
    axes[0].imshow(raw_np, cmap="gray", vmin=lo, vmax=hi)
    axes[0].set_title("Raw Target")
    axes[1].imshow(smooth_np, cmap="gray", vmin=lo, vmax=hi)
    axes[1].set_title("Smoothed Target")
    axes[2].imshow(masked_np, cmap="gray", vmin=lo, vmax=hi)
    axes[2].set_title("Smoothed Masked Input")
    axes[3].imshow(mask_np, cmap="gray", vmin=0, vmax=1)
    axes[3].set_title("Mask")
    axes[4].imshow(pred_np, cmap="gray", vmin=lo, vmax=hi)
    axes[4].set_title("Prediction")
    axes[5].imshow(err_np, cmap="magma", vmin=0, vmax=e_hi)
    axes[5].set_title("Masked Error")
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_multi_split_preview(
    model: torch.nn.Module,
    val_ds: PrecomputedMaskedDataset,
    test_ds: PrecomputedMaskedDataset,
    device: torch.device,
    out_path: Path,
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return

    if len(val_ds) == 0 or len(test_ds) == 0:
        return

    specs = [
        ("Val", val_ds, 0),
        ("Val", val_ds, min(1, len(val_ds) - 1)),
        ("Test", test_ds, 0),
        ("Test", test_ds, min(1, len(test_ds) - 1)),
    ]

    fig, axes = plt.subplots(len(specs), 6, figsize=(20, 2.8 * len(specs)))
    if len(specs) == 1:
        axes = np.expand_dims(axes, 0)

    for r, (split_name, ds, idx) in enumerate(specs):
        batch = ds[int(idx)]
        inp = batch["input"][None, ...].to(device)
        target = batch["target"][None, ...].to(device)
        mask = batch["mask"][None, ...].to(device)
        target_raw = batch.get("target_raw", batch["target"])[None, ...].to(device)
        target_smooth = batch.get("target_smooth", batch["target"])[None, ...].to(device)
        with torch.no_grad():
            pred_delta = scale_pred_delta(model, model(inp))
            pred = compose_full_prediction(inp, mask, pred_delta)

        raw_np = target_raw[0, 0].detach().cpu().numpy().astype(np.float32)
        smooth_np = target_smooth[0, 0].detach().cpu().numpy().astype(np.float32)
        masked_np = inp[0, 0].detach().cpu().numpy().astype(np.float32)
        mask_np = mask[0, 0].detach().cpu().numpy().astype(np.float32)
        pred_np = pred[0, 0].detach().cpu().numpy().astype(np.float32)
        target_np = target[0, 0].detach().cpu().numpy().astype(np.float32)
        err_np = np.abs(pred_np - target_np) * mask_np

        stack = np.concatenate([raw_np.ravel(), smooth_np.ravel(), masked_np.ravel(), pred_np.ravel(), target_np.ravel()])
        lo = float(np.percentile(stack, 1.0))
        hi = float(np.percentile(stack, 99.0))
        if hi <= lo:
            lo = float(np.min(stack))
            hi = float(np.max(stack))
        if hi <= lo:
            hi = lo + 1e-6
        e_hi = float(np.percentile(err_np, 99.0))
        if e_hi <= 0:
            e_hi = float(np.max(err_np))
        if e_hi <= 0:
            e_hi = 1e-6

        axes[r, 0].imshow(raw_np, cmap="gray", vmin=lo, vmax=hi)
        axes[r, 0].set_title("Raw")
        axes[r, 1].imshow(smooth_np, cmap="gray", vmin=lo, vmax=hi)
        axes[r, 1].set_title("Smooth")
        axes[r, 2].imshow(masked_np, cmap="gray", vmin=lo, vmax=hi)
        axes[r, 2].set_title("Smooth Masked")
        axes[r, 3].imshow(mask_np, cmap="gray", vmin=0, vmax=1)
        axes[r, 3].set_title("Mask")
        axes[r, 4].imshow(pred_np, cmap="gray", vmin=lo, vmax=hi)
        axes[r, 4].set_title("Prediction")
        axes[r, 5].imshow(err_np, cmap="magma", vmin=0, vmax=e_hi)
        axes[r, 5].set_title("Masked Error")

        axes[r, 0].set_ylabel(f"{split_name} #{idx+1}", fontsize=9)
        for c in range(6):
            axes[r, c].axis("off")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def _masked_raw_mae(pred: np.ndarray, target: np.ndarray, mask: np.ndarray) -> float:
    sel = mask > 0.5
    if not np.any(sel):
        return float("nan")
    return float(np.abs(pred[sel] - target[sel]).mean())


def _masked_log_mae(pred: np.ndarray, target: np.ndarray, mask: np.ndarray) -> float:
    sel = mask > 0.5
    if not np.any(sel):
        return float("nan")
    pred_l = np.log1p(np.clip(pred, 0.0, None))
    target_l = np.log1p(np.clip(target, 0.0, None))
    return float(np.abs(pred_l[sel] - target_l[sel]).mean())


def _safe_stats(x: np.ndarray) -> dict[str, float]:
    if x.size == 0:
        return {"count": 0.0}
    out: dict[str, float] = {
        "count": float(x.size),
        "mean": float(np.mean(x)),
        "median": float(np.median(x)),
        "std": float(np.std(x)),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
    }
    for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
        out[f"p{p}"] = float(np.percentile(x, p))
    return out


def _local_gaussian_estimate(target: np.ndarray, mask: np.ndarray, sigma: float, eps: float = 1e-6) -> np.ndarray:
    if cv2 is None:
        raise RuntimeError("OpenCV is required for local_gaussian_fill diagnostic baseline.")
    known = 1.0 - mask
    numerator = cv2.GaussianBlur((target * known).astype(np.float32), (0, 0), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REFLECT)
    denominator = cv2.GaussianBlur(known.astype(np.float32), (0, 0), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REFLECT)
    return (numerator / (denominator + eps)).astype(np.float32)


def _smooth_target_with_sigma(target_raw: np.ndarray, mask: np.ndarray, sigma: float, mask_aware: bool = True) -> np.ndarray:
    if cv2 is None:
        raise RuntimeError("OpenCV is required for smoothing diagnostics.")
    if not mask_aware:
        return cv2.GaussianBlur(
            target_raw.astype(np.float32),
            (0, 0),
            sigmaX=float(sigma),
            sigmaY=float(sigma),
            borderType=cv2.BORDER_REFLECT,
        ).astype(np.float32)
    known = 1.0 - mask
    numerator = cv2.GaussianBlur(
        (target_raw * known).astype(np.float32),
        (0, 0),
        sigmaX=float(sigma),
        sigmaY=float(sigma),
        borderType=cv2.BORDER_REFLECT,
    )
    denominator = cv2.GaussianBlur(
        known.astype(np.float32),
        (0, 0),
        sigmaX=float(sigma),
        sigmaY=float(sigma),
        borderType=cv2.BORDER_REFLECT,
    )
    return (numerator / (denominator + 1e-8)).astype(np.float32)


def _arr_stats(x: np.ndarray) -> tuple[float, float, float, float]:
    if x.size == 0:
        return (float("nan"), float("nan"), float("nan"), float("nan"))
    return (float(np.min(x)), float(np.max(x)), float(np.mean(x)), float(np.std(x)))


def debug_mask_target_sample(
    ds: PrecomputedMaskedDataset,
    out_dir: Path,
    sample_index: int = 0,
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[debug] matplotlib unavailable; skipping debug images ({exc})")
        return

    if len(ds) == 0:
        print("[debug] dataset empty; skipping mask/target debug sample")
        return

    idx = int(max(0, min(sample_index, len(ds) - 1)))
    batch = ds[idx]
    inp = batch["input"][0].detach().cpu().numpy().astype(np.float32)
    mask = batch["mask"][0].detach().cpu().numpy().astype(np.float32)
    target = batch["target"][0].detach().cpu().numpy().astype(np.float32)
    target_smooth = batch.get("target_smooth", batch["target"])[0].detach().cpu().numpy().astype(np.float32)

    masked_sel = mask > 0.5
    known_sel = mask <= 0.5
    mask_fraction = float(mask.mean())

    t_mn, t_mx, t_mean, t_std = _arr_stats(target[masked_sel])
    i_mn, i_mx, i_mean, i_std = _arr_stats(inp[masked_sel])
    tk_mn, tk_mx, tk_mean, tk_std = _arr_stats(target[known_sel])
    ik_mn, ik_mx, ik_mean, ik_std = _arr_stats(inp[known_sel])
    masked_abs_mean = float(np.abs(target[masked_sel]).mean()) if np.any(masked_sel) else float("nan")

    print(f"[debug] sample_index={idx} mask_fraction={mask_fraction:.6f}")
    print(f"[debug] target masked   min/max/mean/std={t_mn:.6f}/{t_mx:.6f}/{t_mean:.6f}/{t_std:.6f}")
    print(f"[debug] input masked    min/max/mean/std={i_mn:.6f}/{i_mx:.6f}/{i_mean:.6f}/{i_std:.6f}")
    print(f"[debug] target known    min/max/mean/std={tk_mn:.6f}/{tk_mx:.6f}/{tk_mean:.6f}/{tk_std:.6f}")
    print(f"[debug] input known     min/max/mean/std={ik_mn:.6f}/{ik_mx:.6f}/{ik_mean:.6f}/{ik_std:.6f}")
    print(f"[debug] abs(target[mask>0.5]).mean()={masked_abs_mean:.6f}")

    out_dir.mkdir(parents=True, exist_ok=True)
    sample_dir = out_dir / "mask_target_debug"
    sample_dir.mkdir(parents=True, exist_ok=True)

    stack = np.concatenate([target_smooth.ravel(), inp.ravel(), target.ravel()])
    lo = float(np.percentile(stack, 1.0))
    hi = float(np.percentile(stack, 99.0))
    if hi <= lo:
        lo, hi = float(np.min(stack)), float(np.max(stack))
    if hi <= lo:
        hi = lo + 1e-6

    def _save(path: Path, arr: np.ndarray, is_mask: bool = False) -> None:
        plt.figure(figsize=(5, 4))
        if is_mask:
            plt.imshow(arr, cmap="gray", vmin=0, vmax=1)
        else:
            plt.imshow(arr, cmap="gray", vmin=lo, vmax=hi)
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(path, dpi=140)
        plt.close()

    _save(sample_dir / "mask.png", mask, is_mask=True)
    _save(sample_dir / "smoothed_target.png", target_smooth)
    _save(sample_dir / "masked_input.png", inp)
    _save(sample_dir / "target_times_mask.png", target * mask)
    _save(sample_dir / "input_times_mask.png", inp * mask)
    print(f"[debug] wrote mask/target debug images to: {sample_dir}")


def run_diagnostics(
    model: torch.nn.Module,
    val_preview_ds: PrecomputedMaskedDataset,
    device: torch.device,
    out_dir: Path,
    num_samples: int = 4,
    smooth_sigma_values: list[float] | None = None,
    default_smooth_sigma: float = 32.0,
    smooth_mask_aware: bool = True,
) -> list[dict[str, float | str]]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[warn] matplotlib unavailable; skipping diagnostics ({exc})")
        return []

    out_dir.mkdir(parents=True, exist_ok=True)
    vis_dir = out_dir / "examples"
    vis_dir.mkdir(parents=True, exist_ok=True)

    n_total = len(val_preview_ds)
    n_use = max(1, min(int(num_samples), n_total))
    indices = list(range(n_use))
    print(f"[diag] running diagnostics on fixed val subset size={n_use}/{n_total} (indices={indices})")

    percentile_grid = list(range(0, 101))
    percentile_candidates = [5, 10, 25, 50, 75, 90, 95]
    sigma_candidates = [8, 16, 32, 64, 128]
    smooth_sigmas = smooth_sigma_values if smooth_sigma_values else [16.0, 32.0, 64.0]

    rows: list[dict[str, float | str]] = []
    per_image_rows: list[dict[str, float | str]] = []

    pool_known_raw: list[np.ndarray] = []
    pool_masked_raw: list[np.ndarray] = []
    pool_known_smooth: list[np.ndarray] = []
    pool_masked_smooth: list[np.ndarray] = []

    sums: dict[str, dict[str, float]] = {}

    def _accumulate(method: str, raw_mae: float, log_mae: float, runtime_s: float, best_percentile: float = math.nan) -> None:
        if method not in sums:
            sums[method] = {"raw": 0.0, "log": 0.0, "rt": 0.0, "bp": 0.0, "n": 0.0, "bp_n": 0.0}
        sums[method]["raw"] += float(raw_mae)
        sums[method]["log"] += float(log_mae)
        sums[method]["rt"] += float(runtime_s)
        sums[method]["n"] += 1.0
        if np.isfinite(best_percentile):
            sums[method]["bp"] += float(best_percentile)
            sums[method]["bp_n"] += 1.0

    model.eval()
    with torch.no_grad():
        for i in indices:
            batch = val_preview_ds[i]
            inp_t = batch["input"][None, ...].to(device)
            target_t = batch["target"][None, ...].to(device)
            mask_t = batch["mask"][None, ...].to(device)
            target_raw_t = batch["target_raw"][None, ...].to(device) if "target_raw" in batch else target_t

            masked_input = inp_t[0, 0].detach().cpu().numpy().astype(np.float32)
            target_raw = target_raw_t[0, 0].detach().cpu().numpy().astype(np.float32)
            mask = mask_t[0, 0].detach().cpu().numpy().astype(np.float32)
            known_sel = mask < 0.5
            hole_sel = mask > 0.5
            pool_known_raw.append(target_raw[known_sel].copy())
            pool_masked_raw.append(target_raw[hole_sel].copy())

            # Model prediction
            t0 = time.perf_counter()
            pred_delta_t = scale_pred_delta(model, model(inp_t))
            pred_t = compose_full_prediction(inp_t, mask_t, pred_delta_t)
            rt_model = time.perf_counter() - t0
            model_pred = pred_t[0, 0].detach().cpu().numpy().astype(np.float32)
            frozen_pred = None
            rt_frozen = 0.0
            if hasattr(model, "predict_lama_full"):
                t1 = time.perf_counter()
                frozen_t = model.predict_lama_full(inp_t[:, 0:1], mask_t)
                rt_frozen = time.perf_counter() - t1
                frozen_pred = frozen_t[0, 0].detach().cpu().numpy().astype(np.float32)

            # Sigma sweep against smoothed targets
            for sigma_smooth in smooth_sigmas:
                target_smooth = _smooth_target_with_sigma(target_raw, mask, float(sigma_smooth), mask_aware=bool(smooth_mask_aware))
                known_vals = target_smooth[known_sel]
                hole_vals = target_smooth[hole_sel]
                if float(sigma_smooth) == float(default_smooth_sigma):
                    pool_known_smooth.append(known_vals.copy())
                    pool_masked_smooth.append(hole_vals.copy())

                zero_pred = masked_input.copy()
                zero_pred[hole_sel] = 0.0
                known_mean = float(np.mean(known_vals)) if known_vals.size > 0 else 0.0
                known_median = float(np.median(known_vals)) if known_vals.size > 0 else 0.0
                mean_pred = masked_input.copy()
                mean_pred[hole_sel] = known_mean
                median_pred = masked_input.copy()
                median_pred[hole_sel] = known_median

                best_cand_percentile = math.nan
                best_cand_pred = median_pred
                best_cand_mae = float("inf")
                for p in percentile_candidates:
                    c = float(np.percentile(known_vals, p)) if known_vals.size > 0 else 0.0
                    pred_p = masked_input.copy()
                    pred_p[hole_sel] = c
                    mae_p = _masked_raw_mae(pred_p, target_smooth, mask)
                    if mae_p < best_cand_mae:
                        best_cand_mae = mae_p
                        best_cand_percentile = float(p)
                        best_cand_pred = pred_p

                best_p = math.nan
                best_p_pred = median_pred
                best_p_mae = float("inf")
                for p in percentile_grid:
                    c = float(np.percentile(known_vals, p)) if known_vals.size > 0 else 0.0
                    pred_p = masked_input.copy()
                    pred_p[hole_sel] = c
                    mae_p = _masked_raw_mae(pred_p, target_smooth, mask)
                    if mae_p < best_p_mae:
                        best_p_mae = mae_p
                        best_p = float(p)
                        best_p_pred = pred_p

                best_sigma = math.nan
                best_local_pred = median_pred
                best_local_mae = float("inf")
                tloc = time.perf_counter()
                for sigma in sigma_candidates:
                    local_bg = _local_gaussian_estimate(target_smooth, mask, float(sigma))
                    pred_local = masked_input.copy()
                    pred_local[hole_sel] = local_bg[hole_sel]
                    mae_local = _masked_raw_mae(pred_local, target_smooth, mask)
                    if mae_local < best_local_mae:
                        best_local_mae = mae_local
                        best_sigma = float(sigma)
                        best_local_pred = pred_local
                rt_local = time.perf_counter() - tloc

                method_preds = {
                    f"sigma{int(sigma_smooth)}_zero_fill": (zero_pred, 0.0, math.nan),
                    f"sigma{int(sigma_smooth)}_global_mean_fill": (mean_pred, 0.0, math.nan),
                    f"sigma{int(sigma_smooth)}_global_median_fill": (median_pred, 0.0, 50.0),
                    f"sigma{int(sigma_smooth)}_global_percentile_fill": (best_cand_pred, 0.0, best_cand_percentile),
                    f"sigma{int(sigma_smooth)}_best_percentile_fill": (best_p_pred, 0.0, best_p),
                    f"sigma{int(sigma_smooth)}_local_gaussian_fill": (best_local_pred, rt_local, math.nan),
                    f"sigma{int(sigma_smooth)}_trained_model": (model_pred, rt_model, math.nan),
                }
                if frozen_pred is not None:
                    method_preds[f"sigma{int(sigma_smooth)}_frozen_lama"] = (frozen_pred, rt_frozen, math.nan)
                for m_name, (pred_m, rt_m, bp_m) in method_preds.items():
                    raw_m = _masked_raw_mae(pred_m, target_smooth, mask)
                    log_m = _masked_log_mae(pred_m, target_smooth, mask)
                    _accumulate(m_name, raw_m, log_m, float(rt_m), float(bp_m) if bp_m is not None else math.nan)

                if float(sigma_smooth) == float(default_smooth_sigma):
                    per_image_rows.append(
                        {
                            "image_index": i,
                            "best_percentile": best_p,
                            "best_percentile_mae": best_p_mae,
                            "zero_mae": _masked_raw_mae(zero_pred, target_smooth, mask),
                            "median_mae": _masked_raw_mae(median_pred, target_smooth, mask),
                            "local_gaussian_mae": best_local_mae,
                            "model_mae": _masked_raw_mae(model_pred, target_smooth, mask),
                        }
                    )

                    sample_dir = vis_dir / f"sample_{i:03d}"
                    sample_dir.mkdir(parents=True, exist_ok=True)

                    stack = np.concatenate([target_raw.ravel(), target_smooth.ravel(), masked_input.ravel()])
                    lo = float(np.percentile(stack, 1))
                    hi = float(np.percentile(stack, 99))
                    if hi <= lo:
                        lo, hi = float(np.min(stack)), float(np.max(stack))
                        if hi <= lo:
                            hi = lo + 1e-6

                    def _save_img(path: Path, arr: np.ndarray, is_mask: bool = False) -> None:
                        plt.figure(figsize=(5, 4))
                        if is_mask:
                            plt.imshow(arr, cmap="gray", vmin=0, vmax=1)
                        else:
                            plt.imshow(arr, cmap="gray", vmin=lo, vmax=hi)
                        plt.axis("off")
                        plt.tight_layout()
                        plt.savefig(path, dpi=140)
                        plt.close()

                    _save_img(sample_dir / "target_raw.png", target_raw)
                    _save_img(sample_dir / "target_smooth.png", target_smooth)
                    _save_img(sample_dir / "mask.png", mask, is_mask=True)
                    _save_img(sample_dir / "masked_input.png", masked_input)
                    _save_img(sample_dir / "prediction.png", model_pred)
                    _save_img(sample_dir / "absolute_error_vs_smooth.png", np.abs(model_pred - target_smooth) * mask, is_mask=False)

                    print(
                        f"[diag] idx={i} sigma={int(sigma_smooth)} best_percentile={best_p:.1f} "
                        f"best_percentile_mae={best_p_mae:.6f} zero_mae={_masked_raw_mae(zero_pred, target_smooth, mask):.6f} "
                        f"median_mae={_masked_raw_mae(median_pred, target_smooth, mask):.6f} "
                        f"local_gaussian_mae={best_local_mae:.6f} model_mae={_masked_raw_mae(model_pred, target_smooth, mask):.6f}"
                    )

    # Print pooled stats
    all_known_raw = np.concatenate(pool_known_raw) if pool_known_raw else np.array([], dtype=np.float32)
    all_masked_raw = np.concatenate(pool_masked_raw) if pool_masked_raw else np.array([], dtype=np.float32)
    all_known_smooth = np.concatenate(pool_known_smooth) if pool_known_smooth else np.array([], dtype=np.float32)
    all_masked_smooth = np.concatenate(pool_masked_smooth) if pool_masked_smooth else np.array([], dtype=np.float32)
    print(f"[diag] raw target[mask==0] stats: {_safe_stats(all_known_raw)}")
    print(f"[diag] raw target[mask==1] stats: {_safe_stats(all_masked_raw)}")
    print(f"[diag] smooth target[mask==0] stats: {_safe_stats(all_known_smooth)}")
    print(f"[diag] smooth target[mask==1] stats: {_safe_stats(all_masked_smooth)}")

    # Aggregate diagnostic scores CSV
    for method, s in sorted(sums.items()):
        n = max(1.0, s["n"])
        bp = s["bp"] / s["bp_n"] if s["bp_n"] > 0 else math.nan
        rows.append(
            {
                "method": method,
                "raw_missing_mae": s["raw"] / n,
                "log_missing_mae": s["log"] / n,
                "best_percentile": bp,
                "runtime_s": s["rt"] / n,
            }
        )

    csv_path = out_dir / "diagnostic_scores.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["method", "raw_missing_mae", "log_missing_mae", "best_percentile", "runtime_s"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"[diag] wrote: {csv_path}")

    # Requested shorthand summary for default sigma.
    def _find_metric(prefix: str) -> float:
        key = f"sigma{int(default_smooth_sigma)}_{prefix}"
        for row in rows:
            if row["method"] == key:
                return float(row["raw_missing_mae"])
        return float("nan")

    print(f"[diag] zero_fill_smooth_mae={_find_metric('zero_fill'):.6f}")
    print(f"[diag] global_median_smooth_mae={_find_metric('global_median_fill'):.6f}")
    print(f"[diag] local_gaussian_smooth_mae={_find_metric('local_gaussian_fill'):.6f}")
    print(f"[diag] frozen_lama_smooth_mae={_find_metric('frozen_lama'):.6f}")
    print(f"[diag] trained_model_smooth_mae={_find_metric('trained_model'):.6f}")

    # Extra per-image report for the requested interpretation fields.
    per_image_csv = out_dir / "diagnostic_per_image.csv"
    with per_image_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["image_index", "best_percentile", "best_percentile_mae", "zero_mae", "median_mae", "local_gaussian_mae", "model_mae"],
        )
        writer.writeheader()
        for row in per_image_rows:
            writer.writerow(row)
    print(f"[diag] wrote: {per_image_csv}")
    return rows


def _diag_raw_mae(rows: list[dict[str, float | str]], method_name: str) -> float:
    for row in rows:
        if str(row.get("method", "")) == method_name:
            try:
                return float(row.get("raw_missing_mae", float("nan")))
            except Exception:
                return float("nan")
    return float("nan")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train background reconstruction using precomputed masked data.")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "lama_big_adapter.yaml")
    parser.add_argument("--device", type=str, default=None, help="Override config device (e.g. cpu, cuda).")
    parser.add_argument(
        "--init-checkpoint",
        type=Path,
        default=None,
        help="Optional checkpoint to warm-start model weights (expects model_state or raw state_dict).",
    )
    parser.add_argument(
        "--resume-optimizer",
        action="store_true",
        help="If --init-checkpoint contains optimizer_state, resume optimizer state as well.",
    )
    parser.add_argument(
        "--diagnostic-only",
        action="store_true",
        help="Run diagnostic baselines on a fixed validation subset and exit without training.",
    )
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    print(f"[info] config={args.config.resolve()}")
    set_seed(int(cfg["seed"]))

    device_name = str(args.device if args.device is not None else cfg.get("device", "cpu")).lower()
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available. Use --device cpu.")
    device = torch.device(device_name)
    print(f"[info] device={device}")
    if device.type == "cuda":
        try:
            idx = device.index if device.index is not None else torch.cuda.current_device()
            print(f"[info] cuda_device={idx} name={torch.cuda.get_device_name(idx)}")
        except Exception:
            pass

    train_dir = PROJECT_ROOT / cfg["data"]["train_dir"]
    val_dir = PROJECT_ROOT / cfg["data"]["val_dir"]
    train_masked_dir = PROJECT_ROOT / cfg["data"].get("train_masked_dir", "data/train_masked_inputs")
    val_masked_dir = PROJECT_ROOT / cfg["data"].get("val_masked_dir", "data/val_masked_inputs")
    train_masks_dir = PROJECT_ROOT / cfg["data"].get("train_masks_dir", "data/train_masks")
    val_masks_dir = PROJECT_ROOT / cfg["data"].get("val_masks_dir", "data/val_masks")

    stack_tokens_train = _parse_tokens(cfg["data"].get("train_stack_tokens", ["145155"]))
    stack_tokens_val = _parse_tokens(cfg["data"].get("val_stack_tokens", stack_tokens_train))
    stack_tokens_test = _parse_tokens(cfg["data"].get("test_stack_tokens", stack_tokens_train))
    # Strict dataset restriction for diagnostics/training: first-video-derived frames only.
    stack_tokens_val = list(stack_tokens_train)
    stack_tokens_test = list(stack_tokens_train)
    print(f"[info] enforced stack tokens (train/val/test): {stack_tokens_train}")

    train_triplets = filter_triplets_by_tokens(
        build_triplets(train_dir, train_masked_dir, train_masks_dir),
        stack_tokens_train,
    )
    val_triplets = filter_triplets_by_tokens(
        build_triplets(val_dir, val_masked_dir, val_masks_dir),
        stack_tokens_val,
    )
    test_triplets = filter_triplets_by_tokens(
        build_triplets(val_dir, val_masked_dir, val_masks_dir),
        stack_tokens_test,
    )
    limit_train = int(cfg["data"].get("limit_train_triplets", 0) or 0)
    limit_val = int(cfg["data"].get("limit_val_triplets", 0) or 0)
    limit_test = int(cfg["data"].get("limit_test_triplets", 0) or 0)
    if limit_train > 0:
        train_triplets = train_triplets[:limit_train]
    if limit_val > 0:
        val_triplets = val_triplets[:limit_val]
    if limit_test > 0:
        test_triplets = test_triplets[:limit_test]
    if train_triplets and not val_triplets:
        val_triplets = [train_triplets[0]]
    if not test_triplets:
        test_triplets = val_triplets[:]

    if not train_triplets or not val_triplets:
        raise RuntimeError(
            "Required precomputed triplets not found.\n"
            f"train_triplets={len(train_triplets)} val_triplets={len(val_triplets)}"
        )

    print("[info] using ONLY precomputed pairs from your saved masks")
    print(f"[info] train_triplets={len(train_triplets)} val_triplets={len(val_triplets)}")
    print(f"[info] test_triplets={len(test_triplets)}")
    for i, (t, mi, m) in enumerate(train_triplets[:3], start=1):
        print(f"[info] train_sample_{i}: {t.name} | {mi.name} | {m.name}")
    for i, (t, mi, m) in enumerate(val_triplets[:2], start=1):
        print(f"[info] val_sample_{i}: {t.name} | {mi.name} | {m.name}")

    normalization_mode = str(cfg["data"].get("normalization_mode", "fixed_global")).lower()
    fixed_scale = float(cfg["data"].get("fixed_scale", 65535.0))

    train_ds = PrecomputedMaskedDataset(
        train_triplets,
        normalize_percentile=float(cfg["data"]["normalize_percentile"]),
        normalization_mode=normalization_mode,
        fixed_scale=fixed_scale,
        patch_size=_parse_optional_patch_size(cfg["data"].get("patch_size", 0)),
        samples_per_image=int(cfg["data"].get("samples_per_image", 1)),
        prefer_mask_center=bool(cfg["data"].get("prefer_mask_center", True)),
        target_mode=str(cfg["data"].get("target_mode", "smooth_background")),
        smooth_sigma=float(cfg["data"].get("smooth_sigma", 32.0)),
        smooth_mask_aware=bool(cfg["data"].get("smooth_mask_aware", True)),
        input_mode=str(cfg["data"].get("input_mode", "raw_masked")),
    )
    val_ds = PrecomputedMaskedDataset(
        val_triplets,
        normalize_percentile=float(cfg["data"]["normalize_percentile"]),
        normalization_mode=normalization_mode,
        fixed_scale=fixed_scale,
        patch_size=_parse_optional_patch_size(cfg["data"].get("patch_size", 0)),
        samples_per_image=max(1, int(cfg["data"].get("val_samples_per_image", 1))),
        prefer_mask_center=bool(cfg["data"].get("prefer_mask_center", True)),
        target_mode=str(cfg["data"].get("target_mode", "smooth_background")),
        smooth_sigma=float(cfg["data"].get("smooth_sigma", 32.0)),
        smooth_mask_aware=bool(cfg["data"].get("smooth_mask_aware", True)),
        input_mode=str(cfg["data"].get("input_mode", "raw_masked")),
    )
    # Full-frame validation sample for qualitative preview images.
    val_preview_ds = PrecomputedMaskedDataset(
        val_triplets,
        normalize_percentile=float(cfg["data"]["normalize_percentile"]),
        normalization_mode=normalization_mode,
        fixed_scale=fixed_scale,
        patch_size=None,
        samples_per_image=1,
        prefer_mask_center=False,
        target_mode=str(cfg["data"].get("target_mode", "smooth_background")),
        smooth_sigma=float(cfg["data"].get("smooth_sigma", 32.0)),
        smooth_mask_aware=bool(cfg["data"].get("smooth_mask_aware", True)),
        input_mode=str(cfg["data"].get("input_mode", "raw_masked")),
    )
    test_preview_ds = PrecomputedMaskedDataset(
        test_triplets,
        normalize_percentile=float(cfg["data"]["normalize_percentile"]),
        normalization_mode=normalization_mode,
        fixed_scale=fixed_scale,
        patch_size=None,
        samples_per_image=1,
        prefer_mask_center=False,
        target_mode=str(cfg["data"].get("target_mode", "smooth_background")),
        smooth_sigma=float(cfg["data"].get("smooth_sigma", 32.0)),
        smooth_mask_aware=bool(cfg["data"].get("smooth_mask_aware", True)),
        input_mode=str(cfg["data"].get("input_mode", "raw_masked")),
    )

    # One-time tensor sanity log for model inputs.
    sample_input = train_ds[0]["input"]
    c = int(sample_input.shape[0])
    print(f"[info] input tensor shape={tuple(sample_input.shape)} channels={c}")
    if c < 2:
        raise RuntimeError("Model input must contain 2 channels: [smoothed_masked_image, binary_mask].")
    for ch in range(c):
        ch_t = sample_input[ch]
        print(
            f"[info] input channel {ch}: "
            f"min={float(ch_t.min().item()):.6f} "
            f"max={float(ch_t.max().item()):.6f} "
            f"mean={float(ch_t.mean().item()):.6f}"
        )

    train_loader = DataLoader(
        train_ds,
        batch_size=int(cfg["train"]["batch_size"]),
        shuffle=True,
        num_workers=int(cfg["train"]["num_workers"]),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(cfg["train"]["batch_size"]),
        shuffle=False,
        num_workers=int(cfg["train"]["num_workers"]),
    )

    model = build_model(cfg["model"]).to(device)
    output_scale_init = float(cfg["model"].get("output_scale", 0.01))
    if hasattr(model, "output_scale") and isinstance(model.output_scale, torch.Tensor):
        with torch.no_grad():
            model.output_scale.fill_(output_scale_init)
    else:
        model.output_scale = torch.nn.Parameter(torch.tensor(output_scale_init, dtype=torch.float32, device=device))
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[info] model_name={cfg['model'].get('name', 'unknown')}")
    print(f"[info] model_params={n_params}")
    print(f"[info] model.output_scale_init={output_scale_init}")
    print(f"[info] model.output_scale={float(model.output_scale.detach().item()):.6f}")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["train"]["lr"]),
        weight_decay=float(cfg["train"]["weight_decay"]),
    )

    if args.init_checkpoint is not None:
        ckpt_path = args.init_checkpoint.resolve()
        if not ckpt_path.exists():
            raise FileNotFoundError(f"init checkpoint not found: {ckpt_path}")
        print(f"[info] loading init checkpoint: {ckpt_path}")
        loaded = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if isinstance(loaded, dict) and "model_state" in loaded and isinstance(loaded["model_state"], dict):
            state_dict = loaded["model_state"]
        elif isinstance(loaded, dict):
            state_dict = loaded
        else:
            raise RuntimeError(f"Unsupported checkpoint format in {ckpt_path}")

        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print(f"[info] init load missing_keys={len(missing)} unexpected_keys={len(unexpected)}")
        if missing:
            print(f"[info] sample missing keys: {missing[:8]}")
        if unexpected:
            print(f"[info] sample unexpected keys: {unexpected[:8]}")

        if args.resume_optimizer and isinstance(loaded, dict) and "optimizer_state" in loaded:
            try:
                optimizer.load_state_dict(loaded["optimizer_state"])
                print("[info] optimizer state resumed from init checkpoint")
            except Exception as exc:
                print(f"[warn] failed to resume optimizer_state: {exc}")

    ckpt_dir = PROJECT_ROOT / cfg["output"]["checkpoints_dir"]
    preview_dir = PROJECT_ROOT / cfg["output"]["preview_dir"]
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    preview_dir.mkdir(parents=True, exist_ok=True)

    # Clear previous run previews so only current run images are visible.
    for old in preview_dir.glob("epoch_*.png"):
        try:
            old.unlink()
        except Exception:
            pass

    epochs = int(cfg["train"]["epochs"])
    save_every = int(cfg["train"]["save_every"])
    masked_weight = float(cfg["train"].get("masked_weight", 1.0))
    visible_weight = float(cfg["train"].get("visible_weight", 0.0))
    loss_space = str(cfg["train"].get("loss_space", "linear")).lower()
    if loss_space not in {"linear", "log1p"}:
        raise RuntimeError(f"Unsupported train.loss_space: {loss_space}")
    loss_type = str(cfg["train"].get("loss_type", "legacy")).lower()
    if loss_type not in {"legacy", "lama_masked_l1", "masked_smooth_l1", "masked_l1", "masked_l1_simple"}:
        raise RuntimeError(f"Unsupported train.loss_type: {loss_type}")
    huber_beta = float(cfg["train"].get("huber_beta", 0.01))
    lama_weight_known = float(cfg["train"].get("lama_weight_known", 0.0))
    lama_weight_missing = float(cfg["train"].get("lama_weight_missing", 1.0))
    # Remove known-region loss from optimization and ranking.
    if loss_type == "lama_masked_l1":
        lama_weight_known = 0.0
        lama_weight_missing = 1.0
    else:
        masked_weight = 1.0
        visible_weight = 0.0
    if loss_type == "masked_l1_simple":
        loss_space = "linear"
    print(f"[info] loss_space={loss_space}")
    print(
        f"[info] loss_type={loss_type} "
        f"(lama_weight_known={lama_weight_known}, lama_weight_missing={lama_weight_missing})"
    )
    if loss_type == "masked_smooth_l1":
        print(f"[info] masked SmoothL1 beta={huber_beta}")
    print(f"[info] masked-only optimization active (masked_weight={masked_weight}, visible_weight={visible_weight})")
    print(
        "[info] dataset settings: "
        f"target_mode={cfg['data'].get('target_mode', 'smooth_background')} "
        f"input_mode={cfg['data'].get('input_mode', 'raw_masked')} "
        f"smooth_sigma={cfg['data'].get('smooth_sigma', 32)} "
        f"smooth_mask_aware={cfg['data'].get('smooth_mask_aware', True)} "
        f"normalization_mode={cfg['data'].get('normalization_mode', 'fixed_global')} "
        f"fixed_scale={cfg['data'].get('fixed_scale', 65535)} "
        f"patch_size={cfg['data'].get('patch_size')} "
        f"samples_per_image={cfg['data'].get('samples_per_image')} "
        f"val_samples_per_image={cfg['data'].get('val_samples_per_image', 1)} "
        f"prefer_mask_center={cfg['data'].get('prefer_mask_center', True)}"
    )
    print(
        "[info] fixed val preview source: "
        f"{val_triplets[0][0].name} | {val_triplets[0][1].name} | {val_triplets[0][2].name}"
    )
    zero_fill_baseline_val = evaluate_zero_fill_baseline(
        val_loader,
        device=device,
        masked_weight=masked_weight,
        visible_weight=visible_weight,
        loss_space=loss_space,
        loss_type=loss_type,
        lama_weight_known=lama_weight_known,
        lama_weight_missing=lama_weight_missing,
        huber_beta=huber_beta,
    )
    print(f"[info] val zero-fill baseline loss={zero_fill_baseline_val:.6f}")
    lama_backbone_val = evaluate_lama_backbone_baseline(
        model=model,
        loader=val_loader,
        device=device,
        masked_weight=masked_weight,
        visible_weight=visible_weight,
        loss_space=loss_space,
        loss_type=loss_type,
        lama_weight_known=lama_weight_known,
        lama_weight_missing=lama_weight_missing,
        huber_beta=huber_beta,
    )
    if lama_backbone_val is not None:
        print(f"[info] val frozen-lama baseline loss={lama_backbone_val:.6f}")

    diag_cfg = cfg.get("diagnostic", {}) if isinstance(cfg, dict) else {}
    diag_enabled = bool(diag_cfg.get("enabled", True))
    diag_num_samples = int(diag_cfg.get("num_val_samples", 4))
    diag_out_dir = PROJECT_ROOT / diag_cfg.get("output_dir", "outputs/diagnostics")
    diag_smooth_sigmas = [float(v) for v in diag_cfg.get("smooth_sigma_values", [16, 32, 64])]
    diag_default_sigma = float(cfg["data"].get("smooth_sigma", 32.0))
    diag_smooth_mask_aware = bool(cfg["data"].get("smooth_mask_aware", True))

    # One-sample explicit mask/target/input verification.
    debug_mask_target_sample(
        ds=val_preview_ds,
        out_dir=diag_out_dir,
        sample_index=0,
    )

    pre_diag_rows: list[dict[str, float | str]] = []
    if diag_enabled or args.diagnostic_only:
        pre_diag_rows = run_diagnostics(
            model=model,
            val_preview_ds=val_preview_ds,
            device=device,
            out_dir=diag_out_dir,
            num_samples=diag_num_samples,
            smooth_sigma_values=diag_smooth_sigmas,
            default_smooth_sigma=diag_default_sigma,
            smooth_mask_aware=diag_smooth_mask_aware,
        )
    if args.diagnostic_only:
        print("[done] diagnostic-only run complete.")
        return

    best_val = float("inf")
    history = {"train_loss": [], "val_loss": []}
    stop_event = threading.Event()
    _ = start_stop_listener(stop_event)
    bad_batch_loss_threshold = cfg["train"].get("bad_batch_loss_threshold", None)
    if bad_batch_loss_threshold is not None:
        bad_batch_loss_threshold = float(bad_batch_loss_threshold)
    print("[info] training started. Press 'q' to stop and save outputs.")

    try:
        scale_log_once = {"done": False}
        for epoch in range(1, epochs + 1):
            if stop_event.is_set():
                break

            train_stats = run_epoch(
                model,
                train_loader,
                device,
                optimizer,
                masked_weight,
                visible_weight,
                loss_space,
                loss_type,
                lama_weight_known,
                lama_weight_missing,
                huber_beta,
                True,
                epoch=epoch,
                epochs_total=epochs,
                stop_event=stop_event,
                bad_batch_loss_threshold=bad_batch_loss_threshold,
                log_scale_stats_once=scale_log_once,
            )
            train_loss = float(train_stats["loss"])
            if stop_event.is_set():
                history["train_loss"].append(train_loss)
                history["val_loss"].append(float("nan"))
                print(f"[epoch {epoch:03d}/{epochs:03d}] train={train_loss:.6f} val=nan (stopped during/after train)")
                break

            val_stats = run_epoch(
                model,
                val_loader,
                device,
                optimizer,
                masked_weight,
                visible_weight,
                loss_space,
                loss_type,
                lama_weight_known,
                lama_weight_missing,
                huber_beta,
                False,
                epoch=epoch,
                epochs_total=epochs,
                stop_event=stop_event,
                bad_batch_loss_threshold=bad_batch_loss_threshold,
            )
            val_loss = float(val_stats["loss"])
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            print(f"[epoch {epoch:03d}/{epochs:03d}] train={train_loss:.6f} val={val_loss:.6f}")
            print(f"[epoch {epoch:03d}/{epochs:03d}] output_scale={float(model.output_scale.detach().item()):.6f}")
            print(
                f"[train stats] masked_loss_model={train_stats['masked_loss_model']:.6f} "
                f"masked_loss_zero_fill={train_stats['masked_loss_zero_fill']:.6f} "
                f"masked_target_mean={train_stats['masked_target_mean']:.6f} "
                f"masked_target_std={train_stats['masked_target_std']:.6f} "
                f"masked_prediction_mean={train_stats['masked_prediction_mean']:.6f} "
                f"masked_prediction_std={train_stats['masked_prediction_std']:.6f} "
                f"mask_fraction={train_stats['mask_fraction']:.6f}"
            )
            print(
                f"[val stats] masked_loss_model={val_stats['masked_loss_model']:.6f} "
                f"masked_loss_zero_fill={val_stats['masked_loss_zero_fill']:.6f} "
                f"masked_target_mean={val_stats['masked_target_mean']:.6f} "
                f"masked_target_std={val_stats['masked_target_std']:.6f} "
                f"masked_prediction_mean={val_stats['masked_prediction_mean']:.6f} "
                f"masked_prediction_std={val_stats['masked_prediction_std']:.6f} "
                f"mask_fraction={val_stats['mask_fraction']:.6f}"
            )

            save_fixed_val_preview(
                model=model,
                val_ds=val_preview_ds,
                device=device,
                out_path=preview_dir / f"epoch_{epoch:03d}.png",
                sample_index=0,
            )
            save_multi_split_preview(
                model=model,
                val_ds=val_preview_ds,
                test_ds=test_preview_ds,
                device=device,
                out_path=preview_dir / f"epoch_{epoch:03d}_val2_test2.png",
            )

            state = {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "config": cfg,
                "history": history,
            }
            if epoch % save_every == 0:
                save_checkpoint(ckpt_dir / f"checkpoint_epoch_{epoch:03d}.pt", state)
            if val_loss < best_val:
                best_val = val_loss
                save_checkpoint(ckpt_dir / "best.pt", state)
    except KeyboardInterrupt:
        print("\n[stop] KeyboardInterrupt received. Finishing and saving outputs.")
        stop_event.set()

    last_epoch = len(history["train_loss"])
    state_last = {
        "epoch": last_epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "config": cfg,
        "history": history,
    }
    save_checkpoint(ckpt_dir / "last.pt", state_last)
    save_json(ckpt_dir / "train_history.json", history)
    save_loss_plot(ckpt_dir / "loss_vs_epoch.png", history)

    # Post-training diagnostic pass on the same fixed validation subset.
    post_diag_rows = run_diagnostics(
        model=model,
        val_preview_ds=val_preview_ds,
        device=device,
        out_dir=diag_out_dir,
        num_samples=diag_num_samples,
        smooth_sigma_values=diag_smooth_sigmas,
        default_smooth_sigma=diag_default_sigma,
        smooth_mask_aware=diag_smooth_mask_aware,
    )
    sigma_tag = f"sigma{int(diag_default_sigma)}"
    model_name = f"{sigma_tag}_trained_model"
    zero_name = f"{sigma_tag}_zero_fill"
    local_name = f"{sigma_tag}_local_gaussian_fill"
    model_mae = _diag_raw_mae(post_diag_rows, model_name)
    zero_mae = _diag_raw_mae(post_diag_rows, zero_name)
    local_mae = _diag_raw_mae(post_diag_rows, local_name)
    print(f"[summary] best val masked loss={best_val:.6f}" if best_val < float("inf") else "[summary] best val masked loss=nan")
    if np.isfinite(zero_mae):
        print(f"[summary] zero-fill masked loss={zero_mae:.6f}")
    if np.isfinite(local_mae):
        print(f"[summary] local Gaussian masked loss={local_mae:.6f}")
    if np.isfinite(model_mae) and np.isfinite(zero_mae):
        print(f"[summary] trained model beats zero fill={model_mae < zero_mae} (model={model_mae:.6f}, zero={zero_mae:.6f})")
    if np.isfinite(model_mae) and np.isfinite(local_mae):
        print(f"[summary] trained model beats local Gaussian={model_mae < local_mae} (model={model_mae:.6f}, local={local_mae:.6f})")
    if pre_diag_rows and post_diag_rows:
        pre_mae = _diag_raw_mae(pre_diag_rows, model_name)
        if np.isfinite(pre_mae) and np.isfinite(model_mae):
            print(f"[summary] {model_name} pre_to_post_delta={model_mae - pre_mae:+.6f} (negative is better)")
    print(f"[summary] preview output path={preview_dir}")
    if best_val < float("inf"):
        print(f"[done] epochs_completed={last_epoch} best_val={best_val:.6f}")
    else:
        print(f"[done] epochs_completed={last_epoch} (no completed val epoch)")
    print(f"[done] wrote: {ckpt_dir / 'train_history.json'}")
    print(f"[done] wrote: {ckpt_dir / 'loss_vs_epoch.png'}")
    print(f"[done] wrote: {ckpt_dir / 'last.pt'}")


if __name__ == "__main__":
    main()
