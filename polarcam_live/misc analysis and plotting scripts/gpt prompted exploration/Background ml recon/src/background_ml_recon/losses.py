from __future__ import annotations

import torch
import torch.nn.functional as F


def masked_reconstruction_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    masked_weight: float = 1.0,
    visible_weight: float = 0.1,
    loss_space: str = "linear",
    loss_type: str = "legacy",
    lama_weight_known: float = 10.0,
    lama_weight_missing: float = 0.0,
    huber_beta: float = 0.01,
) -> tuple[torch.Tensor, dict[str, float]]:
    if loss_type == "masked_l1_simple":
        sel = mask > 0.5
        if sel.any():
            abs_err = torch.abs(pred - target)
            total = abs_err[sel].mean()
            masked_l1 = total
        else:
            total = torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
            masked_l1 = torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
        with torch.no_grad():
            vis_sel = mask <= 0.5
            visible_l1 = torch.abs(pred[vis_sel] - target[vis_sel]).mean() if vis_sel.any() else torch.tensor(0.0, device=pred.device)
            mse = F.mse_loss(pred, target).item()
            logs = {
                "loss_total": float(total.item()),
                "loss_masked_l1": float(masked_l1.item()),
                "loss_visible_l1": float(visible_l1.item()),
                "mse": float(mse),
                "loss_space": "linear",
                "loss_type": "masked_l1_simple",
            }
        return total, logs

    if loss_type == "masked_smooth_l1":
        sel = mask > 0.5
        if sel.any():
            total = F.smooth_l1_loss(pred[sel], target[sel], beta=float(huber_beta), reduction="mean")
            masked_l1 = torch.abs(pred[sel] - target[sel]).mean()
        else:
            total = torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
            masked_l1 = torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
        with torch.no_grad():
            vis_sel = mask <= 0.5
            visible_l1 = torch.abs(pred[vis_sel] - target[vis_sel]).mean() if vis_sel.any() else torch.tensor(0.0, device=pred.device)
            mse = F.mse_loss(pred, target).item()
            logs = {
                "loss_total": float(total.item()),
                "loss_masked_l1": float(masked_l1.item()),
                "loss_visible_l1": float(visible_l1.item()),
                "mse": float(mse),
                "loss_space": "linear",
                "loss_type": "masked_smooth_l1",
                "huber_beta": float(huber_beta),
            }
        return total, logs

    if loss_type == "masked_l1":
        sel = mask > 0.5
        if sel.any():
            total = torch.abs(pred[sel] - target[sel]).mean()
        else:
            total = torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
        with torch.no_grad():
            vis_sel = mask <= 0.5
            visible_l1 = torch.abs(pred[vis_sel] - target[vis_sel]).mean() if vis_sel.any() else torch.tensor(0.0, device=pred.device)
            mse = F.mse_loss(pred, target).item()
            logs = {
                "loss_total": float(total.item()),
                "loss_masked_l1": float(total.item()),
                "loss_visible_l1": float(visible_l1.item()),
                "mse": float(mse),
                "loss_space": "linear",
                "loss_type": "masked_l1",
            }
        return total, logs

    if loss_type == "lama_masked_l1":
        # Matches LaMa repository masked_l1 weighting:
        # pixel_weights = mask * weight_missing + (1 - mask) * weight_known
        # where mask=1 means hole/missing region.
        per_pixel_l1 = F.l1_loss(pred, target, reduction="none")
        pixel_weights = mask * float(lama_weight_missing) + (1.0 - mask) * float(lama_weight_known)
        total = (pixel_weights * per_pixel_l1).mean()
        with torch.no_grad():
            mse = F.mse_loss(pred, target).item()
            logs = {
                "loss_total": float(total.item()),
                "loss_masked_l1": float((per_pixel_l1 * mask).mean().item()),
                "loss_visible_l1": float((per_pixel_l1 * (1.0 - mask)).mean().item()),
                "mse": float(mse),
                "loss_space": "linear",
                "loss_type": "lama_masked_l1",
                "lama_weight_known": float(lama_weight_known),
                "lama_weight_missing": float(lama_weight_missing),
            }
        return total, logs

    if loss_space == "log1p":
        # Standard dynamic-range compression: error is measured on log(1+I),
        # so large-intensity targets are not disproportionately penalized.
        # Softplus preserves positivity while keeping gradients non-zero.
        pred_s = torch.log1p(F.softplus(pred))
        target_s = torch.log1p(torch.clamp_min(target, 0.0))
    else:
        pred_s = pred
        target_s = target

    abs_err = torch.abs(pred_s - target_s)
    masked = mask
    visible = 1.0 - mask

    masked_den = masked.sum().clamp_min(1.0)
    visible_den = visible.sum().clamp_min(1.0)

    masked_l1 = (abs_err * masked).sum() / masked_den
    visible_l1 = (abs_err * visible).sum() / visible_den
    total = masked_weight * masked_l1 + visible_weight * visible_l1

    with torch.no_grad():
        mse = F.mse_loss(pred, target).item()
        logs = {
            "loss_total": float(total.item()),
            "loss_masked_l1": float(masked_l1.item()),
            "loss_visible_l1": float(visible_l1.item()),
            "mse": float(mse),
            "loss_space": loss_space,
            "loss_type": "legacy",
        }
    return total, logs
