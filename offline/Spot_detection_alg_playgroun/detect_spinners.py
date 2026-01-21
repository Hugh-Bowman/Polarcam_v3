# detect_spinners.py
from __future__ import annotations

import numpy as np

# -----------------------------------------------------------------------------
# TUNABLE MIXING WEIGHT
# a = 0.0  -> pure "change" metric (median S_t^2)
# a = 1.0  -> pure "intensity" metric (median (Q^2 + U^2))
# 0<a<1    -> linear blend:  a*I_med + (1-a)*st2_med
# -----------------------------------------------------------------------------
a: float = 0.1  # <-- set this between 0 and 1


def median_st2(Q10: np.ndarray, U10: np.ndarray) -> np.ndarray:
    """
    Return a blended per-pixel map:
        out = a * I_med + (1-a) * st2_med

    where
        st2_med = median_t( (Q[t+1]-Q[t])^2 + (U[t+1]-U[t])^2 )
        I_med   = median_t( Q[t]^2 + U[t]^2 )

    Parameters
    ----------
    Q10, U10 : np.ndarray
        Arrays of shape (10, H, W). Any integer/float dtype is accepted.

    Returns
    -------
    out : np.ndarray
        Array of shape (H, W), dtype float32.
    """
    if Q10.shape != U10.shape:
        raise ValueError(
            f"Q10 and U10 must have same shape, got {Q10.shape} vs {U10.shape}"
        )
    if Q10.ndim != 3:
        raise ValueError(
            f"Expected Q10/U10 shape (10,H,W); got ndim={Q10.ndim}, shape={Q10.shape}"
        )
    if Q10.shape[0] < 2:
        raise ValueError("Need at least 2 frames to compute differences.")

    # Clamp a defensively (so GUI doesn't crash if you accidentally set it out of range)
    aa = float(a)
    if aa < 0.0:
        aa = 0.0
    elif aa > 1.0:
        aa = 1.0

    # Promote before squaring to avoid overflow for int16/int32.
    Q = Q10.astype(np.int32, copy=False)
    U = U10.astype(np.int32, copy=False)

    # --- Change metric: median S_t^2 over time differences ---
    dQ = np.diff(Q, axis=0)  # (T-1, H, W)
    dU = np.diff(U, axis=0)
    st2 = dQ * dQ + dU * dU
    st2_med = np.median(st2, axis=0).astype(np.float32, copy=False)

    # --- Intensity metric: median (Q^2 + U^2) over time ---
    I = Q * Q + U * U
    I_med = np.median(I, axis=0).astype(np.float32, copy=False)

    # Blend
    out = aa * I_med + (1.0 - aa) * st2_med
    return out.astype(np.float32, copy=False)


def to_u8_preview(
    img: np.ndarray, lo_pct: float = 10.0, hi_pct: float = 100.0
) -> np.ndarray:
    """
    Convert a 2D image to uint8 for display using percentile clipping,
    but treat exact zeros as background and keep them at 0 in the output.

    Percentiles are computed over NONZERO pixels only.

    Parameters
    ----------
    img : np.ndarray
        2D array (H, W)
    lo_pct, hi_pct : float
        Percentiles for contrast stretching (computed over nonzero pixels).

    Returns
    -------
    u8 : np.ndarray
        uint8 image (H, W)
    """
    if img.ndim != 2:
        raise ValueError(f"Expected 2D image, got shape {img.shape}")

    x = img.astype(np.float32, copy=False)

    zero_mask = (x == 0.0)
    nz = x[~zero_mask]
    if nz.size == 0:
        return np.zeros_like(x, dtype=np.uint8)

    lo = float(np.percentile(nz, lo_pct))
    hi = float(np.percentile(nz, hi_pct))

    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo = float(np.min(nz))
        hi = float(np.max(nz))
        if hi <= lo:
            return np.zeros_like(x, dtype=np.uint8)

    y = (x - lo) / (hi - lo)
    y = np.clip(y, 0.0, 1.0)

    u8 = (y * 255.0 + 0.5).astype(np.uint8)
    u8[zero_mask] = 0
    return u8
