import numpy as np
import time
#matrix based computation of Q and U without resolution loss.
def make_qu_reconstructor(frame_shape, out_dtype=np.int16):
    """
    Fast Q/U reconstruction for fixed frame shape (H, W).

    Input:
      - I: uint8 or uint16 (e.g. 12-bit packed in uint16), shape (H, W)

    Output:
      - Q, U: shape (H-1, W-1), dtype out_dtype (default int16)

    Notes:
      - Preallocates all buffers once (A, B, Q, U).
      - Returns internal Q/U buffers. If you need to keep outputs across frames,
        copy them: Q.copy(), U.copy().
    """
    H, W = frame_shape
    Hn, Wn = H - 1, W - 1

    # Preallocate node-grid intermediates and outputs
    A = np.empty((Hn, Wn), dtype=out_dtype)  # NW - SE
    B = np.empty((Hn, Wn), dtype=out_dtype)  # NE - SW
    Q = np.empty((Hn, Wn), dtype=out_dtype)
    U = np.empty((Hn, Wn), dtype=out_dtype)

    # Parity slice blocks on the node grid
    ee = (slice(0, None, 2), slice(0, None, 2))  # even-even
    oo = (slice(1, None, 2), slice(1, None, 2))  # odd-odd
    eo = (slice(0, None, 2), slice(1, None, 2))  # even-odd
    oe = (slice(1, None, 2), slice(0, None, 2))  # odd-even

    def reconstruct(I_in):
        I = np.asarray(I_in)
        if I.shape != (H, W):
            raise ValueError(f"Expected frame shape {(H, W)}, got {I.shape}")

        # Compute diagonal differences into preallocated A, B (no allocation)
        # A = I[:-1,:-1] - I[1:,1:]
        np.subtract(I[:-1, :-1], I[1:, 1:], out=A, dtype=out_dtype)

        # B = I[:-1,1:] - I[1:,:-1]
        np.subtract(I[:-1, 1:], I[1:, :-1], out=B, dtype=out_dtype)

        # Interleave A/B into Q/U (no masks, always shape-consistent)
        # Q: TL from A => A on (ee, oo), B on (eo, oe)
        Q[ee] = A[ee]
        Q[oo] = A[oo]
        Q[eo] = B[eo]
        Q[oe] = B[oe]

        # U: TL from B => B on (ee, oo), A on (eo, oe)
        U[ee] = B[ee]
        U[oo] = B[oo]
        U[eo] = A[eo]
        U[oe] = A[oe]

        # Row-wise sign correction (in-place)
        # Q: top row incorrect => flip rows 0,2,4,...
        Q[0::2, :] *= -1
        # U: top row correct => flip rows 1,3,5,...
        U[1::2, :] *= -1

        return Q, U

    return reconstruct


def make_xy_reconstructor(
    frame_shape, eps: float = 1e-6, out_dtype=np.float32, normalize: bool = True
):
    """
    Compute normalized anisotropies per 2x2 superpixel:
      X = (I0 - I90) / (I0 + I90 + eps)
      Y = (I45 - I135) / (I45 + I135 + eps)

    Input frame is the raw polarization mosaic (H, W), with:
      I90  at (0,0), I45 at (0,1), I135 at (1,0), I0 at (1,1) within each 2x2 block.

    Output:
      X, Y: shape (H/2, W/2), dtype out_dtype.
    normalize : bool
      If True (default), divide by the corresponding intensity sum (normalised anisotropy).
      If False, keep the raw difference (I0-I90, I45-I135) for S-map range metrics.
    """
    H, W = frame_shape
    if (H % 2) != 0 or (W % 2) != 0:
        raise ValueError(f"Expected even frame shape (H,W), got {(H, W)}")

    ih, iw = H // 2, W // 2
    X = np.empty((ih, iw), dtype=out_dtype)
    Y = np.empty((ih, iw), dtype=out_dtype)

    num = np.empty((ih, iw), dtype=np.float32)
    den = np.empty((ih, iw), dtype=np.float32)
    eps_f = np.float32(eps)
    norm_enabled = bool(normalize)

    def reconstruct(I_in):
        I = np.asarray(I_in)
        if I.shape != (H, W):
            raise ValueError(f"Expected frame shape {(H, W)}, got {I.shape}")

        I90 = I[0::2, 0::2]
        I45 = I[0::2, 1::2]
        I135 = I[1::2, 0::2]
        I0 = I[1::2, 1::2]

        # X
        np.subtract(I0, I90, out=num, dtype=np.float32)
        if norm_enabled:
            np.add(I0, I90, out=den, dtype=np.float32)
            np.add(den, eps_f, out=den)
            np.divide(num, den, out=X)
        else:
            np.copyto(X, num)

        # Y
        np.subtract(I45, I135, out=num, dtype=np.float32)
        if norm_enabled:
            np.add(I45, I135, out=den, dtype=np.float32)
            np.add(den, eps_f, out=den)
            np.divide(num, den, out=Y)
        else:
            np.copyto(Y, num)

        return X, Y

    return reconstruct


# ------------------ Testing speed of polarisation reconstruction prcoess ---------
# if __name__ == "__main__":
#     H, W = 1000, 2000

#     I = np.random.randint(0, 256,size=(H, W),dtype=np.uint8)

#     recon = make_qu_reconstructor(I.shape)
#     for _ in range(10):
#         recon(I)

#     # ---- timing block ----
#     N = 20000  # number of frames to test

#     t0 = time.perf_counter()
#     for _ in range(N):
#         Q, U = recon(I)
#     t1 = time.perf_counter()

#     dt = t1 - t0
#     fps = N / dt

#     print(f"Total time: {dt:.3f} s")
#     print(f"Per-frame: {dt/N*1e3:.3f} ms")
#     print(f"Throughput: {fps:.1f} FPS")
#     Q, U = recon(I)

#     #print("I:")
#     #print(I)
#     #print("\nQ:")
#     #print(Q)
#     #print("\nU:")
#     #print(U)
#     print("done")
