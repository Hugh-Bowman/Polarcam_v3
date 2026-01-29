from __future__ import annotations

import numpy as np

from avi_source import AviSource
from frame_source import FrameSource
from pol_reconstruction import make_qu_reconstructor


def process_stream(source: FrameSource, *, max_frames: int | None = None) -> None:
    """
    Generic processing loop that works for AVI today and live camera tomorrow.
    """
    recon = make_qu_reconstructor(source.shape, out_dtype=np.int16)

    for k, frame in enumerate(source.frames()):
        # frame is 2D, shape == source.shape
        Q, U = recon(frame)

        # TODO: replace this with your real downstream processing
        _ = int(Q[0, 0]) + int(U[0, 0])

        if max_frames is not None and (k + 1) >= max_frames:
            break


if __name__ == "__main__":
    # Example usage with an AVI file
    source = AviSource("/Volumes/HUGHSB/RAOxwork/IDS_peak_raw_videos/Sparse_stationary_rods_various_exposure/sparse_rods_0_05ms.avi", force_gray=True)
    process_stream(source, max_frames=None)
