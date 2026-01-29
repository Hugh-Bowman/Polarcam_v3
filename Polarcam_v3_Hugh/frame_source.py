from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterator
import numpy as np


class FrameSource(ABC):
    """
    Abstract base class defining the contract for a frame source.

    Subclasses MUST set:
      - self.shape: (H, W)
      - self.dtype: numpy dtype

    and MUST implement:
      - frames(): yields 2D numpy arrays with shape == self.shape
    """

    def __init__(self) -> None:
        self.shape: tuple[int, int]
        self.dtype: np.dtype

    @abstractmethod
    def frames(self) -> Iterator[np.ndarray]:
        raise NotImplementedError
