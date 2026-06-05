from typing import Any, Optional
from torch import Tensor

from .path import ProbPath
from .path_sample import PathSample


class AffinePath(ProbPath):
    r"""Affine probability path using linear interpolation.

    The path is defined by:
        x_t = (1 - t) * x_0 + t * x_1
    and the velocity is constant:
        dx_t = x_1 - x_0
    """
    def __init__(self, scheduler: Optional[Any] = None):
        """Optional scheduler can be provided for compatibility with examples.

        The scheduler is stored as `self.scheduler` but is not required for
        the affine (linear) path operations.
        """
        self.scheduler = scheduler

    def sample(self, x_0: Tensor, x_1: Tensor, t: Tensor) -> PathSample:
        self.assert_sample_shape(x_0=x_0, x_1=x_1, t=t)

        batch_size = t.shape[0]
        if x_0.ndim == 1:
            t_reshaped = t
        else:
            t_reshaped = t.view(batch_size, *([1] * (x_0.ndim - 1)))

        x_t = (1.0 - t_reshaped) * x_0 + t_reshaped * x_1
        dx_t = x_1 - x_0

        return PathSample(x_t=x_t, dx_t=dx_t, x_1=x_1, x_0=x_0, t=t)
