from abc import ABC, abstractmethod

from torch import Tensor

from .path_sample import PathSample


class ProbPath(ABC):
    r"""Abstract class representing a probability path.

    A probability path transforms the distribution p(X_0) into p(X_1) over t=0->1.

    The ProbPath class is designed to support model training in the flow matching framework.
    It supports two key features: (1) sampling a conditional probability path and (2)
    converting between different training objectives.
    """

    @abstractmethod
    def sample(self, x_0: Tensor, x_1: Tensor, t: Tensor) -> PathSample:
        r"""Sample from the probability path.

        Args:
            x_0 (Tensor): source samples, shape (batch_size, ...).
            x_1 (Tensor): target samples, shape (batch_size, ...).
            t (Tensor): times in [0,1], shape (batch_size,).

        Returns:
            PathSample: a conditional path sample.
        """
        raise NotImplementedError

    def assert_sample_shape(self, x_0: Tensor, x_1: Tensor, t: Tensor):
        assert (
            t.ndim == 1
        ), f"The time vector t must have shape [batch_size]. Got {t.shape}."
        assert (
            t.shape[0] == x_0.shape[0] == x_1.shape[0]
        ), f"Time t batch size must match x_0 and x_1. Got {t.shape} vs {x_0.shape} / {x_1.shape}."
