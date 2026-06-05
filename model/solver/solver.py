from abc import ABC, abstractmethod

from torch import nn, Tensor


class Solver(ABC, nn.Module):
    """Abstract base class for solver implementations."""

    @abstractmethod
    def sample(self, x_init: Tensor = None, *args, **kwargs) -> Tensor:
        """Returns a sample produced by the solver."""
        ...
