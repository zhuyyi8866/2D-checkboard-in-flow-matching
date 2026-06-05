from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Union

import torch
from torch import Tensor


@dataclass
class SchedulerOutput:
    r"""Represents scheduler outputs for a time-dependent flow matching path.

    Attributes:
        alpha_t (Tensor): α_t.
        sigma_t (Tensor): σ_t.
        d_alpha_t (Tensor): ∂_t α_t.
        d_sigma_t (Tensor): ∂_t σ_t.
    """

    alpha_t: Tensor = field(metadata={"help": "alpha_t"})
    sigma_t: Tensor = field(metadata={"help": "sigma_t"})
    d_alpha_t: Tensor = field(metadata={"help": "Derivative of alpha_t."})
    d_sigma_t: Tensor = field(metadata={"help": "Derivative of sigma_t."})


class Scheduler(ABC):
    """Base scheduler class for flow matching paths."""

    @abstractmethod
    def __call__(self, t: Tensor) -> SchedulerOutput:
        r"""Return schedule parameters at times t.

        Args:
            t (Tensor): times in [0,1], shape (...).

        Returns:
            SchedulerOutput: α_t, σ_t and derivatives.
        """
        ...

    @abstractmethod
    def snr_inverse(self, snr: Tensor) -> Tensor:
        r"""Compute time t from signal-to-noise ratio α_t / σ_t."""
        ...


class ConvexScheduler(Scheduler):
    """Scheduler for convex paths with α_t + σ_t = 1."""

    @abstractmethod
    def __call__(self, t: Tensor) -> SchedulerOutput:
        ...

    @abstractmethod
    def kappa_inverse(self, kappa: Tensor) -> Tensor:
        r"""Compute time t from κ_t = α_t / (α_t + σ_t)."""
        ...

    def snr_inverse(self, snr: Tensor) -> Tensor:
        kappa_t = snr / (1.0 + snr)
        return self.kappa_inverse(kappa=kappa_t)


class CondOTScheduler(ConvexScheduler):
    """Conditional optimal transport scheduler: α_t = t, σ_t = 1 - t."""

    def __call__(self, t: Tensor) -> SchedulerOutput:
        return SchedulerOutput(
            alpha_t=t,
            sigma_t=1 - t,
            d_alpha_t=torch.ones_like(t),
            d_sigma_t=-torch.ones_like(t),
        )

    def kappa_inverse(self, kappa: Tensor) -> Tensor:
        return kappa


class PolynomialScheduler(ConvexScheduler):
    """Polynomial convex scheduler with α_t = t^n and σ_t = 1 - t^n."""

    def __init__(self, n: Union[float, int]) -> None:
        assert isinstance(
            n, (float, int)
        ), f"`n` must be a float or int. Got {type(n)=}."
        assert n > 0, f"`n` must be positive. Got {n=}." 
        self.n = float(n)

    def __call__(self, t: Tensor) -> SchedulerOutput:
        alpha_t = t**self.n
        sigma_t = 1 - alpha_t
        d_alpha_t = self.n * t ** (self.n - 1)
        d_sigma_t = -d_alpha_t
        return SchedulerOutput(
            alpha_t=alpha_t,
            sigma_t=sigma_t,
            d_alpha_t=d_alpha_t,
            d_sigma_t=d_sigma_t,
        )

    def kappa_inverse(self, kappa: Tensor) -> Tensor:
        return torch.pow(kappa, 1.0 / self.n)


class VPScheduler(Scheduler):
    """Variance-preserving scheduler."""

    def __init__(self, beta_min: float = 0.1, beta_max: float = 20.0) -> None:
        self.beta_min = beta_min
        self.beta_max = beta_max

    def __call__(self, t: Tensor) -> SchedulerOutput:
        b = self.beta_min
        B = self.beta_max
        T = 0.5 * (1 - t) ** 2 * (B - b) + (1 - t) * b
        dT = -(1 - t) * (B - b) - b

        alpha_t = torch.exp(-0.5 * T)
        sigma_t = torch.sqrt(1 - torch.exp(-T))
        d_alpha_t = -0.5 * dT * torch.exp(-0.5 * T)
        d_sigma_t = 0.5 * dT * torch.exp(-T) / torch.sqrt(1 - torch.exp(-T))

        return SchedulerOutput(
            alpha_t=alpha_t,
            sigma_t=sigma_t,
            d_alpha_t=d_alpha_t,
            d_sigma_t=d_sigma_t,
        )

    def snr_inverse(self, snr: Tensor) -> Tensor:
        T = -torch.log(snr**2 / (snr**2 + 1))
        b = self.beta_min
        B = self.beta_max
        t = 1 - ((-b + torch.sqrt(b**2 + 2 * (B - b) * T)) / (B - b))
        return t


class LinearVPScheduler(Scheduler):
    """Linear variance-preserving scheduler."""

    def __call__(self, t: Tensor) -> SchedulerOutput:
        alpha_t = t
        sigma_t = torch.sqrt(1 - t**2)
        d_alpha_t = torch.ones_like(t)
        d_sigma_t = -t / torch.sqrt(1 - t**2)
        return SchedulerOutput(
            alpha_t=alpha_t,
            sigma_t=sigma_t,
            d_alpha_t=d_alpha_t,
            d_sigma_t=d_sigma_t,
        )

    def snr_inverse(self, snr: Tensor) -> Tensor:
        return torch.sqrt(snr**2 / (1 + snr**2))


class CosineScheduler(Scheduler):
    """Cosine schedule for α_t = sin(pi/2 * t), σ_t = cos(pi/2 * t)."""

    def __call__(self, t: Tensor) -> SchedulerOutput:
        pi = torch.pi
        return SchedulerOutput(
            alpha_t=torch.sin(pi / 2 * t),
            sigma_t=torch.cos(pi / 2 * t),
            d_alpha_t=(pi / 2) * torch.cos(pi / 2 * t),
            d_sigma_t=-(pi / 2) * torch.sin(pi / 2 * t),
        )

    def snr_inverse(self, snr: Tensor) -> Tensor:
        return 2.0 * torch.atan(snr) / torch.pi
