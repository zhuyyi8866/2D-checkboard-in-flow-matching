from dataclasses import dataclass, field

from torch import Tensor


@dataclass
class PathSample:
    """Represents a sample from a conditional probability path.

    Attributes:
        x_1 (Tensor): target sample X_1.
        x_0 (Tensor): source sample X_0.
        t (Tensor): time samples t.
        x_t (Tensor): path sample X_t ~ p_t(X_t).
        dx_t (Tensor): conditional velocity dX_t.
    """

    x_1: Tensor = field(metadata={"help": "target samples X_1 (batch_size, ...)."})
    x_0: Tensor = field(metadata={"help": "source samples X_0 (batch_size, ...)."})
    t: Tensor = field(metadata={"help": "time samples t (batch_size)."})
    x_t: Tensor = field(metadata={"help": "samples x_t ~ p_t(X_t), shape (batch_size, ...)."})
    dx_t: Tensor = field(metadata={"help": "conditional target dX_t, shape (batch_size, ...)."})


@dataclass
class DiscretePathSample:
    """Represents a sample from a discrete probability path.

    Attributes:
        x_1 (Tensor): target sample X_1.
        x_0 (Tensor): source sample X_0.
        t (Tensor): time samples t.
        x_t (Tensor): path sample X_t ~ p_t(X_t).
    """

    x_1: Tensor = field(metadata={"help": "target samples X_1 (batch_size, ...)."})
    x_0: Tensor = field(metadata={"help": "source samples X_0 (batch_size, ...)."})
    t: Tensor = field(metadata={"help": "time samples t (batch_size)."})
    x_t: Tensor = field(metadata={"help": "samples X_t ~ p_t(X_t), shape (batch_size, ...)."})
