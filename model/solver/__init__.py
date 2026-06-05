"""统一管理solver相关的类和函数，包括Solver和ODESolver等。"""

from .ode_solver import ODESolver
from .solver import Solver

__all__ = ["Solver", "ODESolver"]
