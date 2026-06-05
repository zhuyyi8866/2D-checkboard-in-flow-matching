from typing import Callable, Optional, Sequence, Tuple, Union

import torch
from torch import Tensor

from .solver import Solver

try:
    from torchdiffeq import odeint
except ImportError:  # pragma: no cover
    odeint = None


def gradient(output: Tensor, inputs: Tensor, create_graph: bool = False) -> Tensor:
    if not inputs.requires_grad:
        inputs = inputs.requires_grad_(True)
    grad = torch.autograd.grad(
        outputs=output,
        inputs=inputs,
        grad_outputs=torch.ones_like(output),
        create_graph=create_graph,
        retain_graph=True,
        only_inputs=True,
    )[0]
    return grad


def _euler_integrate(func, x_init: Tensor, time_grid: Tensor) -> Tensor:
    states = [x_init]
    for i in range(time_grid.shape[0] - 1):
        t0 = time_grid[i]
        t1 = time_grid[i + 1]
        dt = t1 - t0
        x0 = states[-1]
        x1 = x0 + dt * func(t0, x0)
        states.append(x1)
    return torch.stack(states)


def _implicit_euler_integrate(
    func, x_init: Tensor, time_grid: Tensor, max_iters: int = 5, tol: float = 1e-4
) -> Tensor:
    r"""Implicit Euler integration with fixed-point iteration.
    
    Solves: x_{n+1} = x_n + dt * f(t_{n+1}, x_{n+1})
    using fixed-point iteration: x^{k+1} = x_n + dt * f(t_{n+1}, x^k)
    
    Args:
        func: velocity function f(t, x)
        x_init: initial state
        time_grid: time points
        max_iters: max iterations for implicit equation solving
        tol: convergence tolerance
    
    Returns:
        states at all time points
    """
    states = [x_init]
    for i in range(time_grid.shape[0] - 1):
        t0 = time_grid[i]
        t1 = time_grid[i + 1]
        dt = t1 - t0
        x0 = states[-1]
        
        # Fixed-point iteration for implicit equation
        x_next = x0 + dt * func(t0, x0)  # Start with explicit Euler guess
        for _ in range(max_iters):
            x_prev = x_next
            x_next = x0 + dt * func(t1, x_prev)
            
            # Check convergence
            residual = torch.norm(x_next - x_prev) / (torch.norm(x_prev) + 1e-8)
            if residual < tol:
                break
        
        states.append(x_next)
    return torch.stack(states)


class ODESolver(Solver):
    """Solver for ordinary differential equations using a velocity model."""

    def __init__(self, velocity_model: Union[Callable, torch.nn.Module]):
        super().__init__()
        self.velocity_model = velocity_model

    def sample(
        self,
        x_init: Tensor,
        step_size: Optional[float] = None,
        method: str = "euler",
        atol: float = 1e-5,
        rtol: float = 1e-5,
        time_grid: Tensor = torch.tensor([0.0, 1.0]),
        return_intermediates: bool = False,
        enable_grad: bool = False,
        **model_extras,
    ) -> Union[Tensor, Sequence[Tensor]]:
        r"""Solve the ODE defined by the velocity model.

        Args:
            x_init (Tensor): initial state, shape [batch_size, ...].
            step_size (Optional[float]): fixed step size for fixed-step solvers.
            method (str): numerical method ('euler', 'implicit_euler', or others supported by torchdiffeq).
            atol (float): absolute tolerance for adaptive solvers.
            rtol (float): relative tolerance for adaptive solvers.
            time_grid (Tensor): time points for integration.
            return_intermediates (bool): whether to return all intermediate states.
            enable_grad (bool): whether to track gradients during integration.
            **model_extras: extra keyword arguments passed to the velocity model.

        Returns:
            Union[Tensor, Sequence[Tensor]]: final state or all intermediate states.
        """
        time_grid = time_grid.to(x_init.device)

        def ode_func(t, x):
            return self.velocity_model(x=x, t=t, **model_extras)

        # Handle implicit_euler specially (not supported by torchdiffeq)
        if method == "implicit_euler":
            with torch.set_grad_enabled(enable_grad):
                sol = _implicit_euler_integrate(
                    ode_func, x_init, time_grid, max_iters=5, tol=1e-4
                )
        elif odeint is not None:
            ode_opts = {"step_size": step_size} if step_size is not None else {}
            with torch.set_grad_enabled(enable_grad):
                sol = odeint(
                    ode_func,
                    x_init,
                    time_grid,
                    method=method,
                    options=ode_opts,
                    atol=atol,
                    rtol=rtol,
                )
        else:
            # Fallback to explicit Euler when torchdiffeq is not available
            with torch.set_grad_enabled(enable_grad):
                sol = _euler_integrate(ode_func, x_init, time_grid)

        if return_intermediates:
            return sol
        return sol[-1]

    def compute_likelihood(
        self,
        x_1: Tensor,
        log_p0: Callable[[Tensor], Tensor],
        step_size: Optional[float],
        method: str = "euler",
        atol: float = 1e-5,
        rtol: float = 1e-5,
        time_grid: Tensor = torch.tensor([1.0, 0.0]),
        return_intermediates: bool = False,
        exact_divergence: bool = False,
        enable_grad: bool = False,
        **model_extras,
    ) -> Union[Tuple[Tensor, Tensor], Tuple[Sequence[Tensor], Tensor]]:
        r"""Compute likelihood by solving the reverse-time ODE and tracking log-density."""
        assert (
            time_grid[0] == 1.0 and time_grid[-1] == 0.0
        ), f"Time grid must start at 1.0 and end at 0.0. Got {time_grid}."

        if not exact_divergence:
            z = (torch.randn_like(x_1) < 0).to(x_1.dtype) * 2.0 - 1.0

        def ode_func(x, t):
            return self.velocity_model(x=x, t=t, **model_extras)

        def dynamics_func(t, states):
            xt = states[0]
            with torch.set_grad_enabled(True):
                xt = xt.requires_grad_(True)
                ut = ode_func(xt, t)

                if exact_divergence:
                    div = 0
                    for i in range(ut.flatten(start_dim=1).shape[1]):
                        g = gradient(ut[:, i], xt, create_graph=True)[:, i]
                        if not enable_grad:
                            g = g.detach()
                        div = div + g
                else:
                    ut_dot_z = torch.einsum(
                        "ij,ij->i", ut.flatten(start_dim=1), z.flatten(start_dim=1)
                    )
                    grad_ut_dot_z = gradient(ut_dot_z, xt, create_graph=enable_grad)
                    div = torch.einsum(
                        "ij,ij->i",
                        grad_ut_dot_z.flatten(start_dim=1),
                        z.flatten(start_dim=1),
                    )

            if not enable_grad:
                ut = ut.detach()
                div = div.detach()
            return ut, div

        y_init = (x_1, torch.zeros(x_1.shape[0], device=x_1.device))
        ode_opts = {"step_size": step_size} if step_size is not None else {}

        if odeint is not None:
            with torch.set_grad_enabled(enable_grad):
                sol, log_det = odeint(
                    dynamics_func,
                    y_init,
                    time_grid,
                    method=method,
                    options=ode_opts,
                    atol=atol,
                    rtol=rtol,
                )
        else:
            raise RuntimeError(
                "compute_likelihood requires torchdiffeq for reverse integration. "
                "Install it with `pip install torchdiffeq`."
            )

        x_source = sol[-1]
        source_log_p = log_p0(x_source)

        if return_intermediates:
            return sol, source_log_p + log_det[-1]
        return sol[-1], source_log_p + log_det[-1]
