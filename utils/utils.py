from typing import Optional

import torch
from torch import Tensor


def expand_dims(tensor: Tensor, target_ndim: int, how: str = "suffix") -> Tensor:
    assert how in {"prefix", "suffix"}, "how must be 'prefix' or 'suffix'."
    dim_diff = target_ndim - tensor.ndim
    if dim_diff <= 0:
        return tensor

    for _ in range(dim_diff):
        tensor = tensor.unsqueeze(0) if how == "prefix" else tensor.unsqueeze(-1)
    return tensor


def expand_tensor_like(input_tensor: Tensor, expand_to: Tensor) -> Tensor:
    assert input_tensor.ndim == 1, "Input tensor must be a 1D vector."
    assert (
        input_tensor.shape[0] == expand_to.shape[0]
    ), f"Batch size mismatch: {input_tensor.shape} vs {expand_to.shape}."

    dim_diff = expand_to.ndim - input_tensor.ndim
    expanded = input_tensor.reshape(-1, *([1] * dim_diff))
    return expanded.expand_as(expand_to)


def gradient(
    output: Tensor,
    x: Tensor,
    grad_outputs: Optional[Tensor] = None,
    create_graph: bool = False,
) -> Tensor:
    if grad_outputs is None:
        grad_outputs = torch.ones_like(output).detach()

    grad = torch.autograd.grad(
        outputs=output,
        inputs=x,
        grad_outputs=grad_outputs,
        create_graph=create_graph,
        retain_graph=True,
        only_inputs=True,
    )[0]
    return grad
