from abc import ABC
from typing import Any, Callable, Dict, Optional

import torch
from torch import Tensor, nn


class ModelWrapper(ABC, nn.Module):
    """Wrapper for a model, optimizer and loss function to simplify training and evaluation."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        loss_fn: Optional[Callable[[Tensor, Tensor], Tensor]] = None,
    ):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn

    def forward(self, x: Tensor, t: Optional[Tensor] = None, **extras: Any) -> Tensor:
        if t is None:
            return self.model(x=x, **extras)
        return self.model(x=x, t=t, **extras)

    def train_step(
        self,
        batch: Dict[str, Tensor],
        device: Optional[torch.device] = None,
        **extras: Any,
    ) -> Dict[str, Any]:
        if self.loss_fn is None:
            raise ValueError("ModelWrapper requires a loss_fn for training.")

        self.train()
        if device is not None:
            batch = {k: v.to(device) if isinstance(v, Tensor) else v for k, v in batch.items()}

        x = batch.get("x", batch.get("input"))
        y = batch.get("y", batch.get("target"))
        t = batch.get("t", None)

        if x is None or y is None:
            raise KeyError("Batch must contain keys 'x' and 'y' (or 'input' and 'target').")

        prediction = self.forward(x, t, **extras)
        loss = self.loss_fn(prediction, y)

        if self.optimizer is not None:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        return {"loss": loss.item(), "pred": prediction.detach()}

    def evaluate(
        self,
        dataloader: torch.utils.data.DataLoader,
        device: Optional[torch.device] = None,
        **extras: Any,
    ) -> Dict[str, float]:
        if self.loss_fn is None:
            raise ValueError("ModelWrapper requires a loss_fn for evaluation.")

        self.eval()
        total_loss = 0.0
        total_count = 0

        with torch.no_grad():
            for batch in dataloader:
                if device is not None:
                    batch = {k: v.to(device) if isinstance(v, Tensor) else v for k, v in batch.items()}

                x = batch.get("x", batch.get("input"))
                y = batch.get("y", batch.get("target"))
                t = batch.get("t", None)

                if x is None or y is None:
                    raise KeyError("Batch must contain keys 'x' and 'y' (or 'input' and 'target').")

                prediction = self.forward(x, t, **extras)
                loss = self.loss_fn(prediction, y)
                total_loss += loss.item() * x.shape[0]
                total_count += x.shape[0]

        average_loss = total_loss / max(total_count, 1)
        return {"loss": average_loss}
