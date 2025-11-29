# %%
import math
import torch
from torch import Tensor
from torch.nn import Module, Linear, init, Embedding, RMSNorm, SiLU, MultiheadAttention
from torch.optim import Optimizer
from einops import rearrange, einsum, reduce, repeat
from jaxtyping import Bool, Float, Int
from torch.optim.optimizer import ParamsT
from collections.abc import Callable, Iterable


# %%


class MySGD(Optimizer):
  def __init__(self, params: ParamsT, lr: float = 1e-3):
    if lr < 0:
      raise ValueError(f"Invalid learning rate: {lr}")
    defaults = dict(lr=lr)
    super().__init__(params, defaults)

  def step(self, closure: Callable | None = None):
    loss = None if closure is None else closure()
    for group in self.param_groups:
      lr: float = group["lr"]  # Get the learning rate
      for p in group["params"]:
        if p.grad is None:
          continue
        state = self.state[p]  # Get state associated with p.
        t = state.get("t", 0)  # Get iteration number from the state, or initial value.
        grad = p.grad.data  # Get the gradient of loss with respect to p.
        p.data -= lr / math.sqrt(t + 1) * grad  # Update weight tensor in-place.
        state["t"] = t + 1  # Increment iteration number
    return loss


class MyAdamW(Optimizer):
  def __init__(self, params: ParamsT, lr: float = 1e-3, weight_decay: float = 0.01,
               betas: tuple[float, float] = (0.9, 0.999), eps: float = 1e-8,):
    defaults = dict(lr=lr, weight_decay=weight_decay, betas=betas, eps=eps)
    super().__init__(params, defaults)
    self.m = 0.0
    self.v = 0.0

  def step(self, closure: Callable | None = None):
    loss = None if closure is None else closure()
    for group in self.param_groups:
      lr: float = group["lr"]
      betas: tuple[float, float] = group["betas"]
      eps = group["eps"]
      weight_decay = group["weight_decay"]
      for p in group["params"]:
        p: Tensor
        if p.grad is None:
          continue
        grad = p.grad.data  # Get the gradient of loss with respect to p.
        self.m = betas[0] * self.m + (1 - betas[0]) * grad  # Update first moment estimate
        self.v = betas[1] * self.v + (1 - betas[1]) * grad ** 2  # Update second moment estimate

        state = self.state[p]  # Get state associated with p.
        t = state.get("t", 0) + 1  # Get iteration number from the state, or initial value.

        new_lr = lr * math.sqrt(1 - betas[1]**t) / (1 - betas[0] ** t)  # Adjust learning rate
        p.data -= new_lr * self.m / (torch.sqrt(self.v) + eps)  # Update weight tensor in-place
        p.data -= lr * weight_decay * p.data  # Apply weight decay
        state["t"] = t  # Increment iteration number
    return loss
