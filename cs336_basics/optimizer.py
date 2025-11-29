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


from cs336_basics.nn_utils import my_softmax
# %%-


def my_cross_entropy(inputs: Float[Tensor, " batch_size vocab_size"], targets: Int[Tensor, " batch_size"]) -> Float[Tensor, ""]:
  max_ = inputs.max(dim=-1, keepdim=True).values
  logsumexp = torch.log(torch.exp(inputs - max_).sum(dim=-1, keepdim=True)) + max_
  log_probs = inputs - logsumexp
  batch_indices = torch.arange(inputs.size(0), device=inputs.device)
  target_log_probs = log_probs[batch_indices, targets]
  return -target_log_probs.mean()


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
      lr = group["lr"]  # Get the learning rate
      for p in group["params"]:
        if p.grad is None:
          continue
      state = self.state[p]  # Get state associated with p.
      t = state.get("t", 0)  # Get iteration number from the state, or initial value.
      grad = p.grad.data  # Get the gradient of loss with respect to p.
      p.data -= lr / math.sqrt(t + 1) * grad  # Update weight tensor in-place.
      state["t"] = t + 1  # Increment iteration number
    return loss
