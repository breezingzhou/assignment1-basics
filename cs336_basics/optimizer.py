# %%
import math
import torch
from torch import Tensor
from torch.nn import Module, Linear, init, Embedding, RMSNorm, SiLU, MultiheadAttention
from einops import rearrange, einsum, reduce, repeat
from jaxtyping import Bool, Float, Int

from cs336_basics.nn_utils import my_softmax
# %%-


def my_cross_entropy(inputs: Float[Tensor, " batch_size vocab_size"], targets: Int[Tensor, " batch_size"]) -> Float[Tensor, ""]:
  max_ = inputs.max(dim=-1, keepdim=True).values
  logsumexp = torch.log(torch.exp(inputs - max_).sum(dim=-1, keepdim=True)) + max_
  log_probs = inputs - logsumexp
  batch_indices = torch.arange(inputs.size(0), device=inputs.device)
  target_log_probs = log_probs[batch_indices, targets]
  return -target_log_probs.mean()
