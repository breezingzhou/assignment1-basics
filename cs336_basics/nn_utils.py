# %%
import math
import torch
from torch import Tensor
from jaxtyping import Bool, Float, Int
from einops import einsum, reduce, rearrange

# %%


def my_softmax(in_features: Float[Tensor, "..."], dim: int) -> Float[Tensor, "..."]:
  """
  Args:
    in_features (Float[Tensor, "..."]): Input features to softmax. Shape is arbitrary.
    dim (int): Dimension of the `in_features` to apply softmax to.

  Returns:
    Float[Tensor, "..."]: Tensor of with the same shape as `in_features` with the output of
    softmax normalizing the specified `dim`.
  """
  max_vals, _indices = torch.max(in_features, dim=dim, keepdim=True)
  exp_vals = torch.exp(in_features - max_vals)
  sum_exp_vals = torch.sum(exp_vals, dim=dim, keepdim=True)
  return exp_vals / sum_exp_vals


# %%
def my_scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... values d_v"],
    mask: Bool[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:
  """
   Attention(Q,K,V)=softmax(Q_T*K / sqrt(d_k)) V
  """
  d_k = Q.size(-1)
  pre_softmax = einsum(Q, K, "... queries d_k, ... keys d_k -> ... queries keys")

  if mask is not None:
    pre_softmax = pre_softmax.masked_fill(~mask, float('-inf'))
  gate = my_softmax(pre_softmax / torch.sqrt(torch.tensor(d_k, dtype=Q.dtype)), dim=-1)
  return einsum(gate, V, "... queries keys, ... keys d_v -> ... queries d_v")


def my_sigmoid(in_features: Float[Tensor, "..."]) -> Float[Tensor, "..."]:
  """
  sigmoid(x) = 1 / (1 + exp(-x))
  """
  return 1 / (1 + torch.exp(-in_features))


def my_silu(in_features: Float[Tensor, "..."]) -> Float[Tensor, "..."]:
  """
  SiLU(x) = x * sigmoid(x)
  """
  return in_features * my_sigmoid(in_features)

# %%


def my_cross_entropy(inputs: Float[Tensor, " batch_size vocab_size"], targets: Int[Tensor, " batch_size"]) -> Float[Tensor, ""]:
  max_ = inputs.max(dim=-1, keepdim=True).values
  logsumexp = torch.log(torch.exp(inputs - max_).sum(dim=-1, keepdim=True)) + max_
  log_probs = inputs - logsumexp
  batch_indices = torch.arange(inputs.size(0), device=inputs.device)
  target_log_probs = log_probs[batch_indices, targets]
  return -target_log_probs.mean()

# %%


def my_get_lr_cosine_schedule(it: int, max_learning_rate: float, min_learning_rate: float, warmup_iters: int, cosine_cycle_iters: int) -> float:
  if it < warmup_iters:
    return max_learning_rate * it / warmup_iters
  elif it > cosine_cycle_iters:
    return min_learning_rate
  else:
    cos_inner = math.pi * (it - warmup_iters) / (cosine_cycle_iters - warmup_iters)
    return min_learning_rate + 0.5 * (1 + math.cos(cos_inner)) * (max_learning_rate - min_learning_rate)
