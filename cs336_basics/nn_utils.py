# %%
import torch
from torch import Tensor


# %%

def my_softmax(in_features: Tensor, dim: int) -> Tensor:
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
