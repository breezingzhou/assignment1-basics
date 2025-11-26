# %%
import math
import torch
from torch.nn import Module, Linear, init, Embedding
from einops import rearrange, einsum

# %%


class MyLinear(Module):
  in_features: int
  out_features: int

  def __init__(self, in_features: int, out_features: int, device: torch.device | None = None, dtype: torch.dtype | None = None):
    super().__init__()
    factory_kwargs = {"device": device, "dtype": dtype}
    self.in_features = in_features
    self.out_features = out_features
    self.weight = torch.nn.Parameter(
        data=torch.empty((out_features, in_features), **factory_kwargs)
    )
    self.reset_parameters()

  def reset_parameters(self) -> None:
    init.kaiming_uniform_(self.weight, a=math.sqrt(5))

  def forward(self, input: torch.Tensor) -> torch.Tensor:
    return einsum(self.weight, input, "out_features in_features, ... in_features -> ... out_features")


class MyEmbedding(Module):
  num_embeddings: int
  embedding_dim: int


  def __init__(self, num_embeddings: int, embedding_dim: int, device: torch.device | None = None, dtype: torch.dtype | None = None):
    super().__init__()
    factory_kwargs = {"device": device, "dtype": dtype}
    self.num_embeddings = num_embeddings
    self.embedding_dim = embedding_dim
    self.weight = torch.nn.Parameter(
        data=torch.empty((num_embeddings, embedding_dim), **factory_kwargs)
    )
    self.reset_parameters()

  def reset_parameters(self) -> None:
    init.normal_(self.weight, mean=0.0, std=1.0)

  def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
    return self.weight[token_ids]
