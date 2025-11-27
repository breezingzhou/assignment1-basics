# %%
import math
import torch
from torch.nn import Module, Linear, init, Embedding, RMSNorm, SiLU
from einops import rearrange, einsum, reduce

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


class MyRMSNorm(Module):
  d_model: int
  eps: float

  def __init__(self, d_model: int, eps: float = 1e-5, device: torch.device | None = None, dtype: torch.dtype | None = None):
    super().__init__()
    factory_kwargs = {"device": device, "dtype": dtype}
    self.d_model = d_model
    self.eps = eps
    self.weight = torch.nn.Parameter(
        data=torch.empty(self.d_model, **factory_kwargs)
    )
    self.reset_parameters()

  def reset_parameters(self) -> None:
    init.ones_(self.weight)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """
    RMSNorm(a_i) = ( a_i / RMS(a) ) * g_i
    RMS(a) = sqrt( (1/d) * sum(a_i^2) + eps )
    """
    in_dtype = x.dtype
    x = x.to(torch.float32)

    rms = torch.sqrt_(reduce(x.pow(2), '... d -> ...', "mean") + self.eps)
    rms = rearrange(rms, '... -> ... 1')
    x_normalized = x / rms
    result = einsum(self.weight, x_normalized, "d_model, ... d_model -> ... d_model")

    return result.to(in_dtype)


class MySwiGLU(Module):
  d_model: int
  d_ff: int

  def __init__(self, d_model: int, d_ff: int, device: torch.device | None = None, dtype: torch.dtype | None = None):
    """
    w1_weight (Float[Tensor, "d_ff d_model"])
    w2_weight (Float[Tensor, "d_model d_ff"])
    w3_weight (Float[Tensor, "d_ff d_model"])
    """
    super().__init__()
    self.d_model = d_model
    self.d_ff = d_ff
    self.w1 = MyLinear(d_model, d_ff, device=device, dtype=dtype)
    self.w2 = MyLinear(d_ff, d_model, device=device, dtype=dtype)
    self.w3 = MyLinear(d_model, d_ff, device=device, dtype=dtype)
    self.reset_parameters()

  def reset_parameters(self) -> None:
    self.w1.reset_parameters()
    self.w2.reset_parameters()
    self.w3.reset_parameters()

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """
    FFN(x) = SwiGLU(x,W1,W2,W3) = W2(SiLU(W1x)⊙W3x)
    """
    return self.w2(einsum(SiLU()(self.w1(x)), self.w3(x), "... d_ff, ... d_ff -> ... d_ff"))
