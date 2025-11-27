# %%
import math
import torch
from torch.nn import Module, Linear, init, Embedding, RMSNorm, SiLU
from einops import rearrange, einsum, reduce, repeat

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


class MyRotaryPositionalEmbedding(Module):
  def __init__(self, theta: float, d_k: int, max_seq_len: int, device: torch.device | None = None):
    assert d_k % 2 == 0, "d_k must be even for RoPE"
    super().__init__()
    self.theta = theta
    self.d_k = d_k
    self.max_seq_len = max_seq_len

    self.rot_matrix = self.create_rot_matrix(theta, d_k, max_seq_len, device)
    self.register_buffer("rot_matrix", self.rot_matrix, persistent=False)

  def create_rot_matrix(self, theta: float, d_k: int, max_seq_len: int, device: torch.device | None = None) -> torch.Tensor:
    """
    Create the r tensor for RoPE.
    θ_ik = i / ( θ ** ((2k - 2) / d_k) )  for k = 1, 2, ..., d_k/2
    """
    d = d_k // 2
    i = torch.arange(0, max_seq_len, device=device)
    k = torch.arange(0, d, device=device)
    theta_k = 1 / (theta ** (2 * k / d_k))
    theta_ik = einsum(i, theta_k, "i, k -> i k")  # Shape: (max_seq_len, d_k/2)

    cos_val = torch.cos(theta_ik)
    sin_val = torch.sin(theta_ik)
    rot_submat = torch.zeros(*theta_ik.shape, 2, 2, device=device)
    rot_submat[..., 0, 0] = cos_val  # [d] → [d,1,1]
    rot_submat[..., 0, 1] = -sin_val
    rot_submat[..., 1, 0] = sin_val
    rot_submat[..., 1, 1] = cos_val

    rot_submat_reshape = rearrange(rot_submat, 'i k h w -> i h (k w)')
    rot_submat_repeat = repeat(rot_submat_reshape, 'i h d_k -> i (d h) d_k', d=d)

    eye = torch.eye(d, device=device)
    block_eye = repeat(eye, 'i j -> i h j w', h=2, w=2)
    block_eye = rearrange(block_eye, 'i h j w -> (i h) (j w)')  # 形状: (d_k, d_k)

    rot_matrix = einsum(rot_submat_repeat, block_eye, "... d_k1 d_k2, d_k1 d_k2 -> ... d_k1 d_k2")
    return rot_matrix

  def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
    """
    x: (Float[Tensor, "... sequence_length d_k"]): Input tensor to run RoPE on.
    token_positions: (Int[Tensor, "... sequence_length"]): Tensor of shape (batch_size, sequence_length) with the token positions
    """

    return einsum(x, self.rot_matrix[token_positions], "... sequence_length d_k2 , ... sequence_length d_k1 d_k2   -> ... sequence_length d_k1")
