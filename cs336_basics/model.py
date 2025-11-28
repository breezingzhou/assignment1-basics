# %%
import math
import torch
from torch import Tensor
from torch.nn import Module, Linear, init, Embedding, RMSNorm, SiLU, MultiheadAttention
from einops import rearrange, einsum, reduce, repeat
from jaxtyping import Bool, Float, Int

from cs336_basics.nn_utils import my_scaled_dot_product_attention

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

  def forward(self, input: Float[Tensor, "... d_in"]) -> Float[Tensor, "... d_out"]:
    return einsum(self.weight, input, "d_out d_in, ... d_in -> ... d_out")


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

  def forward(self, token_ids: Int[Tensor, "..."]) -> Float[Tensor, "... d_model"]:
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

  def forward(self, x: Float[Tensor, "... d_model"]) -> Float[Tensor, "... d_model"]:
    """
    RMSNorm(a_i) = ( a_i / RMS(a) ) * g_i
    RMS(a) = sqrt( (1/d) * sum(a_i^2) + eps )
    """
    in_dtype = x.dtype
    x = x.to(torch.float32)

    rms = torch.sqrt(reduce(x.pow(2), '... d_model -> ...', "mean") + self.eps)
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

  def forward(self, x: Float[Tensor, "... d_model"]) -> Float[Tensor, "... d_model"]:
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
    self.register_buffer("my_rot_matrix", self.rot_matrix, persistent=False)

  def create_rot_matrix(self, theta: float, d_k: int, max_seq_len: int, device: torch.device | None = None) -> Tensor:
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

  def forward(self, x: Float[Tensor, "... sequence_length d_k"], token_positions: Int[Tensor, "... sequence_length"]) -> Float[Tensor, " ... sequence_length d_k"]:
    return einsum(x, self.rot_matrix[token_positions], "... sequence_length d_k2 , ... sequence_length d_k1 d_k2   -> ... sequence_length d_k1")


class MyMultiHeadSelfAttention(Module):

  def __init__(self, d_model: int, num_heads: int, rope: MyRotaryPositionalEmbedding | None = None, device: torch.device | None = None, dtype: torch.dtype | None = None):
    """
      q_proj_weight: Float[Tensor, " d_k d_in"],
      k_proj_weight: Float[Tensor, " d_k d_in"],
      v_proj_weight: Float[Tensor, " d_v d_in"],
      o_proj_weight: Float[Tensor, " d_model d_v"],
      in_features: Float[Tensor, " ... sequence_length d_in"],
    """
    super().__init__()
    self.d_model = d_model
    self.num_heads = num_heads
    self.device = device
    self.rope = rope

    self.d_k = d_model // num_heads
    self.d_v = d_model // num_heads

    self.q_proj = MyLinear(self.d_k * num_heads, d_model, device=device, dtype=dtype)
    self.k_proj = MyLinear(self.d_k * num_heads, d_model, device=device, dtype=dtype)
    self.v_proj = MyLinear(self.d_v * num_heads, d_model, device=device, dtype=dtype)
    self.output_proj = MyLinear(d_model, self.d_v * num_heads, device=device, dtype=dtype)
    self.reset_parameters()

  def reset_parameters(self) -> None:
    self.q_proj.reset_parameters()
    self.k_proj.reset_parameters()
    self.v_proj.reset_parameters()
    self.output_proj.reset_parameters()

  def _create_look_ahead_mask(self, seq_len: int) -> Bool[Tensor, " ... sequence_length sequence_length"]:
    mask = torch.tril(torch.ones((seq_len, seq_len), device=self.device)).bool()
    return mask

  def forward(self, in_features: Float[Tensor, " ... sequence_length d_in"], token_positions: Int[Tensor, " ... sequence_length"] | None = None) -> Float[Tensor, " ... sequence_length d_out"]:
    """
      MultiHead(Q, K, V) = Concat(head_1,...,head_h)
        for head_i = Attention(Q_i,K_i,V_i)

      MultiHeadSelfAttention(x) = W_o * MultiHead(W_q * x, W_k * x, W_v * x)
    """
    seq_len = in_features.size(-2)
    Q = self.q_proj(in_features)  # type: Float[Tensor, " ... sequence_length d_k"]
    K = self.k_proj(in_features)  # type: Float[Tensor, " ... sequence_length d_k"]
    V = self.v_proj(in_features)  # type: Float[Tensor, " ... sequence_length d_v"]

    Q = rearrange(Q, '... s (h d_k) -> ... h s d_k', h=self.num_heads)
    K = rearrange(K, '... s (h d_k) -> ... h s d_k', h=self.num_heads)
    V = rearrange(V, '... s (h d_v) -> ... h s d_v', h=self.num_heads)

    if self.rope is not None:
      if token_positions is None:
        token_positions = torch.arange(seq_len, device=in_features.device)
      Q = self.rope(Q, token_positions)
      K = self.rope(K, token_positions)

    mask = self._create_look_ahead_mask(seq_len)
    output = my_scaled_dot_product_attention(Q, K, V, mask)
    output = rearrange(output, '... h s d_v -> ... s (h d_v)')

    return self.output_proj(output)


class MyTransformerBlock(Module):
  def __init__(self, d_model: int, num_heads: int, d_ff: int, max_seq_len: int, theta: float, device: torch.device | None = None, dtype: torch.dtype | None = None):
    super().__init__()
    self.rope = MyRotaryPositionalEmbedding(theta, d_model // num_heads, max_seq_len, device=device)
    self.attn = MyMultiHeadSelfAttention(
        d_model, num_heads, rope=self.rope, device=device, dtype=dtype)
    self.ln1 = MyRMSNorm(d_model, device=device, dtype=dtype)
    self.ffn = MySwiGLU(d_model, d_ff, device=device, dtype=dtype)
    self.ln2 = MyRMSNorm(d_model, device=device, dtype=dtype)

  def forward(self, x: Float[Tensor, "batch sequence_length d_model"], token_positions: Int[Tensor, "sequence_length"] | None = None) -> Float[Tensor, "batch sequence_length d_model"]:
    x = x + self.attn(self.ln1(x), token_positions=token_positions)
    x = x + self.ffn(self.ln2(x))
    return x


class MyTransformerLM(Module):

  def __init__(self, vocab_size: int, context_length: int, d_model: int, num_layers: int, num_heads: int, d_ff: int, rope_theta: float, device: torch.device | None = None, dtype: torch.dtype | None = None):
    super().__init__()
    self.token_embeddings = MyEmbedding(vocab_size, d_model, device=device, dtype=dtype)
    self.layers = torch.nn.ModuleList(
        [MyTransformerBlock(d_model, num_heads, d_ff, context_length, rope_theta,
                            device=device, dtype=dtype) for _ in range(num_layers)]
    )
    self.ln_final = MyRMSNorm(d_model, device=device, dtype=dtype)
    self.lm_head = MyLinear(d_model, vocab_size, device=device, dtype=dtype)

  def forward(self, in_indices: Int[Tensor, "batch_size sequence_length"]) -> Float[Tensor, "batch_size sequence_length vocab_size"]:
    x = self.token_embeddings(in_indices)
    for layer in self.layers:
      x = layer(x)
    x = self.ln_final(x)
    logits = self.lm_head(x)
    return logits
