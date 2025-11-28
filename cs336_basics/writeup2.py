
# %%
from dataclasses import dataclass


@dataclass
class LMConfig:
  vocab_size: int
  context_length: int
  num_layers: int
  d_model: int
  num_heads: int
  d_ff: int


class Linear():
  def __init__(self, in_features: int, out_features: int) -> None:
    self.in_features = in_features
    self.out_features = out_features

  @property
  def param_count(self) -> int:
    return self.in_features * self.out_features


class Embedding():
  def __init__(self, num_embeddings: int, embedding_dim: int) -> None:
    self.num_embeddings = num_embeddings
    self.embedding_dim = embedding_dim

  @property
  def param_count(self) -> int:
    return self.num_embeddings * self.embedding_dim


class MSNorm():
  def __init__(self, d_model: int) -> None:
    self.d_model = d_model

  @property
  def param_count(self) -> int:
    return self.d_model


class SwiGLU():
  def __init__(self, d_model: int, d_ff: int) -> None:
    self.w1 = Linear(d_model, d_ff)
    self.w2 = Linear(d_ff, d_model)
    self.w3 = Linear(d_model, d_ff)

  @property
  def param_count(self) -> int:
    return self.w1.param_count + self.w2.param_count + self.w3.param_count


class RotaryPositionalEmbedding():
  def __init__(self) -> None:
    pass

  @property
  def param_count(self) -> int:
    return 0


class MultiHeadSelfAttention():
  def __init__(self, d_model: int, num_heads: int) -> None:
    self.rope = RotaryPositionalEmbedding()
    self.q_proj = Linear(d_model, d_model)
    self.k_proj = Linear(d_model, d_model)
    self.v_proj = Linear(d_model, d_model)
    self.output_proj = Linear(d_model, d_model)

  @property
  def param_count(self) -> int:
    return (
        self.q_proj.param_count + self.k_proj.param_count + self.v_proj.param_count +
        self.output_proj.param_count + self.rope.param_count
    )


class TransformerBlock():
  def __init__(self, d_model: int, num_heads: int, d_ff: int,) -> None:
    self.attn = MultiHeadSelfAttention(d_model, num_heads)
    self.ln1 = MSNorm(d_model)
    self.ffn = SwiGLU(d_model, d_ff)
    self.ln2 = MSNorm(d_model)

  @property
  def param_count(self) -> int:
    return (
        self.attn.param_count + self.ln1.param_count + self.ffn.param_count + self.ln2.param_count
    )


class TransformerLM():
  def __init__(self, config: LMConfig) -> None:
    self.token_embedding = Embedding(config.vocab_size, config.d_model)
    self.layers = [
        TransformerBlock(config.d_model, config.num_heads, config.d_ff)
        for _ in range(config.num_layers)
    ]
    self.ln_final = MSNorm(config.d_model)
    self.lm_head = Linear(config.d_model, config.vocab_size)

  @property
  def param_count(self) -> int:
    return (
        self.token_embedding.param_count +
        sum(layer.param_count for layer in self.layers) + self.ln_final.param_count +
        self.lm_head.param_count
    )


# %%
config = LMConfig(
    vocab_size=50257,
    context_length=1024,
    num_layers=48,
    d_model=1600,
    num_heads=25,
    d_ff=6400,
)
lm = TransformerLM(config)
lm.param_count
