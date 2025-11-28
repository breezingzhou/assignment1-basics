
# %%
from dataclasses import dataclass
import math
import humanize


@dataclass
class LMConfig:
  vocab_size: int
  context_length: int
  num_layers: int
  d_model: int
  num_heads: int
  d_ff: int


class Linear():
  def __init__(self, in_features: int, out_features: int, input_shape: list[int]) -> None:
    self.in_features = in_features
    self.out_features = out_features
    self.input_shape = input_shape

  @property
  def param_count(self) -> int:
    return self.in_features * self.out_features

  @property
  def flops(self) -> int:
    return 2 * math.prod(self.input_shape) * self.out_features


class Embedding():
  def __init__(self, num_embeddings: int, embedding_dim: int, input_shape: list[int]) -> None:
    self.num_embeddings = num_embeddings
    self.embedding_dim = embedding_dim
    self.input_shape = input_shape

  @property
  def param_count(self) -> int:
    return self.num_embeddings * self.embedding_dim

  @property
  def flops(self) -> int:
    return 0


class RMSNorm():
  def __init__(self, d_model: int, input_shape: list[int]) -> None:
    self.d_model = d_model
    self.input_shape = input_shape

  @property
  def param_count(self) -> int:
    return self.d_model

  @property
  def flops(self) -> int:
    prod = math.prod(self.input_shape)
    return 3 * prod


class SwiGLU():
  def __init__(self, d_model: int, d_ff: int, input_shape: list[int]) -> None:
    self.w1 = Linear(d_model, d_ff, input_shape)
    self.w2 = Linear(d_ff, d_model, input_shape[:-1] + [d_ff])
    self.w3 = Linear(d_model, d_ff, input_shape)

  @property
  def param_count(self) -> int:
    return self.w1.param_count + self.w2.param_count + self.w3.param_count

  @property
  def flops(self) -> int:
    return self.w1.flops + self.w2.flops + self.w3.flops


class RotaryPositionalEmbedding():
  def __init__(self) -> None:
    pass

  @property
  def param_count(self) -> int:
    return 0

  @property
  def flops(self) -> int:
    return 0


class MultiHeadSelfAttention():
  def __init__(self, d_model: int, num_heads: int, input_shape: list[int]) -> None:
    self.d_model = d_model
    self.rope = RotaryPositionalEmbedding()
    self.q_proj = Linear(d_model, d_model, input_shape)
    self.k_proj = Linear(d_model, d_model, input_shape)
    self.v_proj = Linear(d_model, d_model, input_shape)
    self.output_proj = Linear(d_model, d_model, input_shape)

  @property
  def param_count(self) -> int:
    return (
        self.q_proj.param_count + self.k_proj.param_count + self.v_proj.param_count +
        self.output_proj.param_count + self.rope.param_count
    )

  @property
  def flops(self) -> int:
    scaled_dot_product_attention = 2 * math.prod(self.q_proj.input_shape[:-1]) * self.d_model
    return scaled_dot_product_attention + self.q_proj.flops + self.k_proj.flops + self.v_proj.flops + self.output_proj.flops + self.rope.flops * 2


class TransformerBlock():
  def __init__(self, d_model: int, num_heads: int, d_ff: int, input_shape: list[int]) -> None:
    # input_shape: [sequence_length, d_model]
    self.attn = MultiHeadSelfAttention(d_model, num_heads, input_shape)
    self.ln1 = RMSNorm(d_model, input_shape)
    self.ffn = SwiGLU(d_model, d_ff, input_shape)
    self.ln2 = RMSNorm(d_model, input_shape)

  @property
  def param_count(self) -> int:
    return (
        self.attn.param_count + self.ln1.param_count + self.ffn.param_count + self.ln2.param_count
    )

  @property
  def flops(self) -> int:
    return self.attn.flops + self.ln1.flops + self.ffn.flops + self.ln2.flops


class TransformerLM():
  def __init__(self, config: LMConfig) -> None:
    self.token_embedding = Embedding(config.vocab_size, config.d_model,
                                     input_shape=[config.context_length])
    self.layers = [
        TransformerBlock(config.d_model, config.num_heads, config.d_ff,
                         input_shape=[config.context_length, config.d_model])
        for _ in range(config.num_layers)
    ]
    self.ln_final = RMSNorm(config.d_model, input_shape=[config.context_length, config.d_model])
    self.lm_head = Linear(config.d_model, config.vocab_size, input_shape=[
                          config.context_length, config.d_model])

  @property
  def param_count(self) -> int:
    return (
        self.token_embedding.param_count +
        sum(layer.param_count for layer in self.layers) + self.ln_final.param_count +
        self.lm_head.param_count
    )

  def display_param_count(self) -> None:
    layer_param_count = self.layers[0].param_count
    print(f"| Component | Parameters |")
    print(f"|----------|-------|")
    print(f"| Token Embedding | {self.token_embedding.param_count} |")
    print(f"| Transformer Block | {layer_param_count} |")
    print(f"| Transformer Block Total | {layer_param_count * len(self.layers)} |")
    print(f"| Final LayerNorm | {self.ln_final.param_count} |")
    print(f"| LM Head | {self.lm_head.param_count} |")
    print(f"| Total Parameters | {self.param_count} |")

  @property
  def flops(self) -> int:
    return (
        self.token_embedding.flops +
        sum(layer.flops for layer in self.layers) + self.ln_final.flops +
        self.lm_head.flops
    )

  def display_flops(self) -> None:
    token_embedding = self.token_embedding.flops
    layer_flops = self.layers[0].flops
    layers_flops = layer_flops * len(self.layers)
    ln_final = self.ln_final.flops
    lm_head = self.lm_head.flops
    total_flops = token_embedding + layers_flops + ln_final + lm_head

    print(f"| Component | FLOPs | rate |")
    print(f"|----------|-------|")
    print(
        f"| Token Embedding | {humanize.intword(token_embedding)} | {token_embedding / total_flops} |")
    print(f"| Transformer Block | {humanize.intword(layer_flops)} | {layer_flops / total_flops} |")
    print(
        f"| Transformer Block Total | {humanize.intword(layers_flops)} | {layers_flops / total_flops} |")
    print(f"| Final LayerNorm | {humanize.intword(ln_final)} | {ln_final / total_flops} |")
    print(f"| LM Head | {humanize.intword(lm_head)} | {lm_head / total_flops} |")
    print(f"| Total FLOPs | {humanize.intword(total_flops)} | 1 |")


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

lm.display_param_count()
print("")
lm.display_flops()
# %%
small_config = LMConfig(
    vocab_size=50257,
    context_length=1024,
    num_layers=12,
    d_model=768,
    num_heads=12,
    d_ff=6400,
)
medium_config = LMConfig(
    vocab_size=50257,
    context_length=1024,
    num_layers=24,
    d_model=1024,
    num_heads=16,
    d_ff=6400,
)
large_config = LMConfig(
    vocab_size=50257,
    context_length=1024,
    num_layers=36,
    d_model=1280,
    num_heads=20,
    d_ff=6400,
)
lm_small = TransformerLM(small_config)
lm_medium = TransformerLM(medium_config)
lm_large = TransformerLM(large_config)
print("Small Model:")
lm_small.display_flops()
print("")
print("Medium Model:")
lm_medium.display_flops()
print("")
print("Large Model:")
lm_large.display_flops()
# %%
context_length_config = LMConfig(
    vocab_size=50257,
    context_length=16384,
    num_layers=48,
    d_model=1600,
    num_heads=25,
    d_ff=6400,
)

lm_context_length = TransformerLM(context_length_config)

print("Original Model:")
lm.display_flops()
print("")
print("Context Length Model:")
lm_context_length.display_flops()
