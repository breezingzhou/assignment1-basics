
# %%
from abc import abstractmethod, ABC
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
  batch_size: int = 1

  def activation_count_transformer_block(self) -> int:
    QKV_projections = 3 * self.batch_size * self.context_length * self.d_model
    QK_matrix_multiply = self.batch_size * self.num_heads * self.context_length * self.context_length
    softmax = QK_matrix_multiply
    weighted_sum_of_values = self.batch_size * self.context_length * self.d_model
    output_projection = self.batch_size * self.context_length * self.d_model
    return QKV_projections + QK_matrix_multiply + softmax + weighted_sum_of_values + output_projection

  def activation_count_ln_final(self) -> int:
    return self.batch_size * self.context_length * self.d_model

  def activation_count_output_embedding(self) -> int:
    return self.batch_size * self.context_length * self.vocab_size

  def activation_count_cross_entropy_on_logits(self) -> int:
    return self.batch_size * self.context_length * self.vocab_size

  def activation_count_total(self) -> int:
    return self.activation_count_transformer_block() * self.num_layers + self.activation_count_ln_final() + self.activation_count_output_embedding() + self.activation_count_cross_entropy_on_logits()

  def params_count_token_embedding(self) -> int:
    return self.vocab_size * self.d_model

  def params_count_transformer_block(self) -> int:
    attn = 4 * self.d_model * self.d_model
    ln1 = self.d_model
    ffn = 3 * self.d_model * self.d_ff
    ln2 = self.d_model
    return attn + ln1 + ffn + ln2

  def params_count_ln_final(self) -> int:
    return self.d_model

  def params_count_lm_head(self) -> int:
    return self.d_model * self.vocab_size

  def params_count_total(self) -> int:
    return self.params_count_token_embedding() + self.params_count_transformer_block() * self.num_layers + self.params_count_ln_final() + self.params_count_lm_head()


class Base(ABC):
  @property
  @abstractmethod
  def param_count(self) -> int:
    pass

  @property
  @abstractmethod
  def flops(self) -> int:
    pass

  @property
  def activation_count(self) -> int:
    return 0


class Linear(Base):
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

  @property
  def activation_count(self) -> int:
    return math.prod(self.input_shape[:-1]) * self.out_features


class Embedding(Base):
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


class RMSNorm(Base):
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

  @property
  def activation_count(self) -> int:
    return math.prod(self.input_shape)


class SwiGLU(Base):
  def __init__(self, d_model: int, d_ff: int, input_shape: list[int]) -> None:
    self.d_model = d_model
    self.d_ff = d_ff
    self.input_shape = input_shape
    self.w1 = Linear(d_model, d_ff, input_shape)
    self.w2 = Linear(d_ff, d_model, input_shape[:-1] + [d_ff])
    self.w3 = Linear(d_model, d_ff, input_shape)

  @property
  def param_count(self) -> int:
    return self.w1.param_count + self.w2.param_count + self.w3.param_count

  @property
  def flops(self) -> int:
    return self.w1.flops + self.w2.flops + self.w3.flops

  @property
  def activation_count(self) -> int:
    w1_matrix_multipl = math.prod(self.input_shape) * self.d_ff // self.d_model
    silu = w1_matrix_multipl
    w2_matrix_multipl = math.prod(self.input_shape)
    return w1_matrix_multipl + silu + w2_matrix_multipl


class RotaryPositionalEmbedding(Base):
  def __init__(self) -> None:
    pass

  @property
  def param_count(self) -> int:
    return 0

  @property
  def flops(self) -> int:
    return 0


class MultiHeadSelfAttention(Base):
  def __init__(self, d_model: int, num_heads: int, input_shape: list[int]) -> None:
    self.d_model = d_model
    self.num_heads = num_heads
    self.input_shape = input_shape
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

  @property
  def activation_count(self) -> int:
    # input_shape = [batch_size, sequence_length, d_model]
    # ignore batch_size
    assert len(self.input_shape) == 2
    sequence_length, d_model = self.input_shape
    prod = math.prod(self.input_shape)
    QKV_projections = 3 * prod
    # QK_matrix_multiply [batch_size, num_heads, sequence_length, sequence_length]
    QK_matrix_multiply = self.num_heads * sequence_length * sequence_length
    softmax = QK_matrix_multiply
    weighted_sum_of_values = prod
    output_projection = prod
    return QKV_projections + QK_matrix_multiply + softmax + weighted_sum_of_values + output_projection


class TransformerBlock(Base):
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


class TransformerLM(Base):
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
        sum(layer.flops for layer in self.layers) +
        self.ln_final.flops +
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
lm = TransformerLM(config=config)

lm.display_param_count()
print("")
lm.display_flops()

# %%
params_count = config.params_count_total()
activations_count = config.activation_count_total()

total_memory = 80 * 1024 * 1024 * 1024   # 80 GB
# 4 bytes per float32
batch_size = (total_memory / 4 - params_count * 4) // activations_count

print("params_count:", params_count)
print("activations_count:", activations_count)
print("batch_size:", batch_size)
# %%
special_config = LMConfig(
    vocab_size=50257,
    context_length=1024,
    num_layers=48,
    d_model=1600,
    num_heads=25,
    d_ff=6400,
    batch_size=1024
)
lm = TransformerLM(config=special_config)

param_count = lm.param_count
forward_flops = lm.flops
backward_flops = forward_flops * 2

step_floaps = (forward_flops + backward_flops) * special_config.batch_size + param_count * 14

step_num = 400_000
total_flops = step_floaps * step_num
A100_peak_flops = 19.5 * 1e12  # 19.5teraFLOP
mfu = 0.5
total_seconds = total_flops / (A100_peak_flops * mfu)
total_days = total_seconds / (3600 * 24)
print(f"forward in step : {humanize.intword(forward_flops * special_config.batch_size)}")
print(f"backward in step : {humanize.intword(backward_flops * special_config.batch_size)}")
print(f"optimizer in step : {humanize.intword(param_count * 14)}")
print(f"total_days : {total_days}")
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
