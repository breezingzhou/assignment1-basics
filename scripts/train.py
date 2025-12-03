# %%

from dataclasses import dataclass
from cs336_basics.model import MyTransformerLM
from cs336_basics.optimizer import MyAdamW, MyCosineAnnealingLR
from cs336_basics.nn_utils import my_cross_entropy, my_get_batch, my_save_checkpoint, my_load_checkpoint, my_get_lr_cosine_schedule, my_gradient_clipping
from cs336_basics.tokenizer import BpeTokenizer
import torch
import numpy as np
import json
import os
from datetime import datetime
from pathlib import Path
from common import OUTPUT_DIR, CHECKPOINTS_DIR
# %%


@dataclass
class ModelHyperParams:
  vocab_size: int
  context_length: int
  d_model: int
  num_layers: int
  num_heads: int
  d_ff: int
  rope_theta: float


@dataclass
class OptimizerHyperParams:
  learning_rate: float = 3e-4
  weight_decay: float = 1e-2
  betas: tuple[float, float] = (0.9, 0.999)
  eps: float = 1e-8


@dataclass
class SechduleParams:
  min_lr_coeff: float = 0.1
  warmup_iters: int = 1000
  cosine_cycle_iters: int = 10000


@dataclass
class TrainConfig:
  module_params: ModelHyperParams
  optimizer_params: OptimizerHyperParams
  schedule_params: SechduleParams

  checkpoint_dir: Path

  num_epochs: int
  batch_size: int

  max_l2_norm: float = 1e-2


# %%# %%


def load_data(data_path: Path) -> np.ndarray:
  # TODO verify dtype based on tokenizer vocab size
  data = np.memmap(data_path, dtype=np.int32, mode="r")
  return data


def prepare(config: TrainConfig):
  config.checkpoint_dir.mkdir(parents=True, exist_ok=True)


def train_model(
    model: MyTransformerLM,
    optimizer: MyAdamW,
    sechdule: MyCosineAnnealingLR | None,
    train_data: np.ndarray,
    val_data: np.ndarray,
    config: TrainConfig,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
):
  model.to(device)
  model.train()

  iteration = 0

  for epoch in range(config.num_epochs):
    print(f"Starting epoch {epoch + 1}/{config.num_epochs}")
    optimizer.zero_grad()

    x, y = my_get_batch(train_data, config.batch_size,
                        config.module_params.context_length, device)

    logits = model(x)
    loss = my_cross_entropy(logits, y)

    loss.backward()
    my_gradient_clipping(model.parameters(), config.max_l2_norm)

    optimizer.step()
    if sechdule:
      sechdule.step()

    iteration += 1
    if iteration % 100 == 0:
      log_str = f"Iteration {iteration}, Loss: {loss.item():.4f}"
      if sechdule:
        log_str += f", LR: {sechdule.get_last_lr()[0]:.6f}"
      print(log_str)

    # Save checkpoint periodically
    if iteration % 1000 == 0:
      checkpoint_path = config.checkpoint_dir / f"checkpoint_iter_{iteration}.pth"
      my_save_checkpoint(model, optimizer, iteration, checkpoint_path)
      print(f"Checkpoint saved at iteration {iteration}")

  # Final checkpoint
  my_save_checkpoint(model, optimizer, iteration, config.checkpoint_dir / "checkpoint_final.pth")
  print("Training completed")


# %%

_module_params = ModelHyperParams(
    vocab_size=10000,
    context_length=32,
    d_model=32,
    num_layers=2,
    num_heads=4,
    d_ff=128,
    rope_theta=10000.0
)
_optimizer_params = OptimizerHyperParams(
    learning_rate=3e-4,
    weight_decay=1e-2,
    betas=(0.9, 0.999),
    eps=1e-8
)
_schedule_params = SechduleParams(
    warmup_iters=1000,
    cosine_cycle_iters=10000,
)

config = TrainConfig(
    module_params=_module_params,
    optimizer_params=_optimizer_params,
    schedule_params=_schedule_params,
    checkpoint_dir=CHECKPOINTS_DIR / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    num_epochs=10,
    batch_size=64
)

# %%
model = MyTransformerLM(
    vocab_size=config.module_params.vocab_size,
    context_length=config.module_params.context_length,
    d_model=config.module_params.d_model,
    num_layers=config.module_params.num_layers,
    num_heads=config.module_params.num_heads,
    d_ff=config.module_params.d_ff,
    rope_theta=config.module_params.rope_theta
)
optimizer = MyAdamW(
    model.parameters(),
    lr=config.optimizer_params.learning_rate,
    weight_decay=config.optimizer_params.weight_decay,
    betas=config.optimizer_params.betas,
    eps=config.optimizer_params.eps
)
sechdule = MyCosineAnnealingLR(
    optimizer,
    warmup_iters=config.schedule_params.warmup_iters,
    cosine_cycle_iters=config.schedule_params.cosine_cycle_iters,
    min_lr=config.optimizer_params.learning_rate * config.schedule_params.min_lr_coeff
)
# %%
train_data = load_data(OUTPUT_DIR / "TinyStoriesV2-GPT4-valid.npy")
prepare(config)
train_model(model, optimizer, None, train_data, train_data, config)
