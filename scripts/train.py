# %%

from dataclasses import asdict, dataclass
from cs336_basics.model import MyTransformerLM
from cs336_basics.optimizer import MyAdamW, MyCosineAnnealingLR
from cs336_basics.nn_utils import my_cross_entropy, my_get_batch, my_save_checkpoint, my_load_checkpoint, my_get_lr_cosine_schedule, my_gradient_clipping
from cs336_basics.tokenizer import BpeTokenizer
import torch
import numpy as np
import json
import os
import wandb
from datetime import datetime
from pathlib import Path
from common import OUTPUT_DIR, CHECKPOINTS_DIR, get_tokenizer_from_vocab_merges_path
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
  train_epochs: int
  batch_size: int
  project_name: str
  name: str

  max_l2_norm: float = 1e-2

  def wandb_config(self) -> dict:
    return asdict(self)
# %%# %%


def sumary_model(config: TrainConfig):
  from torchinfo import summary
  model, _, _ = create_from_config(config)
  train_data = load_data(OUTPUT_DIR / "TinyStoriesV2-GPT4-train.npy")
  x, y = my_get_batch(train_data, config.batch_size,
                      config.module_params.context_length, device='cpu')
  print(summary(model, input_data=x, verbose=0))


def load_data(data_path: Path) -> np.ndarray:
  # TODO verify dtype based on tokenizer vocab size
  data = np.memmap(data_path, dtype=np.int32, mode="r")
  return data


def prepare(config: TrainConfig):
  config.checkpoint_dir.mkdir(parents=True, exist_ok=True)


def create_from_config(config: TrainConfig) -> tuple[MyTransformerLM, MyAdamW, MyCosineAnnealingLR]:
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
  return model, optimizer, sechdule

# %%


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

  with wandb.init(project=config.project_name, config=asdict(config), name=config.name) as run:
    for epoch in range(config.train_epochs):
      print(f"Starting epoch {epoch + 1}/{config.train_epochs}")
      optimizer.zero_grad()
      # start_time = datetime.now()

      x, y = my_get_batch(train_data, config.batch_size,
                          config.module_params.context_length, device)
      # print(f"[{datetime.now() - start_time}] Data loaded ")
      logits = model(x)
      # print(f"[{datetime.now() - start_time}] Forward pass completed ")
      loss = my_cross_entropy(logits, y)
      # print(f"[{datetime.now() - start_time}] Loss computed ")
      # print(f"memory_allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
      # print(f"memory_reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
      loss.backward()
      # print(f"[{datetime.now() - start_time}] Backward pass completed ")
      my_gradient_clipping(model.parameters(), config.max_l2_norm)

      optimizer.step()
      # print(f"[{datetime.now() - start_time}] Optimizer step completed ")
      if sechdule:
        sechdule.step()

      run.log({"train_loss": loss.item()}, step=epoch)

      # Save checkpoint periodically
      if epoch % 1000 == 0:
        checkpoint_path = config.checkpoint_dir / f"checkpoint_iter_{epoch}.pth"
        my_save_checkpoint(model, optimizer, epoch, checkpoint_path)
        print(f"Checkpoint saved at iteration {epoch}")

  # Final checkpoint
  my_save_checkpoint(model, optimizer, config.train_epochs,
                     config.checkpoint_dir / "checkpoint_final.pth")
  print("Training completed")


# %%

_module_params = ModelHyperParams(
    vocab_size=10000,
    context_length=256,
    d_model=512,
    d_ff=1344,
    num_layers=4,
    num_heads=16,
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
_name = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
config = TrainConfig(
    module_params=_module_params,
    optimizer_params=_optimizer_params,
    schedule_params=_schedule_params,
    checkpoint_dir=CHECKPOINTS_DIR / _name,
    # num_epochs=40000, # 20倍参数量 / content_length / batch_size
    # num_epochs=66068, # 实际语料tokens数量 / content_length / batch_size
    train_epochs=40000,
    batch_size=32,
    project_name="TinyStoriesV2-GPT4",
    name=_name
)

# %%


def train():
  model, optimizer, sechdule = create_from_config(config)

  train_data = load_data(OUTPUT_DIR / "TinyStoriesV2-GPT4-train.npy")
  # val_data = load_data(data_path=OUTPUT_DIR / "TinyStoriesV2-GPT4-valid.npy")
  prepare(config)
  # torch.cuda.memory._record_memory_history()
  train_model(model, optimizer, None, train_data, train_data, config)
  # torch.cuda.memory._dump_snapshot("my_snapshot.pickle")

# %%


def validate_model(
    checkpoint_path: Path,
    val_data: np.ndarray,
    config: TrainConfig,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
):
  model, optimizer, _ = create_from_config(config)
  iteration = my_load_checkpoint(checkpoint_path, model, optimizer)
  print(f"Model loaded from checkpoint at iteration {iteration}")
  model.to(device)
  model.eval()

  eval_epochs = 100

  with torch.no_grad():
    losses = []
    for epoch in range(eval_epochs):
      x, y = my_get_batch(val_data, config.batch_size,
                          config.module_params.context_length, device, seed=epoch)
      logits = model(x)
      loss = my_cross_entropy(logits, y)
      losses.append(loss.item())

    print(f"Validation loss: {sum(losses) / len(losses):.4f}")


# %%
# val_data = load_data(data_path=OUTPUT_DIR / "TinyStoriesV2-GPT4-valid.npy")
# checkpoint_path = CHECKPOINTS_DIR / "20251204_183731" / "checkpoint_final.pth"
# validate_model(checkpoint_path, val_data, config)

# %%
def inference():
  pass


dataset_name = "TinyStoriesV2-GPT4"
device = 'cuda' if torch.cuda.is_available() else 'cpu'
checkpoint_path = CHECKPOINTS_DIR / "20251204_183731" / "checkpoint_final.pth"
inference_num = 100
tokenizer: BpeTokenizer = get_tokenizer_from_vocab_merges_path(
    OUTPUT_DIR / f"{dataset_name}_vocab.json",
    OUTPUT_DIR / f"{dataset_name}_merges.txt",
    special_tokens=["<|endoftext|>"]
)

model, optimizer, _ = create_from_config(config)
iteration = my_load_checkpoint(checkpoint_path, model, optimizer)

input_str = "Tom and Lily were playing with their toys in the living room. They liked to build towers and bridges with their blocks and cars."
input_tokens = tokenizer.encode(input_str)
input_tensor = torch.tensor(input_tokens, dtype=torch.int64, device=device)
model.to(device)
model.eval()
with torch.no_grad():
  for _ in range(inference_num):
    logits = model(input_tensor)
    predicted_tokens = torch.argmax(logits, dim=-1).squeeze()
    input_tensor = torch.cat((input_tensor, predicted_tokens[-1:]), dim=0)  # Shift input for next prediction

predicted_ids = input_tensor.tolist()
print(f"Predicted token IDs: {predicted_ids}")
predicted_text = tokenizer.decode(predicted_ids)
print(f"Input: {input_str}")
print(f"Predicted: {predicted_text}")

#%%
