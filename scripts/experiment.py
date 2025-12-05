# %%

from dataclasses import asdict, dataclass
from cs336_basics.model import MyTransformerLM
from cs336_basics.optimizer import MyAdamW, MyCosineAnnealingLR
from cs336_basics.nn_utils import my_cross_entropy, my_get_batch, my_save_checkpoint, my_load_checkpoint, my_get_lr_cosine_schedule, my_gradient_clipping
from cs336_basics.tokenizer import BpeTokenizer
from tests.test_tokenizer import get_tokenizer_from_vocab_merges_path
import torch
import numpy as np
import wandb
from datetime import datetime
from pathlib import Path
from common import OUTPUT_DIR, EXPERIMENT_DIR, CHECKPOINT_FINAL_NAME
import toml
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
  learning_rate: float
  weight_decay: float
  betas: tuple[float, float]
  eps: float

  def __post_init__(self):
    if isinstance(self.betas, list):
      self.betas = (self.betas[0], self.betas[1])


@dataclass
class SechduleParams:
  min_lr_coeff: float
  warmup_iters: int
  cosine_cycle_iters: int


@dataclass
class ClippingParams:
  max_l2_norm: float = 1e-2


@dataclass
class TrainConfig:
  module_params: ModelHyperParams
  optimizer_params: OptimizerHyperParams
  schedule_params: SechduleParams | None
  clipping_params: ClippingParams

  train_epochs: int
  eval_epochs: int
  batch_size: int

  dataset_name: str
  name: str
  save_every_n_epochs: int

  @property
  def checkpoint_dir(self) -> Path:
    return EXPERIMENT_DIR / self.name / "checkpoints"

  @property
  def experiment_dir(self) -> Path:
    return EXPERIMENT_DIR / self.name

# %%


def load_config(path: Path) -> TrainConfig:
  with open(path, 'r') as f:
    config_dict = toml.load(f)
  module_params = ModelHyperParams(**config_dict['module_params'])
  optimizer_params = OptimizerHyperParams(**config_dict['optimizer_params'])
  schedule_params = SechduleParams(
      **config_dict['schedule_params']) if 'schedule_params' in config_dict and config_dict['schedule_params'] is not None else None
  clipping_params = ClippingParams(**config_dict['clipping_params'])
  config = TrainConfig(
      module_params=module_params,
      optimizer_params=optimizer_params,
      schedule_params=schedule_params,
      clipping_params=clipping_params,
      train_epochs=config_dict['train_epochs'],
      eval_epochs=config_dict['eval_epochs'],
      batch_size=config_dict['batch_size'],
      dataset_name=config_dict['dataset_name'],
      name=config_dict['name'],
      save_every_n_epochs=config_dict['save_every_n_epochs'],
  )
  return config


def save_config(config: TrainConfig, path: Path):
  path.parent.mkdir(parents=True, exist_ok=True)
  with open(path, 'w') as f:
    toml.dump(asdict(config), f)


# %%


def summary_model(config: TrainConfig):
  from torchinfo import summary
  model, _, _ = create_from_config(config)
  train_data = load_data(OUTPUT_DIR / f"{config.dataset_name}-train.npy")
  x, y = my_get_batch(train_data, config.batch_size,
                      config.module_params.context_length, device='cpu')
  print(summary(model, input_data=x, verbose=0))


def load_data(data_path: Path) -> np.ndarray:
  # TODO verify dtype based on tokenizer vocab size
  data = np.memmap(data_path, dtype=np.int32, mode="r")
  return data


def train_prepare(config: TrainConfig):
  config.experiment_dir.mkdir(parents=True, exist_ok=True)
  config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
  save_config(config, config.experiment_dir / "config.toml")


def create_from_config(config: TrainConfig) -> tuple[MyTransformerLM, MyAdamW, MyCosineAnnealingLR | None]:
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
  ) if config.schedule_params else None
  return model, optimizer, sechdule

# %%


def train_model(
    model: MyTransformerLM,
    optimizer: MyAdamW,
    sechdule: MyCosineAnnealingLR | None,
    train_data: np.ndarray,
    val_data: np.ndarray | None,
    config: TrainConfig,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
):
  model.to(device)
  model.train()

  with wandb.init(project=config.dataset_name, config=asdict(config), name=config.name) as run:
    for epoch in range(config.train_epochs):
      print(f"Starting epoch {epoch + 1}/{config.train_epochs}")
      optimizer.zero_grad()

      x, y = my_get_batch(train_data, config.batch_size,
                          config.module_params.context_length, device)
      logits = model(x)
      loss = my_cross_entropy(logits, y)
      # print(f"memory_allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
      # print(f"memory_reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
      loss.backward()
      my_gradient_clipping(model.parameters(), config.clipping_params.max_l2_norm)

      optimizer.step()
      if sechdule:
        sechdule.step()

      run.log({
          "train_loss": loss.item(),
          "lr": sechdule.get_last_lr()[0] if sechdule else config.optimizer_params.learning_rate
      }, step=epoch)

      # Save checkpoint periodically
      if epoch % config.save_every_n_epochs == 0 and epoch != 0:
        checkpoint_path = config.checkpoint_dir / f"checkpoint_iter_{epoch}.pth"
        my_save_checkpoint(model, optimizer, epoch, checkpoint_path)
        print(f"Checkpoint saved at iteration {epoch}")

  # Final checkpoint
  my_save_checkpoint(model, optimizer, config.train_epochs,
                     config.checkpoint_dir / CHECKPOINT_FINAL_NAME)
  print("Training completed")


# %%


def train(config: TrainConfig):
  model, optimizer, sechdule = create_from_config(config)
  dataset_name = config.dataset_name
  train_data = load_data(OUTPUT_DIR / f"{dataset_name}-train.npy")
  # val_data = load_data(data_path=OUTPUT_DIR / f"{dataset_name}-valid.npy")
  train_prepare(config)
  # torch.cuda.memory._record_memory_history()
  train_model(model, optimizer, sechdule, train_data, None, config)
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

  with torch.no_grad():
    losses = []
    for epoch in range(config.eval_epochs):
      x, y = my_get_batch(val_data, config.batch_size,
                          config.module_params.context_length, device, seed=epoch)
      logits = model(x)
      loss = my_cross_entropy(logits, y)
      losses.append(loss.item())

    print(f"Validation loss: {sum(losses) / len(losses):.4f}")


# %%
def validate(config: TrainConfig, checkpoint_name: str | None = None):
  dataset_name = config.dataset_name
  val_data = load_data(data_path=OUTPUT_DIR / f"{dataset_name}-valid.npy")
  checkpoint_name = checkpoint_name or CHECKPOINT_FINAL_NAME
  checkpoint_path = config.checkpoint_dir / checkpoint_name
  validate_model(checkpoint_path, val_data, config)

# %%


def inference(input_str: str, config: TrainConfig, checkpoint_name: str | None, inference_num: int = 100, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
  checkpoint_name = checkpoint_name or CHECKPOINT_FINAL_NAME
  checkpoint_path = config.checkpoint_dir / checkpoint_name

  dataset_name = config.dataset_name
  tokenizer: BpeTokenizer = get_tokenizer_from_vocab_merges_path(
      OUTPUT_DIR / f"{dataset_name}_vocab.json",
      OUTPUT_DIR / f"{dataset_name}_merges.txt",
      special_tokens=["<|endoftext|>"]
  )

  model, optimizer, _ = create_from_config(config)
  iteration = my_load_checkpoint(checkpoint_path, model, optimizer)
  model.to(device)
  model.eval()

  input_tokens = tokenizer.encode(input_str)
  input_tensor = torch.tensor(input_tokens, dtype=torch.int64, device=device)
  with torch.no_grad():
    for _ in range(inference_num):
      logits = model(input_tensor)
      predicted_tokens = torch.argmax(logits, dim=-1).squeeze()
      input_tensor = torch.cat((input_tensor, predicted_tokens[-1:]), dim=0)
  predicted_ids = input_tensor.tolist()
  predicted_text = tokenizer.decode(predicted_ids)
  print(f"Input: {input_str}")
  print(f"Predicted token IDs: {predicted_ids}")
  print(f"Predicted text: {predicted_text}")

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
    min_lr_coeff=0.1,
    warmup_iters=500,
    cosine_cycle_iters=40000,
)
_clipping_params = ClippingParams(
    max_l2_norm=1e-2
)
_name = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
config = TrainConfig(
    module_params=_module_params,
    optimizer_params=_optimizer_params,
    schedule_params=_schedule_params,
    clipping_params=_clipping_params,
    # num_epochs=40000, # 20倍参数量 / content_length / batch_size
    # num_epochs=66068, # 实际语料tokens数量 / content_length / batch_size
    train_epochs=40000,
    batch_size=32,
    eval_epochs=1000,
    dataset_name="TinyStoriesV2-GPT4",
    name=_name,
    save_every_n_epochs=1000,
)

# %%
summary_model(config)
# %%
# train(config)

# %%
# input_str = "Tom and Lily were playing with their toys in the living room. They liked to build towers and bridges with their blocks and cars."
# inference(input_str, config, checkpoint_name=None, inference_num=200)
