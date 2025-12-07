# %%
from dataclasses import asdict, dataclass
from pathlib import Path

import toml

from tests.common import gpt2_bytes_to_unicode


# %%

WORKSPACE = Path(__file__).parent.parent
OUTPUT_DIR = WORKSPACE / "output"
DATA_DIR = WORKSPACE / "data"
EXPERIMENT_DIR = WORKSPACE / "experiment"
CHECKPOINT_FINAL_NAME = "checkpoint_final.pth"
CONFIG_DIR = EXPERIMENT_DIR / "configs"
BASE_CONFIG_PATH = CONFIG_DIR / "base.toml"
TMP_DIR = WORKSPACE / "tmp"


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

# TODO use pydantic


@dataclass
class ExperimentConfig:
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
  run_id: str | None = None
  train_start_epoch: int = 0

  @property
  def checkpoint_dir(self) -> Path:
    return EXPERIMENT_DIR / self.name / "checkpoints"

  @property
  def snapshot_dir(self) -> Path:
    return EXPERIMENT_DIR / self.name / "snapshots"

  @property
  def experiment_dir(self) -> Path:
    return EXPERIMENT_DIR / self.name

# %%


def load_config(path: Path) -> ExperimentConfig:
  with open(path, 'r') as f:
    config_dict = toml.load(f)
  module_params = ModelHyperParams(**config_dict['module_params'])
  optimizer_params = OptimizerHyperParams(**config_dict['optimizer_params'])
  schedule_params = SechduleParams(
      **config_dict['schedule_params']) if 'schedule_params' in config_dict and config_dict['schedule_params'] is not None else None
  clipping_params = ClippingParams(**config_dict['clipping_params'])
  config = ExperimentConfig(
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


def save_config(config: ExperimentConfig, path: Path):
  path.parent.mkdir(parents=True, exist_ok=True)
  with open(path, 'w') as f:
    toml.dump(asdict(config), f)


gpt2_byte_encoder = gpt2_bytes_to_unicode()
gpt2_byte_decoder = {v: k for k, v in gpt2_byte_encoder.items()}


def gpt2_encode(token: str) -> str:
  """
  将特殊符号编码成可展示的符号
  """
  byte_token = token.encode("utf-8")
  encoded_token = ''.join(gpt2_byte_encoder[b] for b in byte_token)
  return encoded_token


def gpt2_decode(encoded_token: str) -> str:
  """
  将可展示的符号解码成原始符号
  """
  byte_token = bytes(gpt2_byte_decoder[c] for c in encoded_token)
  decoded_token = byte_token.decode("utf-8", errors="replace")
  return decoded_token


__all__ = ["WORKSPACE", "OUTPUT_DIR", "DATA_DIR", "EXPERIMENT_DIR", "CHECKPOINT_FINAL_NAME",]
