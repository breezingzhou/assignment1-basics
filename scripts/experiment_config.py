from dataclasses import asdict
from pydantic import BaseModel, field_serializer, field_validator
from pydantic.dataclasses import dataclass
from pathlib import Path
import yaml
from common import EXPERIMENT_DIR
from cs336_basics.model import MyTransformerLM
from cs336_basics.optimizer import MyAdamW, MyCosineAnnealingLR


class ModelHyperParams(BaseModel):
  vocab_size: int
  context_length: int
  d_model: int
  num_layers: int
  num_heads: int
  d_ff: int
  rope_theta: float


class OptimizerHyperParams(BaseModel):
  learning_rate: float
  weight_decay: float
  betas: tuple[float, float]
  eps: float

  @field_validator("betas", mode="before")
  @classmethod
  def validate_betas(cls, v):
    if isinstance(v, list):
      return (v[0], v[1])
    return v

  @field_serializer("betas")
  def serialize_betas(self, v: tuple[float, float]):
    return list(v)


class SechduleParams(BaseModel):
  min_lr_coeff: float
  warmup_iters: int
  cosine_cycle_iters: int


class ClippingParams(BaseModel):
  max_l2_norm: float = 1e-2

# TODO use pydantic


class ExperimentConfig(BaseModel):
  model_params: ModelHyperParams
  optimizer_params: OptimizerHyperParams
  schedule_params: SechduleParams | None = None
  clipping_params: ClippingParams

  train_epochs: int
  eval_epochs: int
  batch_size: int

  dataset_name: str
  name: str
  save_every_n_epochs: int
  train_start_epoch: int = 0

  @classmethod
  def load_config(cls, path: Path) -> "ExperimentConfig":
    with open(path, 'r') as f:
      config_dict = yaml.safe_load(f)
    return ExperimentConfig(**config_dict)

  def save_config(self, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
      yaml.dump(self.model_dump(), f)

  def create_llm(self) -> tuple[MyTransformerLM, MyAdamW, MyCosineAnnealingLR | None]:
    model = MyTransformerLM(
        vocab_size=self.model_params.vocab_size,
        context_length=self.model_params.context_length,
        d_model=self.model_params.d_model,
        num_layers=self.model_params.num_layers,
        num_heads=self.model_params.num_heads,
        d_ff=self.model_params.d_ff,
        rope_theta=self.model_params.rope_theta
    )
    optimizer = MyAdamW(
        model.parameters(),
        lr=self.optimizer_params.learning_rate,
        weight_decay=self.optimizer_params.weight_decay,
        betas=self.optimizer_params.betas,
        eps=self.optimizer_params.eps
    )
    sechdule = MyCosineAnnealingLR(
        optimizer,
        warmup_iters=self.schedule_params.warmup_iters,
        cosine_cycle_iters=self.schedule_params.cosine_cycle_iters,
        min_lr=self.optimizer_params.learning_rate * self.schedule_params.min_lr_coeff
    ) if self.schedule_params else None
    return model, optimizer, sechdule


class EnvConfig(BaseModel):
  workspace: Path

  @property
  def output_dir(self) -> Path:
    return self.workspace / "output"

  @property
  def data_dir(self) -> Path:
    return self.workspace / "data"

  @property
  def experiments_dir(self) -> Path:
    return self.workspace / "experiments"


class Config(BaseModel):
  env: EnvConfig
  experiment: ExperimentConfig
  run_id: str | None = None

  @property
  def output_dir(self) -> Path:
    return self.env.output_dir

  @property
  def data_dir(self) -> Path:
    return self.env.data_dir

  @property
  def experiments_dir(self) -> Path:
    return self.env.experiments_dir

  @property
  def experiment_dir(self) -> Path:
    return self.env.workspace / self.experiment.name

  @property
  def checkpoint_dir(self) -> Path:
    return self.experiment_dir / "checkpoints"

  @property
  def snapshot_dir(self) -> Path:
    return self.experiment_dir / "snapshots"

  @property
  def log_file(self) -> Path:
    return self.experiment_dir / "logs.log"
