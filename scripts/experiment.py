# %%

from cs336_basics.model import MyTransformerLM
from cs336_basics.optimizer import MyAdamW, MyCosineAnnealingLR
from cs336_basics.nn_utils import my_cross_entropy, MyDataLoader, my_save_checkpoint, my_load_checkpoint, my_gradient_clipping, my_save_xy_snapshot
from cs336_basics.tokenizer import BpeTokenizer
from tests.test_tokenizer import get_tokenizer_from_vocab_merges_path

import hydra
from hydra.utils import to_absolute_path
import logging
import torch
import numpy as np
import wandb
from pathlib import Path

from common import CONFIG_DIR, CHECKPOINT_FINAL_NAME
from experiment_config import Config, EnvConfig, ExperimentConfig
# %%


def _setup_base_logger(config: Config):
  """配置基础日志格式"""
  log_format = "%(asctime)s - %(levelname)s - %(message)s"
  formatter = logging.Formatter(log_format)

  # 初始化logger
  logger = logging.getLogger()
  logger.setLevel(config.log_level)

  # 控制台Handler
  console_handler = logging.StreamHandler()
  console_handler.setFormatter(formatter)
  logger.addHandler(console_handler)

  # 文件Handler
  config.log_file.parent.mkdir(parents=True, exist_ok=True)
  file_handler = logging.FileHandler(config.log_file, mode="a", encoding="utf-8")
  file_handler.setFormatter(formatter)
  logger.addHandler(file_handler)





def load_data(data_path: Path) -> np.ndarray:
  # TODO verify dtype based on tokenizer vocab size
  data = np.memmap(data_path, dtype=np.uint32, mode="r")
  return data


def get_last_checkpoint(checkpoint_dir: Path) -> Path | None:
  if not checkpoint_dir.exists():
    return None
  checkpoints = list(checkpoint_dir.glob("checkpoint_iter_*.pt"))
  if not checkpoints:
    return None
  last_checkpoint = max(checkpoints, key=lambda x: int(x.stem.split("_")[-1]))
  return last_checkpoint


def train_prepare(config: Config, model: MyTransformerLM, optimizer: MyAdamW, sechdule: MyCosineAnnealingLR | None):
  # 创建相应文件夹
  config.experiment_dir.mkdir(parents=True, exist_ok=True)
  config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
  config.snapshot_dir.mkdir(parents=True, exist_ok=True)
  config.experiment.save_config(config.experiment_dir / "config.yaml")

  # 如果文件夹中有上次的检查点，则加载模型和优化器状态，更新sechdule
  # resume 必须与run_id 配合
  last_checkpoint = get_last_checkpoint(config.checkpoint_dir)
  assert (last_checkpoint is not None) == (
      config.run_id is not None), "run_id should be set only if training is resumed"

  if last_checkpoint:
    logging.info(f"Resuming from checkpoint: {last_checkpoint}")
    last_epoch = my_load_checkpoint(last_checkpoint, model, optimizer)
    config.experiment.train_start_epoch = last_epoch + 1
    if sechdule:
      sechdule.last_epoch = last_epoch
    logging.info(f"Resuming training from epoch {config.experiment.train_start_epoch}")


# %%


def train_model(
    model: MyTransformerLM,
    optimizer: MyAdamW,
    sechdule: MyCosineAnnealingLR | None,
    train_loader: MyDataLoader,
    val_loader: MyDataLoader | None,
    config: Config,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
):
  model.to(device)
  optimizer.to(device)
  model.train()
  resume = "must" if config.run_id else "allow"

  with wandb.init(project=config.experiment.dataset_name, config=config.experiment.model_dump(), name=config.experiment.name, dir=config.env.workspace, id=config.run_id, resume=resume) as run:
    debug_every_n_epochs = config.experiment.train_epochs // 100
    for epoch in range(config.experiment.train_start_epoch, config.experiment.train_epochs):
      if epoch % debug_every_n_epochs == 0:
        logging.debug(f"Starting epoch {epoch}/{config.experiment.train_epochs}")
      optimizer.zero_grad()
      x, y = train_loader[epoch]
      logits = model(x)
      loss = my_cross_entropy(logits, y)
      # print(f"memory_allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
      # print(f"memory_reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
      loss.backward()
      grad_norm, clip_coef = my_gradient_clipping(
          model.parameters(), config.experiment.clipping_params.max_l2_norm)

      optimizer.step()
      if sechdule:
        sechdule.step()

      run.log({
          "train_loss": loss.item(),
          "lr": sechdule.get_last_lr()[0] if sechdule else config.experiment.optimizer_params.learning_rate,
          "grad_norm": grad_norm,
          "grad_clip_coef": clip_coef,
      }, step=epoch)

      # Save checkpoint periodically
      if epoch % config.experiment.save_every_n_epochs == 0 and epoch != 0:
        checkpoint_path = config.checkpoint_dir / f"checkpoint_iter_{epoch}.pt"
        my_save_checkpoint(model, optimizer, epoch, checkpoint_path)
        logging.info(f"Checkpoint saved at iteration {epoch}")
      if config.experiment.snapshot_every_n_epochs > 0 and epoch % config.experiment.snapshot_every_n_epochs == 0 and epoch != 0:
        snapshot_path = config.snapshot_dir / f"snapshot_iter_{epoch}.pt"
        my_save_xy_snapshot(x, y, logits, snapshot_path)
        logging.info(f"Snapshot saved at iteration {epoch}")

  # Final checkpoint
  my_save_checkpoint(model, optimizer, config.experiment.train_epochs,
                     config.checkpoint_dir / CHECKPOINT_FINAL_NAME)
  logging.debug("Training completed")


# %%


def train(config: Config):
  model, optimizer, sechdule = config.experiment.create_llm()
  device = 'cuda' if torch.cuda.is_available() else 'cpu'
  dataset_name = config.experiment.dataset_name
  train_loader = MyDataLoader(
      load_data(config.output_dir / f"idxs.{dataset_name}_train.npy"),
      config.experiment.batch_size,
      config.experiment.model_params.context_length,
      device
  )
  val_loader = MyDataLoader(
      load_data(config.output_dir / f"idxs.{dataset_name}_valid.npy"),
      config.experiment.batch_size,
      config.experiment.model_params.context_length,
      device
  )
  train_prepare(config, model, optimizer, sechdule)

  # torch.cuda.memory._record_memory_history()
  train_model(model, optimizer, sechdule, train_loader, None, config)
  # torch.cuda.memory._dump_snapshot("my_snapshot.pickle")


# %%


def validate_model(
    model: MyTransformerLM,
    val_loader: MyDataLoader,
    config: Config,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
):

  model.to(device)
  model.eval()

  with torch.no_grad():
    losses = []
    for epoch in range(config.experiment.eval_epochs):
      x, y = val_loader[epoch]
      logits = model(x)
      loss = my_cross_entropy(logits, y)
      losses.append(loss.item())

  val_loss = sum(losses) / len(losses)
  logging.info(f"Validation loss: {val_loss:.4f}")
  return val_loss


# %%
def validate(config: Config, checkpoint_name: str | None = None):
  checkpoint_name = checkpoint_name or CHECKPOINT_FINAL_NAME
  checkpoint_path = config.checkpoint_dir / checkpoint_name
  device = 'cuda' if torch.cuda.is_available() else 'cpu'

  model, _, _ = config.experiment.create_llm()
  iteration = my_load_checkpoint(checkpoint_path, model)
  logging.info(f"Model loaded from checkpoint at iteration {iteration}")

  dataset_name = config.experiment.dataset_name
  val_loader = MyDataLoader(
      load_data(config.output_dir / f"idxs.{dataset_name}_valid.npy"),
      config.experiment.batch_size,
      config.experiment.model_params.context_length,
      device
  )

  val_loss = validate_model(model, val_loader, config, device)
  return val_loss

# %%


def inference(input_str: str, config: Config, checkpoint_name: str | None, inference_num: int = 100, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
  checkpoint_name = checkpoint_name or CHECKPOINT_FINAL_NAME
  checkpoint_path = config.checkpoint_dir / checkpoint_name

  dataset_name = config.experiment.dataset_name
  tokenizer: BpeTokenizer = get_tokenizer_from_vocab_merges_path(
      config.output_dir / f"{dataset_name}_vocab.json",
      config.output_dir / f"{dataset_name}_merges.txt",
      special_tokens=["<|endoftext|>"]
  )

  model, optimizer, _ = config.experiment.create_llm()
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
@hydra.main(version_base=None, config_path=str(CONFIG_DIR), config_name="config")
def main(cfg):
  workspace = Path(to_absolute_path(cfg.env.workspace))
  env = EnvConfig(workspace=workspace)
  kwargs = dict(cfg)
  kwargs['env'] = env
  config = Config(**kwargs)

  _setup_base_logger(config)
  train(config)


# %%
if __name__ == "__main__":
  main()  # 调用主函数
