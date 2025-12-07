# %%

from dataclasses import asdict
from cs336_basics.model import MyTransformerLM
from cs336_basics.optimizer import MyAdamW, MyCosineAnnealingLR
from cs336_basics.nn_utils import my_cross_entropy, my_get_batch, my_save_checkpoint, my_load_checkpoint, my_gradient_clipping, my_save_xy_snapshot
from cs336_basics.tokenizer import BpeTokenizer
from tests.test_tokenizer import get_tokenizer_from_vocab_merges_path
import torch
import numpy as np
import wandb
import sys
from datetime import datetime
from pathlib import Path
from common import CONFIG_DIR, OUTPUT_DIR, WORKSPACE, CHECKPOINT_FINAL_NAME, ClippingParams, ModelHyperParams, OptimizerHyperParams, SechduleParams, ExperimentConfig, save_config, load_config
# %%


def summary_model(config: ExperimentConfig):
  from torchinfo import summary
  model, _, _ = create_from_config(config)
  train_data = load_data(OUTPUT_DIR / f"{config.dataset_name}_train.npy")
  x, y = my_get_batch(train_data, config.batch_size,
                      config.module_params.context_length, device='cpu')
  print(summary(model, input_data=x, verbose=0))


def load_data(data_path: Path) -> np.ndarray:
  # TODO verify dtype based on tokenizer vocab size
  data = np.memmap(data_path, dtype=np.int32, mode="r")
  return data


def get_last_checkpoint(checkpoint_dir: Path) -> Path | None:
  if not checkpoint_dir.exists():
    return None
  checkpoints = list(checkpoint_dir.glob("checkpoint_iter_*.pt"))
  if not checkpoints:
    return None
  last_checkpoint = max(checkpoints, key=lambda x: int(x.stem.split("_")[-1]))
  return last_checkpoint


def train_prepare(config: ExperimentConfig, model: MyTransformerLM, optimizer: MyAdamW, sechdule: MyCosineAnnealingLR | None):
  # 创建相应文件夹
  config.experiment_dir.mkdir(parents=True, exist_ok=True)
  config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
  config.snapshot_dir.mkdir(parents=True, exist_ok=True)
  save_config(config, config.experiment_dir / "config.toml")

  # 如果文件夹中有上次的检查点，则加载模型和优化器状态，更新sechdule
  # resume 必须与run_id 配合
  last_checkpoint = get_last_checkpoint(config.checkpoint_dir)
  assert (last_checkpoint is not None) == (
      config.run_id is not None), "run_id should be set only if training is resumed"

  if last_checkpoint:
    print(f"Resuming from checkpoint: {last_checkpoint}")
    last_epoch = my_load_checkpoint(last_checkpoint, model, optimizer)
    config.train_start_epoch = last_epoch + 1
    if sechdule:
      sechdule.last_epoch = last_epoch
    print(f"Resuming training from epoch {config.train_start_epoch}")


def create_from_config(config: ExperimentConfig) -> tuple[MyTransformerLM, MyAdamW, MyCosineAnnealingLR | None]:
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
    config: ExperimentConfig,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
):
  model.to(device)
  optimizer.to(device)
  model.train()
  resume = "must" if config.run_id else "allow"

  with wandb.init(project=config.dataset_name, config=asdict(config), name=config.name, dir=WORKSPACE, id=config.run_id, resume=resume) as run:
    for epoch in range(config.train_start_epoch, config.train_epochs):
      if epoch % 10 == 0:
        print(f"Starting epoch {epoch}/{config.train_epochs}")
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
        checkpoint_path = config.checkpoint_dir / f"checkpoint_iter_{epoch}.pt"
        snapshot_path = config.snapshot_dir / f"snapshot_iter_{epoch}.pt"
        my_save_checkpoint(model, optimizer, epoch, checkpoint_path)
        my_save_xy_snapshot(x, y, logits, snapshot_path)
        print(f"Checkpoint saved at iteration {epoch}")

  # Final checkpoint
  my_save_checkpoint(model, optimizer, config.train_epochs,
                     config.checkpoint_dir / CHECKPOINT_FINAL_NAME)
  print("Training completed")


# %%


def train(config: ExperimentConfig):
  model, optimizer, sechdule = create_from_config(config)
  dataset_name = config.dataset_name
  train_data = load_data(OUTPUT_DIR / f"{dataset_name}_train.npy")
  # val_data = load_data(data_path=OUTPUT_DIR / f"{dataset_name}_valid.npy")
  train_prepare(config, model, optimizer, sechdule)

  # torch.cuda.memory._record_memory_history()
  train_model(model, optimizer, sechdule, train_data, None, config)
  # torch.cuda.memory._dump_snapshot("my_snapshot.pickle")


# %%


def validate_model(
    checkpoint_path: Path,
    val_data: np.ndarray,
    config: ExperimentConfig,
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
def validate(config: ExperimentConfig, checkpoint_name: str | None = None):
  dataset_name = config.dataset_name
  val_data = load_data(data_path=OUTPUT_DIR / f"{dataset_name}-valid.npy")
  checkpoint_name = checkpoint_name or CHECKPOINT_FINAL_NAME
  checkpoint_path = config.checkpoint_dir / checkpoint_name
  validate_model(checkpoint_path, val_data, config)

# %%


def inference(input_str: str, config: ExperimentConfig, checkpoint_name: str | None, inference_num: int = 100, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
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


# %%
# train(config)
# %%
# input_str = "Tom and Lily were playing with their toys in the living room. They liked to build towers and bridges with their blocks and cars."
# inference(input_str, config, checkpoint_name=None, inference_num=200)
# %%
# config_path = CONFIG_DIR / "base_lr_1e-03.toml"
# run_id = "x46r29n8"
# config = load_config(config_path)
# config.run_id = run_id
# train(config)
# %%
if __name__ == "__main__":
  config_path = Path(sys.argv[1])
  run_id: str | None = sys.argv[2] if len(sys.argv) > 2 else None
  config = load_config(config_path)
  config.run_id = run_id
  train(config)
