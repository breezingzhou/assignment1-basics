# %%
from copy import deepcopy
import numpy as np
from common import EXPERIMENT_CONFIG_DIR, BASE_EXPERIMENT_CONFIG_PATH, OUTPUT_DIR
from cs336_basics.nn_utils import MyDataLoader
from experiment_config import Config, ExperimentConfig

# %%
base_config = ExperimentConfig.load_config(BASE_EXPERIMENT_CONFIG_PATH)

# %%


def summary_model(config: ExperimentConfig):
  from torchinfo import summary
  model, _, _ = config.create_llm()
  data_path = OUTPUT_DIR / f"idxs.{config.dataset_name}_train.npy"
  train_data = np.memmap(data_path, dtype=np.uint32, mode="r")
  train_loader = MyDataLoader(train_data, config.batch_size,
                              config.model_params.context_length, device='cpu')
  x, y = train_loader[0]
  print(summary(model, input_data=x, verbose=0))


# %%


def experiment_lr(base_config: ExperimentConfig):
  base_config.schedule_params = None

  for lr in [1e-3, 3e-3, 1e-4, 3e-4, 3e-5]:
    config = deepcopy(base_config)
    config.optimizer_params.learning_rate = lr
    config.name = f"base_lr_{lr:.0e}"
    config_path = EXPERIMENT_CONFIG_DIR / f"base_lr_{lr:.0e}.yaml"
    config.save_config(config_path)


# %%

def experiment_batch_size(base_config: ExperimentConfig):
  base_config.schedule_params = None

  for batch_size in [32, 64, 128, 256]:
    config = deepcopy(base_config)
    config.batch_size = batch_size
    config.name = f"base_bs_{batch_size}"
    config_path = EXPERIMENT_CONFIG_DIR / f"base_bs_{batch_size}.yaml"
    config.save_config(config_path)

# %%
# config = deepcopy(base_config)
# config.batch_size = 256
# summary_model(config)

# %%
# experiment_lr(base_config)
experiment_batch_size(base_config)
