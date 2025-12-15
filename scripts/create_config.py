# %%
from common import EXPERIMENT_CONFIG_DIR, BASE_EXPERIMENT_CONFIG_PATH
from experiment_config import ExperimentConfig
# %%
base_config = ExperimentConfig.load_config(BASE_EXPERIMENT_CONFIG_PATH)

# %%


def experiment_lr(base_config: ExperimentConfig):
  base_config.schedule_params = None

  for lr in [1e-3, 3e-3, 1e-4, 3e-4]:
    config = base_config
    config.optimizer_params.learning_rate = lr
    config.name = f"base_lr_{lr:.0e}"
    config_path = EXPERIMENT_CONFIG_DIR / f"base_lr_{lr:.0e}.yaml"
    config.save_config(config_path)


# %%
experiment_lr(base_config)
