# %%
from common import EXPERIMENT_DIR, ExperimentConfig, save_config, load_config
# %%
base_config_path = EXPERIMENT_DIR / "configs" / "base.toml"
base_config = load_config(base_config_path)

# %%


def experiment_lr(base_config: ExperimentConfig):
  base_config.schedule_params = None

  for lr in [1e-3, 3e-3, 1e-4, 3e-4]:
    config = base_config
    config.optimizer_params.learning_rate = lr
    config.name = f"base_lr_{lr:.0e}"
    config_path = EXPERIMENT_DIR / "configs" / f"base_lr_{lr:.0e}.toml"
    save_config(config, config_path)


# %%
experiment_lr(base_config)
