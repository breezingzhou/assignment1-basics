# %%
from pathlib import Path


# %%

WORKSPACE = Path(__file__).parent.parent
OUTPUT_DIR = WORKSPACE / "output"
DATA_DIR = WORKSPACE / "data"
EXPERIMENT_DIR = WORKSPACE / "experiment"
CHECKPOINT_FINAL_NAME = "checkpoint_final.pth"


__all__ = ["WORKSPACE", "OUTPUT_DIR", "DATA_DIR", "EXPERIMENT_DIR", "CHECKPOINT_FINAL_NAME",]
