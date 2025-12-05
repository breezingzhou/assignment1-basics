# %%
from pathlib import Path


# %%

WORKSPACE = Path(__file__).parent.parent
OUTPUT_DIR = WORKSPACE / "output"
DATA_DIR = WORKSPACE / "data"
EXPERIMENT_DIR = WORKSPACE / "experiment"


__all__ = ["WORKSPACE", "OUTPUT_DIR", "DATA_DIR", "EXPERIMENT_DIR",]
