# %%
from pathlib import Path


# %%

WORKSPACE = Path(__file__).parent.parent
OUTPUT_DIR = WORKSPACE / "output"
DATA_DIR = WORKSPACE / "data"
CHECKPOINTS_DIR = WORKSPACE / "checkpoints"


__all__ = ["WORKSPACE", "OUTPUT_DIR", "DATA_DIR", "CHECKPOINTS_DIR",]
