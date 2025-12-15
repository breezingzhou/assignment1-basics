# %%
from pathlib import Path
from tests.common import gpt2_bytes_to_unicode


# %%

WORKSPACE = Path(__file__).parent.parent
OUTPUT_DIR = WORKSPACE / "output"
DATA_DIR = WORKSPACE / "data"
EXPERIMENT_DIR = WORKSPACE / "experiment"
CHECKPOINT_FINAL_NAME = "checkpoint_final.pth"
CONFIG_DIR = WORKSPACE / "configs"
EXPERIMENT_CONFIG_DIR = CONFIG_DIR / "experiment"
BASE_EXPERIMENT_CONFIG_PATH = EXPERIMENT_CONFIG_DIR / "base.yaml"
TMP_DIR = WORKSPACE / "tmp"


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
