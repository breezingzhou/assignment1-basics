# %%
from pathlib import Path
from cs336_basics.tokenizer import BpeTokenizer
from tests.common import gpt2_bytes_to_unicode
import json

# %%

WORKSPACE = Path(__file__).parent.parent
OUTPUT_DIR = WORKSPACE / "output"
DATA_DIR = WORKSPACE / "data"
CHECKPOINTS_DIR = WORKSPACE / "checkpoints"


def get_tokenizer_from_vocab_merges_path(
    vocab_path: Path,
    merges_path: Path,
    special_tokens: list[str] | None = None,
):
  gpt2_byte_decoder = {v: k for k, v in gpt2_bytes_to_unicode().items()}
  with open(vocab_path) as vocab_f:
    gpt2_vocab: dict[int, str] = json.load(vocab_f)

  gpt2_merges = []
  with open(merges_path) as merges_f:
    for line in merges_f:
      cleaned_line = line.rstrip()
      if cleaned_line and len(cleaned_line.split(" ")) == 2:
        gpt2_merges.append(tuple(cleaned_line.split(" ")))

  vocab = {
      int(index): bytes([gpt2_byte_decoder[token] for token in item])
      for index, item in gpt2_vocab.items()
  }
  merges = [
      (
          bytes([gpt2_byte_decoder[token] for token in merge_token_1]),
          bytes([gpt2_byte_decoder[token] for token in merge_token_2]),
      )
      for merge_token_1, merge_token_2 in gpt2_merges
  ]
  return BpeTokenizer(vocab, merges, special_tokens=special_tokens)


__all__ = ["WORKSPACE", "OUTPUT_DIR", "DATA_DIR",
           "CHECKPOINTS_DIR", "get_tokenizer_from_vocab_merges_path"]
