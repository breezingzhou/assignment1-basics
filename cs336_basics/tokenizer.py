# %%
from collections.abc import Iterable, Iterator
import json

# %%


class Bpe_Tokenizer:
  vocab: dict[int, bytes]
  merges: list[tuple[bytes, bytes]]
  special_tokens: list[str] | None

  def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None):
    self.vocab = vocab
    self.merges = merges
    self.special_tokens = special_tokens

  def encode(self, text: str) -> list[int]:
    return [0]

  def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
    for item in iterable:
      yield from self.encode(item)

  def decode(self, ids: list[int]) -> str:
    return ""
