# %%
from collections.abc import Iterable, Iterator
from functools import lru_cache
import json

from cs336_basics.bpe_utils import PAT, merge_tokens, split_special_tokens_tokenizer
from cs336_basics.bpe_types import Content, ContentType, SimplePair


# %%


class BpeTokenizer:
  vocab: dict[int, bytes]
  merges: list[tuple[bytes, bytes]]
  special_tokens: list[str] | None = None
  vocab_rev: dict[bytes, int]

  def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None):
    self.vocab = vocab
    self.merges = merges
    self.vocab_rev = {v: k for k, v in vocab.items()}
    if special_tokens is not None:
      self.special_tokens = special_tokens[::]
      self.special_tokens.sort(key=lambda x: len(x), reverse=True)

  def encode(self, text: str) -> list[int]:
    res = []

    if self.special_tokens:
      contents = split_special_tokens_tokenizer(text, self.special_tokens)
    else:
      contents = [Content.normal(text)]

    for content in contents:
      if content.type == ContentType.SPECIAL:
        res.append(self.vocab_rev.get(content.content.encode(), 0))
      else:
        tokens: list[str] = PAT.findall(content.content)
        for token in tokens:
          res.extend(self.encode_token(token))
    return res

  @lru_cache(maxsize=10000)
  def encode_token(self, token: str) -> list[int]:
    origin = token.encode("utf-8")
    byte_tokens = [origin[i:i + 1] for i in range(len(origin))]
    for first, second in self.merges:
      byte_tokens = merge_tokens(byte_tokens, SimplePair(first, second))
    token_ids = [self.vocab_rev.get(byte_token, 0) for byte_token in byte_tokens]
    return token_ids

  def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
    for item in iterable:
      yield from self.encode(item)

  def decode(self, ids: list[int]) -> str:
    bytes_list = [self.vocab.get(i, b"") for i in ids]
    decoded = b"".join(bytes_list).decode("utf-8", errors="replace")
    return decoded
