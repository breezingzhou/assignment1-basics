# %%
from collections import Counter
from collections.abc import Iterable, Iterator
from functools import lru_cache
import json

from cs336_basics.bpe_utils import PAT, merge_tokens, split_special_tokens_tokenizer
from cs336_basics.bpe_types import BpePair, Content, ContentType, SimplePair
from cs336_basics.train_bpe import init_bpe_pairs, init_bpe_tokens, update_bpe_pairs_dict


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
    self._cache: dict[str, list[int]] = {}
    if special_tokens is not None:
      self.special_tokens = special_tokens[::]
      self.special_tokens.sort(key=lambda x: len(x), reverse=True)

  @lru_cache(maxsize=None)
  def _encode_token(self, token: str) -> list[int]:
    origin = token.encode("utf-8")
    byte_tokens = [origin[i:i + 1] for i in range(len(origin))]
    for first, second in self.merges:
      byte_tokens = merge_tokens(byte_tokens, SimplePair(first, second))
    token_ids = [self.vocab_rev.get(byte_token, 0) for byte_token in byte_tokens]
    return token_ids

  def encode_token(self, token: str) -> list[int]:
    if token in self._cache:
      return self._cache[token]
    token_ids = self._encode_token(token)
    return token_ids

  def encode(self, text: str, debug: bool = False) -> list[int]:
    res = []

    if self.special_tokens:
      contents = split_special_tokens_tokenizer(text, self.special_tokens)
    else:
      contents = [Content.normal(text)]

    debug_every_n = max(len(contents) // 40, 1)
    for index, content in enumerate(contents):
      if debug and index % debug_every_n == 0:
        print(f"Processing content {index}/{len(contents)}")
      if content.type == ContentType.SPECIAL:
        res.append(self.vocab_rev.get(content.content.encode(), 0))
      else:
        tokens: list[str] = PAT.findall(content.content)
        for token in tokens:
          res.extend(self.encode_token(token))
    return res

  def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
    for item in iterable:
      yield from self.encode(item)

  def decode(self, ids: list[int]) -> str:
    bytes_list = [self.vocab.get(i, b"") for i in ids]
    decoded = b"".join(bytes_list).decode("utf-8", errors="replace")
    return decoded

  def create_cache(self, tokens: Counter[str]) -> dict[str, list[int]]:
    bpe_tokens = init_bpe_tokens(tokens)
    bpe_pairs = init_bpe_pairs(bpe_tokens)
    bpe_pairs_dict = {bp.to_simple(): bp for bp in bpe_pairs}

    for merge in self.merges:
      target_bpe_pair = bpe_pairs_dict.get(SimplePair(merge[0], merge[1]))
      if target_bpe_pair is None:
        continue
      update_bpe_pairs_dict(bpe_pairs_dict, target_bpe_pair)

    cache: dict[str, list[int]] = {}
    for bpe_token in bpe_tokens:
      token_ids = [self.vocab_rev.get(byte_token, 0) for byte_token in bpe_token.tokens]
      cache[bpe_token.origin_str] = token_ids
    return cache

  def pre_cache(self, tokens: Counter[str]) -> None:
    self._cache = {}
    cache_data = self.create_cache(tokens)
    self._cache = cache_data
