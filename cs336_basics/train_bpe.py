# %%
from dataclasses import dataclass
from pathlib import Path
import os
import regex
from collections import Counter

# %%
PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
PAT = regex.compile(PATTERN)
# %%


@dataclass
class SimplePair:
  first: bytes
  second: bytes

  def concat(self) -> bytes:
    return self.first + self.second

  def __eq__(self, other: "SimplePair") -> bool:
    return self.first == other.first and self.second == other.second

  def __hash__(self) -> int:
    return hash((self.first, self.second))

  def to_pair(self) -> tuple[bytes, bytes]:
    return (self.first, self.second)


@dataclass
class BpeToken:
  origin: bytes
  count: int
  tokens: list[bytes]

  def __repr__(self) -> str:
    return f"BpeToken({self.origin!r}, count={self.count})"


@dataclass
class BpePair:
  first: bytes
  second: bytes
  count: int
  froms_: list[BpeToken]

  def __lt__(self, other: "BpePair") -> bool:
    if self.count != other.count:
      return self.count < other.count
    if self.first != other.first:
      return self.first < other.first
    return self.second < other.second

  def to_pair(self) -> tuple[bytes, bytes]:
    return (self.first, self.second)

  def __repr__(self) -> str:
    return f"BpePair({self.first!r}, {self.second!r}, count={self.count})\n  from_: {self.froms_}"


def init_bpe_tokens(tokens: list[str]) -> list[BpeToken]:
  bpe_tokens = []
  for token, count in Counter(tokens).items():
    origin = token.encode("utf-8")
    byte_tokens = [origin[i:i + 1] for i in range(len(origin))]
    bpe_token = BpeToken(origin=origin, count=count, tokens=byte_tokens)
    bpe_tokens.append(bpe_token)
  return bpe_tokens


def init_bpe_pairs(bpe_tokens: list[BpeToken]) -> list[BpePair]:
  pair_2_froms: dict[SimplePair, list[BpeToken]] = {}
  pair_2_num: Counter[SimplePair] = Counter()

  for bpe_token in bpe_tokens:
    origin = bpe_token.origin
    count = bpe_token.count
    tokens = bpe_token.tokens
    for i in range(len(tokens) - 1):
      pair = SimplePair(tokens[i], tokens[i + 1])
      pair_2_froms.setdefault(pair, []).append(bpe_token)
      pair_2_num[pair] += count

  bpe_pairs: list[BpePair] = []
  for pair, froms_ in pair_2_froms.items():
    bpe_pair = BpePair(
        first=pair.first,
        second=pair.second,
        count=pair_2_num[pair],
        froms_=froms_
    )
    bpe_pairs.append(bpe_pair)

  return bpe_pairs


# %%
def update_bpe_pairs(bpe_pairs: list[BpePair], target_bpe_pair: BpePair, vocab_no: int) -> list[BpePair]:
  new_bpe_pairs: list[BpePair] = []
  target_pair = SimplePair(target_bpe_pair.first, target_bpe_pair.second)

  pair_2_from: dict[SimplePair, list[BpeToken]] = {}
  pair_2_num: Counter[SimplePair] = Counter()

  for bpe_pair in bpe_pairs:
    # 过滤原先的pair
    if bpe_pair.first == target_pair.first and bpe_pair.second == target_pair.second:
      continue
    # 与当前合并无关 直接忽略
    if bpe_pair.second != target_pair.first and bpe_pair.first != target_pair.second:
      new_bpe_pairs.append(bpe_pair)
      continue

    new_bpe_pair = BpePair(
        first=bpe_pair.first,
        second=bpe_pair.second,
        count=bpe_pair.count,
        froms_=bpe_pair.froms_[:]
    )

    # TODO 处理abcabc 这种情况
    # TODO 处理ining这种情况

    # (a, b) 合并的为 (b, c) 更新成 (a, bc)
    if bpe_pair.second == target_pair.first:
      new_pair = SimplePair(bpe_pair.first, target_pair.concat())
      for from_ in bpe_pair.froms_:
        if new_pair.concat() in from_.origin:
          pair_2_from.setdefault(new_pair, []).append(from_)
          pair_2_num[new_pair] += from_.count
          # 当前pair删除对应的from_以及数量
          if from_ not in new_bpe_pair.froms_:
            print(f"Warning1: vocab_no: {vocab_no} {from_} not in froms of {new_bpe_pair}")
          if from_ in new_bpe_pair.froms_:
            new_bpe_pair.count -= from_.count
            new_bpe_pair.froms_.remove(from_)
    if bpe_pair.first == target_pair.second:
      new_pair = SimplePair(target_pair.concat(), bpe_pair.second)
      for from_ in bpe_pair.froms_:
        if new_pair.concat() in from_.origin:
          pair_2_from.setdefault(new_pair, []).append(from_)
          pair_2_num[new_pair] += from_.count
          # 当前pair删除对应的from_以及数量
          if from_ not in new_bpe_pair.froms_:
            print(
                f"Warning2: vocab_no: {vocab_no} {from_} not in froms of {new_bpe_pair} target_pair: {target_bpe_pair}")
          if from_ in new_bpe_pair.froms_:
            new_bpe_pair.count -= from_.count
            new_bpe_pair.froms_.remove(from_)

    new_bpe_pairs.append(new_bpe_pair)

  # add new pairs
  for pair, from_ in pair_2_from.items():
    bpe_pair = BpePair(
        first=pair.first,
        second=pair.second,
        count=pair_2_num[pair],
        froms_=from_
    )
    new_bpe_pairs.append(bpe_pair)
  return new_bpe_pairs


def get_tokens(input_path: str | os.PathLike, special_tokens: list[str]) -> list[str]:
  tokens: list[str] = []
  with open(input_path, "r", encoding="utf-8") as f:
    data = f.read()

  # TODO 这里的正则表达式 special_tokens 里面的|会被处理
  contents = regex.split("|".join(special_tokens), data)
  for content in contents:
    tokens.extend(PAT.findall(content))

  return tokens
  ####


def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
  vocab: dict[int, bytes] = {}
  merges: list[tuple[bytes, bytes]] = []

  tokens = get_tokens(input_path, special_tokens)
  ####

  bpe_tokens = init_bpe_tokens(tokens)
  bpe_pairs = init_bpe_pairs(bpe_tokens)

  for i in range(len(special_tokens)):
    vocab[i] = special_tokens[i].encode("utf-8")

  vocab_single_start = len(special_tokens)
  for i in range(256):
    vocab[vocab_single_start + i] = bytes([i])

  vocab_merge_start = 256 + len(special_tokens)
  for vocab_no in range(vocab_merge_start, vocab_size):
    bpe_pair = max(bpe_pairs)
    vocab[vocab_no] = bpe_pair.first + bpe_pair.second
    merges.append(bpe_pair.to_pair())
    bpe_pairs = update_bpe_pairs(bpe_pairs, bpe_pair, vocab_no)
  return vocab, merges


def print_top(bpe_pairs: list[BpePair], n=8):
  bpe_pairs.sort(reverse=True)
  print("==" * 20)
  print("Top BPE Pairs:")
  for i in range(min(n, len(bpe_pairs))):
    print(f"  {i + 1}: {bpe_pairs[i]}")


# %%

# %%
tokens = ["training", "ing"]
