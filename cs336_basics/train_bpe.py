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
    return f"BpeToken({self.origin!r}, count={self.count}, tokens={self.tokens})"

  def __hash__(self) -> int:
    return hash(self.origin)


@dataclass
class BpePair:
  first: bytes
  second: bytes
  count: int
  froms_: set[BpeToken]

  def __lt__(self, other: "BpePair") -> bool:
    if self.count != other.count:
      return self.count < other.count
    if self.first != other.first:
      return self.first < other.first
    return self.second < other.second

  def to_pair(self) -> tuple[bytes, bytes]:
    return (self.first, self.second)

  def to_simple(self) -> SimplePair:
    return SimplePair(self.first, self.second)

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
        froms_=set(froms_)
    )
    bpe_pairs.append(bpe_pair)

  return bpe_pairs

# %%


def merge_tokens(tokens: list[bytes], target_pair: SimplePair) -> list[bytes]:
  merged_tokens: list[bytes] = []
  i = 0
  while i < len(tokens):
    if i < len(tokens) - 1 and tokens[i:i + 2] == [target_pair.first, target_pair.second]:
      merged_tokens.append(target_pair.concat())
      i += 2
    else:
      merged_tokens.append(tokens[i])
      i += 1
  return merged_tokens

# %%


def diff_tokens(new_tokens: list[bytes], target_pair: SimplePair):
  to_add = []
  to_remove = []
  concat = target_pair.concat()
  for index, token in enumerate(new_tokens):
    if token != concat:
      continue
    pre_index = index - 1
    post_index = index + 1
    if pre_index >= 0 and new_tokens[pre_index] != concat:
      to_add.append(SimplePair(new_tokens[pre_index], concat))
      to_remove.append(SimplePair(new_tokens[pre_index], target_pair.first))
    if post_index < len(new_tokens):
      to_add.append(SimplePair(concat, new_tokens[post_index]))
      # 如果后一个token不是concat
      if new_tokens[post_index] != concat:
        to_remove.append(SimplePair(target_pair.second, new_tokens[post_index]))
      # 后一个也是合并生成的
      else:
        to_remove.append(SimplePair(target_pair.second, target_pair.first))
  return to_add, to_remove
# %%


def count_apperence(tokens: list[bytes], pair: SimplePair) -> int:
  count = 0
  for first, second in zip(tokens, tokens[1:]):
    if first == pair.first and second == pair.second:
      count += 1
  return count


def update_bpe_pairs_dict(bpe_pairs_dict: dict[SimplePair, BpePair], target_bpe_pair: BpePair, vocab_no: int):
  target_pair = SimplePair(target_bpe_pair.first, target_bpe_pair.second)
  bpe_pairs_dict.pop(target_pair)

  for bpe_token in target_bpe_pair.froms_:
    # 更新bpe_token的tokens
    origin_tokens = bpe_token.tokens
    new_tokens = merge_tokens(origin_tokens, target_pair)
    bpe_token.tokens = new_tokens

    # 重新计算各个froms
    to_add, to_remove = diff_tokens(new_tokens, target_pair)
    for pair, num in Counter(to_remove).items():
      bpe_pair = bpe_pairs_dict.get(pair)
      if bpe_pair is None:
        # 合并(0,0)
        # bpe_token = (0,0,0)
        # to_add = [(00,0)]  to_remove = [(0,0)]
        continue
      # "strengthen" 合并'h e'的时候 删除'e n'的计数 但是不影响前一个'e n'
      total_num = count_apperence(origin_tokens, pair)
      remain_num = total_num - num
      assert remain_num >= 0
      bpe_pair.count -= bpe_token.count * num
      if remain_num == 0:
        bpe_pair.froms_.remove(bpe_token)
    for pair, num in Counter(to_add).items():
      bpe_pair = bpe_pairs_dict.get(pair)
      if bpe_pair is None:
        bpe_pair = BpePair(
            first=pair.first,
            second=pair.second,
            count=0,
            froms_=set()
        )
        bpe_pairs_dict[pair] = bpe_pair
      bpe_pair.count += (bpe_token.count * num)
      bpe_pair.froms_.add(bpe_token)


def get_tokens(input_path: str | os.PathLike, special_tokens: list[str]) -> list[str]:
  tokens: list[str] = []
  with open(input_path, "r", encoding="utf-8") as f:
    data = f.read()

  contents = regex.split("|".join(regex.escape(token) for token in special_tokens), data)
  for content in contents:
    tokens.extend(PAT.findall(content))
  return tokens

# %%


def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
  vocab: dict[int, bytes] = {}
  merges: list[tuple[bytes, bytes]] = []

  tokens = get_tokens(input_path, special_tokens)
  bpe_tokens = init_bpe_tokens(tokens)
  bpe_pairs = init_bpe_pairs(bpe_tokens)
  bpe_pairs_dict = {bp.to_simple(): bp for bp in bpe_pairs}

  for i in range(len(special_tokens)):
    vocab[i] = special_tokens[i].encode("utf-8")

  vocab_single_start = len(special_tokens)
  for i in range(256):
    vocab[vocab_single_start + i] = bytes([i])

  vocab_merge_start = 256 + len(special_tokens)
  for vocab_no in range(vocab_merge_start, vocab_size):
    bpe_pair = max(bpe_pairs_dict.values())
    vocab[vocab_no] = bpe_pair.first + bpe_pair.second
    merges.append(bpe_pair.to_pair())
    update_bpe_pairs_dict(bpe_pairs_dict, bpe_pair, vocab_no)
  return vocab, merges


# %%
