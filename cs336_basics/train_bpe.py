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
  byte_token: bytes
  count: int

  def __repr__(self) -> str:
    return f"BpeToken({self.byte_token!r}, count={self.count})"


@dataclass
class BpePair:
  first: bytes
  second: bytes
  count: int
  from_: list[BpeToken]

  def __lt__(self, other: "BpePair") -> bool:
    if self.count != other.count:
      return self.count < other.count
    if self.first != other.first:
      return self.first < other.first
    return self.second < other.second

  def to_pair(self) -> tuple[bytes, bytes]:
    return (self.first, self.second)

  def __repr__(self) -> str:
    return f"BpePair({self.first!r}, {self.second!r}, count={self.count})\n  from_: {self.from_}"


def init_bpe_pairs(bpe_tokens: list[BpeToken]) -> list[BpePair]:
  pair_2_from: dict[SimplePair, list[BpeToken]] = {}
  pair_2_num: Counter[SimplePair] = Counter()

  for bpe_token in bpe_tokens:
    byte_token = bpe_token.byte_token
    count = bpe_token.count
    for i in range(len(byte_token) - 1):
      pair = SimplePair(byte_token[i:i + 1], byte_token[i + 1:i + 2])
      pair_2_from.setdefault(pair, []).append(bpe_token)
      pair_2_num[pair] += count

  bpe_pairs: list[BpePair] = []
  for pair, from_ in pair_2_from.items():
    bpe_pair = BpePair(
        first=pair.first,
        second=pair.second,
        count=pair_2_num[pair],
        from_=from_
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
        from_=bpe_pair.from_[:]
    )

    # TODO 处理abcabc 这种情况
    # TODO 处理ining这种情况

    # (a, b) 合并的为 (b, c) 更新成 (a, bc)
    if bpe_pair.second == target_pair.first:
      new_pair = SimplePair(bpe_pair.first, target_pair.concat())
      for from_ in bpe_pair.from_:
        if new_pair.concat() in from_.byte_token:
          pair_2_from.setdefault(new_pair, []).append(from_)
          pair_2_num[new_pair] += from_.count
          # 当前pair删除对应的from_以及数量
          if from_ not in new_bpe_pair.from_:
            print(f"Warning1: vocab_no: {vocab_no} {from_} not in froms of {new_bpe_pair}")
          if from_ in new_bpe_pair.from_:
            new_bpe_pair.count -= from_.count
            new_bpe_pair.from_.remove(from_)
    if bpe_pair.first == target_pair.second:
      new_pair = SimplePair(target_pair.concat(), bpe_pair.second)
      for from_ in bpe_pair.from_:
        if new_pair.concat() in from_.byte_token:
          pair_2_from.setdefault(new_pair, []).append(from_)
          pair_2_num[new_pair] += from_.count
          # 当前pair删除对应的from_以及数量
          if from_ not in new_bpe_pair.from_:
            print(
                f"Warning2: vocab_no: {vocab_no} {from_} not in froms of {new_bpe_pair} target_pair: {target_bpe_pair}")
          if from_ in new_bpe_pair.from_:
            new_bpe_pair.count -= from_.count
            new_bpe_pair.from_.remove(from_)

    new_bpe_pairs.append(new_bpe_pair)

  # add new pairs
  for pair, from_ in pair_2_from.items():
    bpe_pair = BpePair(
        first=pair.first,
        second=pair.second,
        count=pair_2_num[pair],
        from_=from_
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

  byte_tokens = [token.encode("utf-8") for token in tokens]
  bpe_tokens = [BpeToken(byte_token=bt, count=count) for bt, count in Counter(byte_tokens).items()]
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

# input_path = Path(__file__).parent.parent / "tests/fixtures/tinystories_sample.txt"
# train_bpe(
#   input_path=input_path,
#   vocab_size=1000,
#   special_tokens=["<|endoftext|>"],
# )

from tests.common import gpt2_bytes_to_unicode
import json


def diff(a, b):
  print(set(a) - set(b))
  print(set(b) - set(a))


gpt2_byte_encoder: dict[int, str] = gpt2_bytes_to_unicode()


def encode(value: bytes, encoder=gpt2_byte_encoder) -> str:
  return "".join([encoder[b] for b in value])


FIXTURES_PATH = Path(__file__).parent.parent / "tests/fixtures"
input_path = FIXTURES_PATH / "corpus.en"
vocab, merges = train_bpe(
    input_path=input_path,
    vocab_size=500,
    special_tokens=["<|endoftext|>"],
)
vocabs = [encode(v) for v in vocab.values()]
merges_list = [(encode(v1), encode(v2)) for v1, v2 in merges]

# %%

reference_vocab_path = FIXTURES_PATH / "train-bpe-reference-vocab.json"
reference_merges_path = FIXTURES_PATH / "train-bpe-reference-merges.txt"

with open(reference_vocab_path, encoding="utf-8") as f:
  gpt2_reference_vocab = json.load(f)
reference_vocab = gpt2_reference_vocab.keys()


diff(vocabs, reference_vocab)

with open(reference_merges_path, encoding="utf-8") as f:
  gpt2_reference_merges = [tuple(line.rstrip().split(" ")) for line in f]

for index, (merge1, merge2) in enumerate(zip(merges_list, gpt2_reference_merges)):
  if merge1 != merge2:
    print(f"Diff at index {index}: {merge1} != {merge2}")
    break
