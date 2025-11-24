#%%
from dataclasses import dataclass
from pathlib import Path
import os
import regex
from collections import Counter

#%%
PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
PAT = regex.compile(PATTERN)
#%%
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

@dataclass
class BpeToken:
  byte_token: bytes
  count: int

@dataclass
class BpePair:
  first: bytes
  second: bytes
  count: int
  from_ : list[BpeToken]

  def __lt__(self, other: "BpePair") -> bool:
    if self.count != other.count:
      return self.count < other.count
    if self.first != other.first:
      return self.first < other.first
    return self.second < other.second

  def to_pair(self) -> tuple[bytes, bytes]:
    return (self.first, self.second)

  def __repr__(self) -> str:
    return f"BpePair({self.first!r}, {self.second!r}, count={self.count})"

def init_bpe_pairs(bpe_tokens: list[BpeToken]) -> list[BpePair]:
  pair_2_from: dict[SimplePair, list[BpeToken]] = {}
  pair_2_num: Counter[SimplePair] = Counter()

  for bpe_token in bpe_tokens:
    byte_token = bpe_token.byte_token
    count = bpe_token.count
    for i in range(len(byte_token)-1):
      pair = SimplePair(byte_token[i:i+1], byte_token[i+1:i+2])
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




#%%
def update_bpe_pairs(bpe_pairs: list[BpePair], target_bpe_pair: BpePair) -> list[BpePair]:
  new_bpe_pairs: list[BpePair] = []
  target_pair = SimplePair(target_bpe_pair.first, target_bpe_pair.second)

  pair_2_from: dict[SimplePair, list[BpeToken]] = {}
  pair_2_num: Counter[SimplePair] = Counter()
  for bpe_pair in bpe_pairs:
    if bpe_pair.first == target_pair.first and bpe_pair.second == target_pair.second:
      continue
    # pair后面可能与当前pair合并
    if bpe_pair.second == target_pair.first:
      new_pair = SimplePair(bpe_pair.first, target_pair.concat())
      for from_ in bpe_pair.from_:
        if new_pair.concat() in from_.byte_token:
          pair_2_from.setdefault(new_pair, []).append(from_)
          pair_2_num[new_pair] += from_.count
          # 当前pair删除对应的from_以及数量
          bpe_pair.count -= from_.count
          bpe_pair.from_.remove(from_)
    if bpe_pair.first == target_pair.second:
      new_pair = SimplePair(target_pair.concat(), bpe_pair.second)
      for from_ in bpe_pair.from_:
        if new_pair.concat() in from_.byte_token:
          pair_2_from.setdefault(new_pair, []).append(from_)
          pair_2_num[new_pair] += from_.count
          # 当前pair删除对应的from_以及数量
          bpe_pair.count -= from_.count
          bpe_pair.from_.remove(from_)

    # TODO create a new BpePair dont modify last turn
    new_bpe_pairs.append(bpe_pair)
  return new_bpe_pairs

def train_bpe(
  input_path: str | os.PathLike,
  vocab_size: int,
  special_tokens: list[str],
  **kwargs
  ) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
  vocab: dict[int, bytes] = {}
  merges: list[tuple[bytes, bytes]] = []

  tokens: list[str] = []
  with open(input_path, "r", encoding="utf-8") as f:
    data = f.read()

    contents= regex.split("|".join(special_tokens), data)
    for content in contents:
      tokens.extend(PAT.findall(content))

  ####

  byte_tokens = [token.encode("utf-8") for token in tokens]
  bpe_tokens = [BpeToken(byte_token=bt, count=count) for bt, count in Counter(byte_tokens).items()]


  bpe_pairs = init_bpe_pairs(bpe_tokens)


  # a = pair_2_num.most_common(1)[0]

  vocab_special_start = 256
  for i in range(len(special_tokens)):
    vocab[vocab_special_start + i] = special_tokens[i].encode("utf-8")

  vocab_merge_start = 256+ len(special_tokens)
  for vocab_no in range(vocab_merge_start, vocab_size):
    bpe_pair = max(bpe_pairs)
    vocab[vocab_no]  = bpe_pair.first + bpe_pair.second
    merges.append(bpe_pair.to_pair())
    bpe_pairs = update_bpe_pairs(bpe_pairs, bpe_pair)
  return vocab, merges


#%%

#%%

# input_path = Path(__file__).parent.parent / "tests/fixtures/tinystories_sample.txt"
# train_bpe(
#   input_path=input_path,
#   vocab_size=1000,
#   special_tokens=["<|endoftext|>"],
# )
FIXTURES_PATH = Path(__file__).parent.parent / "tests/fixtures"
input_path = FIXTURES_PATH / "corpus.en"
vocab, merges = train_bpe(
    input_path=input_path,
    vocab_size=500,
    special_tokens=["<|endoftext|>"],
)
