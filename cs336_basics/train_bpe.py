# %%
from dataclasses import dataclass
from pathlib import Path
import os
import regex
from collections import Counter
from multiprocessing import Pool
from cs336_basics.pretokenization_example import find_chunk_boundaries
from cs336_basics.bpe_utils import PAT, merge_tokens, diff_tokens, count_apperence, split_special_tokens_training
from cs336_basics.bpe_types import BpeToken, BpePair, SimplePair
# %%


def init_bpe_tokens(tokens: Counter[str]) -> list[BpeToken]:
  bpe_tokens = []
  for token, count in tokens.items():
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


def get_tokens(input_path: str | os.PathLike, special_tokens: list[str]) -> Counter[str]:
  with open(input_path, "r", encoding="utf-8") as f:
    data = f.read()
  tokens = split_special_tokens_training(data, special_tokens)
  return tokens


def process_task(input_path: str, start: int, end: int, special_tokens: list[str]) -> Counter[str]:
  with open(input_path, "rb") as f:
    f.seek(start)
    data = f.read(end - start).decode("utf-8", errors="ignore")
  tokens = split_special_tokens_training(data, special_tokens)
  return tokens


def get_tokens_v2(input_path: str | os.PathLike, special_tokens: list[str], split_special_token: bytes = b"<|endoftext|>", num_chunks: int = 16, num_processes: int = 8) -> Counter[str]:
  with open(input_path, "rb") as f:
    boundaries = find_chunk_boundaries(f, num_chunks, split_special_token)

  args_list = [
      (str(input_path), start, end, special_tokens) for start, end in zip(boundaries[:-1], boundaries[1:])
  ]

  with Pool(processes=num_processes) as pool:
    results = pool.starmap(process_task, args_list)
  total_tokens = Counter[str]()
  for result in results:
    total_tokens.update(result)
  return total_tokens
# %%


def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
  num_chunks = kwargs.get("num_chunks")
  num_processes = kwargs.get("num_processes")
  split_special_token = kwargs.get("split_special_token", b"<|endoftext|>")

  if num_chunks and num_processes:
    tokens = get_tokens_v2(input_path, special_tokens, split_special_token=split_special_token,
                           num_chunks=num_chunks, num_processes=num_processes)
  else:
    tokens = get_tokens(input_path, special_tokens)

  bpe_tokens = init_bpe_tokens(tokens)
  bpe_pairs = init_bpe_pairs(bpe_tokens)
  bpe_pairs_dict = {bp.to_simple(): bp for bp in bpe_pairs}

  vocab: dict[int, bytes] = {}
  merges: list[tuple[bytes, bytes]] = []
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
