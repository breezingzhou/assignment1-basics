# %%

# 文本数据分词脚本 最终生成numpy的1维int数据(1D numpy array of integers)
# 存储成bin文件 提供给dataloader使用  np.memmap

from collections.abc import Generator
import json
import time
from cs336_basics.pretokenization_example import find_chunk_boundaries
from cs336_basics.tokenizer import BpeTokenizer
from cs336_basics.train_bpe import get_tokens, get_tokens_v2
from tests.common import gpt2_bytes_to_unicode
from tests.test_tokenizer import get_tokenizer_from_vocab_merges_path
from common import OUTPUT_DIR, DATA_DIR
from pathlib import Path
import numpy as np
import math
# %%


def get_contents(input_file: Path) -> str:
  with open(input_file, "r", encoding="utf-8") as f:
    text = f.read()
  return text


def get_contents_v2(input_file: Path, num_chunks: int, split_special_token: bytes) -> Generator[str]:
  with open(input_file, "rb") as f:
    boundaries = find_chunk_boundaries(f, num_chunks, split_special_token)

  for start, end in zip(boundaries[:-1], boundaries[1:]):
    with open(input_file, "rb") as f:
      f.seek(start)
      data = f.read(end - start).decode("utf-8", errors="ignore")
    yield data


def bpe_encode(dataset_name: str, groups: list[str] = ["valid", "train",]):
  print(f"Encoding dataset: {dataset_name} with groups: {groups}")
  tokenizer: BpeTokenizer = get_tokenizer_from_vocab_merges_path(
      OUTPUT_DIR / f"{dataset_name}_vocab.json",
      OUTPUT_DIR / f"{dataset_name}_merges.txt",
      special_tokens=["<|endoftext|>"]
  )

  for group in groups:
    print(f"Processing group: {group}")
    input_file = DATA_DIR / f"{dataset_name}_{group}.txt"
    output_file = OUTPUT_DIR / f"{dataset_name}_{group}.npy"
    size_bytes = input_file.stat().st_size
    chunk_size = int(0.5 * 1024 * 1024 * 1024)  # 0.5 GB

    start_time = time.time()
    if size_bytes <= chunk_size:  # less than 0.5 GB
      text = get_contents(input_file)
      all_tokens = tokenizer.encode(text, debug=True)
    else:
      split_special_token = b"<|endoftext|>"
      num_chunks = math.ceil(size_bytes / chunk_size)
      all_tokens = []
      for index, content in enumerate(get_contents_v2(input_file, num_chunks, split_special_token)):
        print(f"[{(time.time() - start_time):.2f}] Processing chunk {index + 1}/{num_chunks} for group {group}")
        tokens = tokenizer.encode(content, debug=True)
        all_tokens.extend(tokens)

    tokens_array = np.array(all_tokens, dtype=np.int32)
    np.save(output_file, tokens_array)
    print(f"[{(time.time() - start_time):.2f}] Saved encoded tokens to {output_file} with shape {tokens_array.shape}")


# %%
def count_tokens(dataset_name: str, groups: list[str] = ["valid", "train"]):
  gpt2_byte_encoder = gpt2_bytes_to_unicode()
  for group in groups:
    input_file = DATA_DIR / f"{dataset_name}_{group}.txt"
    output_file = OUTPUT_DIR / f"{dataset_name}_{group}_tokens.json"
    size_bytes = input_file.stat().st_size
    chunk_size = int(0.5 * 1024 * 1024 * 1024)  # 0.5 GB

    start_time = time.time()
    if size_bytes <= chunk_size:  # less than 0.5 GB
      tokens = get_tokens(input_file, special_tokens=["<|endoftext|>"])
    else:
      num_chunks = math.ceil(size_bytes / chunk_size)
      tokens = get_tokens_v2(input_file, special_tokens=["<|endoftext|>"],
                             num_chunks=num_chunks, num_processes=8)
    print(f"[{(time.time() - start_time):.2f}] Token counts for {dataset_name} {group}: {len(tokens)}")

    to_dump: dict[str, int] = {}
    for token, count in tokens.items():
      byte_token = token.encode("utf-8")
      encoded_chars = ''.join(gpt2_byte_encoder[b] for b in byte_token)
      to_dump[encoded_chars] = count

    with open(output_file, "w", encoding="utf-8") as f:
      json.dump(to_dump, f, indent=2, ensure_ascii=False)


# %%
# dataset_name = "TinyStoriesV2-GPT4"
dataset_name = "owt"
# bpe_encode(dataset_name)
count_tokens(dataset_name)
