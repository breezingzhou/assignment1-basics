# %%

# 文本数据分词脚本 最终生成numpy的1维int数据(1D numpy array of integers)
# 存储成bin文件 提供给dataloader使用  np.memmap

from collections.abc import Generator
import time
from cs336_basics.pretokenization_example import find_chunk_boundaries
from cs336_basics.tokenizer import BpeTokenizer
from common import OUTPUT_DIR, DATA_DIR, get_tokenizer_from_vocab_merges_path
from pathlib import Path
import numpy as np
import json
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


def bpe_encode(dataset_name: str, groups=["valid", "train",]):
  print(f"Encoding dataset: {dataset_name} with groups: {groups}")
  tokenizer: BpeTokenizer = get_tokenizer_from_vocab_merges_path(
      OUTPUT_DIR / f"{dataset_name}_vocab.json",
      OUTPUT_DIR / f"{dataset_name}_merges.txt",
      special_tokens=["<|endoftext|>"]
  )

  for group in groups:
    print(f"Processing group: {group}")
    input_file = DATA_DIR / f"{dataset_name}-{group}.txt"
    output_file = OUTPUT_DIR / f"{dataset_name}-{group}.npy"
    size_bytes = input_file.stat().st_size
    chunk_size = int(0.5 * 1024 * 1024 * 1024)  # 0.5 GB

    start_time = time.time()
    if size_bytes <= chunk_size:  # less than 0.5 GB
      text = get_contents(input_file)
      all_tokens = tokenizer.encode(text)
    else:
      split_special_token = b"<|endoftext|>"
      num_chunks = math.ceil(size_bytes / chunk_size)
      all_tokens = []
      for index, content in enumerate(get_contents_v2(input_file, num_chunks, split_special_token)):
        print(f"[{(time.time() - start_time):.2f}] Processing chunk {index + 1}/{num_chunks} for group {group}")
        tokens = tokenizer.encode(content)
        all_tokens.extend(tokens)

    tokens_array = np.array(all_tokens, dtype=np.int32)
    np.save(output_file, tokens_array)
    print(f"[{(time.time() - start_time):.2f}] Saved encoded tokens to {output_file} with shape {tokens_array.shape}")


# %%
dataset_name = "TinyStoriesV2-GPT4"
bpe_encode(dataset_name)
