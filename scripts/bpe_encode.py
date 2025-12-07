# %%

# 文本数据分词脚本 最终生成numpy的1维int数据(1D numpy array of integers)
# 存储成bin文件 提供给dataloader使用  np.memmap

from collections import Counter
from collections.abc import Generator
import json
import time
from cs336_basics.pretokenization_example import find_chunk_boundaries
from cs336_basics.tokenizer import BpeTokenizer
from cs336_basics.train_bpe import get_tokens, get_tokens_v2
from tests.test_tokenizer import get_tokenizer_from_vocab_merges_path
from common import OUTPUT_DIR, DATA_DIR, TMP_DIR, gpt2_decode, gpt2_encode
from pathlib import Path
import numpy as np
import math
import tempfile

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


def merge_tokens(target_dir: Path):
  # 合并目录下所有chunk_{index}.npy文件 生成一个numpy数组
  files = list(target_dir.glob("chunk_*.npy"))
  files.sort(key=lambda x: int(x.stem.split("_")[1]))
  return np.concatenate([np.load(file) for file in files])


def bpe_encode(dataset_name: str, groups: list[str] = ["valid", "train",], create_cache: bool = False):
  print(f"Encoding dataset: {dataset_name} with groups: {groups}")
  tokenizer: BpeTokenizer = get_tokenizer_from_vocab_merges_path(
      OUTPUT_DIR / f"{dataset_name}_vocab.json",
      OUTPUT_DIR / f"{dataset_name}_merges.txt",
      special_tokens=["<|endoftext|>"]
  )

  for group in groups:
    print(f"Processing group: {group}")
    start_time = time.time()

    # 创建缓存
    if create_cache:
      print("Creating cache for tokenizer...")
      with open(OUTPUT_DIR / f"{dataset_name}_{group}_tokens.json", "r", encoding="utf-8") as f:
        data: dict[str, int] = json.load(f)
      tokens = {gpt2_decode(token): num for token, num in data.items()}
      tokenizer.pre_cache(Counter(tokens))
      print(f"[{(time.time() - start_time):.2f}] Cache created for {dataset_name} {group}")

    input_file = DATA_DIR / f"{dataset_name}_{group}.txt"
    output_file = OUTPUT_DIR / f"{dataset_name}_{group}.npy"
    size_bytes = input_file.stat().st_size
    chunk_size = int(0.5 * 1024 * 1024 * 1024)  # 0.5 GB 500MB

    if size_bytes <= chunk_size:  # less than 0.5 GB
      text = get_contents(input_file)
      all_tokens = tokenizer.encode(text, debug=True)
      tokens_array = np.array(all_tokens, dtype=np.int32)

    else:
      split_special_token = b"<|endoftext|>"
      num_chunks = math.ceil(size_bytes / chunk_size)
      temp_dir = Path(tempfile.mkdtemp(dir=TMP_DIR))
      print(f"[{(time.time() - start_time):.2f}] Created temporary directory at {temp_dir} for chunk files")
      for index, content in enumerate(get_contents_v2(input_file, num_chunks, split_special_token)):
        print(f"[{(time.time() - start_time):.2f}] Processing chunk {index + 1}/{num_chunks} for group {group}")
        tokens = tokenizer.encode(content, debug=True)
        np.save(temp_dir / f"chunk_{index}.npy", np.array(tokens, dtype=np.int32))
      print(f"[{(time.time() - start_time):.2f}] started merging chunks")
      tokens_array = merge_tokens(temp_dir)

    np.save(output_file, tokens_array)
    print(f"[{(time.time() - start_time):.2f}] Saved encoded tokens to {output_file} with shape {tokens_array.shape}")


# %%
def count_tokens(dataset_name: str, groups: list[str] = ["valid", "train"]):
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
      encoded_chars = gpt2_encode(token)
      to_dump[encoded_chars] = count

    with open(output_file, "w", encoding="utf-8") as f:
      json.dump(to_dump, f, indent=2, ensure_ascii=False)

# %%
# 运行的时候先执行count_tokens 创建出对应的tokens.json文件
# bpe_encode的时候 如果create_cache=True 会读取tokens.json文件创建tokenizer的缓存
# 如果不需要缓存 可以直接设置create_cache=False
# 然后再执行bpe_encode


# %%
# dataset_name = "TinyStoriesV2-GPT4"
dataset_name = "owt"
# count_tokens(dataset_name)
bpe_encode(dataset_name, groups=["train"], create_cache=True)

# %%
