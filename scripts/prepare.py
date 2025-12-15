# %%
# 处理从dataset到tokenizer的准备工作
# 取代了writeup/writeup.py 和 bpe_encode.py
from collections.abc import Callable
from dataclasses import dataclass
import gzip
from pathlib import Path
import sys
from unitoken import BpeTrainer, BpeEncoder, PreTokenizer
from common import DATA_DIR, OUTPUT_DIR
import numpy as np
import requests
# %%
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
# %%


@dataclass
class Config:
  dataset_name: str
  vocab_size: int
  special_tokens: list[str]
  num_chunks: int
  download_fn: Callable[[], None]

# %%


def download_file(url: str, dest_path: Path):

  if dest_path.exists():
    print(f"{dest_path} already exists, skipping download.")
    return

  print(f"Downloading {url}")
  response = requests.get(url)
  with open(dest_path, "wb") as f:
    f.write(response.content)


def decompress_gz_file(gz_path: Path, output_path: Path, chunk_size=1024 * 1024):
  output_path.parent.mkdir(parents=True, exist_ok=True)
  if output_path.exists():
    print(f"{output_path} already exists, skipping decompression.")
    return

  print(f"Decompressing {gz_path} to {output_path}")
  try:
    with gzip.open(gz_path, "rb") as f_in:
      with open(output_path, "wb") as f_out:
        while True:
          chunk = f_in.read(chunk_size)
          if not chunk:
            break
          f_out.write(chunk)
    print(f"Decompressed {output_path}")
  except Exception as e:
    print(f"Decompression failed: {e}")


def download_tinystory():
  train_url = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt"
  valid_url = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt"
  train_file = DATA_DIR / "TinyStoriesV2-GPT4_train.txt"
  valid_file = DATA_DIR / "TinyStoriesV2-GPT4_valid.txt"
  download_file(train_url, train_file)
  download_file(valid_url, valid_file)


def download_owt():
  train_url = "https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_train.txt.gz"
  valid_url = "https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_valid.txt.gz"
  train_gzfile = DATA_DIR / "owt_train.txt.gz"
  valid_gzfile = DATA_DIR / "owt_valid.txt.gz"
  train_file = DATA_DIR / "owt_train.txt"
  valid_file = DATA_DIR / "owt_valid.txt"
  download_file(train_url, train_gzfile)
  download_file(valid_url, valid_gzfile)
  decompress_gz_file(train_gzfile, train_file)
  decompress_gz_file(valid_gzfile, valid_file)


def download_dataset(config: Config):
  config.download_fn()


def bpe_train(config: Config):
  # 生成vocab和merges文件
  input_file = DATA_DIR / f"{config.dataset_name}_train.txt"
  vocab_file = OUTPUT_DIR / f"vocab.{config.dataset_name}[u8].json"
  merges_file = OUTPUT_DIR / f"merges.{config.dataset_name}[u8].txt"
  if vocab_file.exists() and merges_file.exists():
    print(f"Vocab and merges files already exist for {config.dataset_name}, skipping BPE training.")
    return

  print(f"Starting BPE training for {config.dataset_name}")
  pre_tokenizer = PreTokenizer(
      special_tokens=config.special_tokens,
      eot_token=None
  )
  words = pre_tokenizer.get_words_from_file(
      input_file,
      desired_num_chunks=config.num_chunks
  )

  trainer = BpeTrainer(
      special_tokens=config.special_tokens,
  )
  trainer.add_words(words)
  trainer.train(num_steps=config.vocab_size)
  trainer.save(
      config.dataset_name,
      outdir=OUTPUT_DIR
  )
  print("BPE training completed")


def bpe_encode(config: Config):
  encoder = None

  for group in ["train", "valid"]:
    input_file = DATA_DIR / f"{config.dataset_name}_{group}.txt"
    output_file = OUTPUT_DIR / f"idxs.{input_file.stem}.npy"
    if output_file.exists():
      print(
          f"Encoded idxs file already exists at {output_file}, skipping BPE encoding for {group}.")
      continue

    print(f"Starting BPE encoding for {config.dataset_name} {group}")
    if encoder is None:
      encoder = BpeEncoder.load(name=config.dataset_name, input_dir=OUTPUT_DIR)
    idxs = encoder.encode_file(
        path=input_file,
        num_chunks=config.num_chunks
    )
    print("BPE encoding completed, start saving")
    np.save(output_file, idxs)
    print(f"Saved encoded idxs to {output_file}")


# %%
tiny_story_config = Config(
    dataset_name="TinyStoriesV2-GPT4",
    vocab_size=10000,
    special_tokens=["<|endoftext|>"],
    num_chunks=32,
    download_fn=download_tinystory,
)

owt_config = Config(
    dataset_name="owt",
    vocab_size=32000,
    special_tokens=["<|endoftext|>"],
    num_chunks=256,
    download_fn=download_owt,
)
name_2_config = {
    "tinystory": tiny_story_config,
    "owt": owt_config,
}
# %%
# bpe_train(config=tiny_story_config)
# bpe_encode(config=tiny_story_config)
# %%
# %%
if __name__ == "__main__":
  dataset_name = sys.argv[1]
  if name_2_config.get(dataset_name) is None:
    print(f"Unknown dataset name: {dataset_name}")
    print("Available dataset names:", list(name_2_config.keys()))
    sys.exit(1)
  config = name_2_config[dataset_name]
  download_dataset(config)
  bpe_train(config)
  bpe_encode(config)
