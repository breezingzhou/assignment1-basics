# %%
# 处理从dataset到tokenizer的准备工作
# 取代了writeup/writeup.py 和 bpe_encode.py
from collections.abc import Callable
from dataclasses import dataclass
import gzip
from pathlib import Path
import sys
import hydra
from hydra.utils import to_absolute_path
from unitoken import BpeTrainer, BpeEncoder, PreTokenizer
from common import CONFIG_DIR
import numpy as np
import requests

from scripts.experiment_config import Config, EnvConfig
# %%
# %%


@dataclass
class PrepareConfig:
  dataset_name: str
  vocab_size: int
  special_tokens: list[str]
  num_chunks: int
  download_fn: Callable[["PrepareConfig"], None]
  workspace: Path

  @property
  def data_dir(self) -> Path:
    return self.workspace / "data"

  @property
  def output_dir(self) -> Path:
    return self.workspace / "output"

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


def download_tinystory(config: PrepareConfig):
  train_url = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt"
  valid_url = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt"
  train_file = config.data_dir / "TinyStoriesV2-GPT4_train.txt"
  valid_file = config.data_dir / "TinyStoriesV2-GPT4_valid.txt"
  download_file(train_url, train_file)
  download_file(valid_url, valid_file)


def download_owt(config: PrepareConfig):
  train_url = "https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_train.txt.gz"
  valid_url = "https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_valid.txt.gz"
  train_gzfile = config.data_dir / "owt_train.txt.gz"
  valid_gzfile = config.data_dir / "owt_valid.txt.gz"
  train_file = config.data_dir / "owt_train.txt"
  valid_file = config.data_dir / "owt_valid.txt"
  download_file(train_url, train_gzfile)
  download_file(valid_url, valid_gzfile)
  decompress_gz_file(train_gzfile, train_file)
  decompress_gz_file(valid_gzfile, valid_file)


def download_dataset(config: PrepareConfig):
  config.download_fn(config)


def bpe_train(config: PrepareConfig):
  # 生成vocab和merges文件
  input_file = config.data_dir / f"{config.dataset_name}_train.txt"
  vocab_file = config.output_dir / f"vocab.{config.dataset_name}[u8].json"
  merges_file = config.output_dir / f"merges.{config.dataset_name}[u8].txt"
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
      outdir=config.output_dir
  )
  print("BPE training completed")


def bpe_encode(config: PrepareConfig):
  encoder = None

  for group in ["train", "valid"]:
    input_file = config.data_dir / f"{config.dataset_name}_{group}.txt"
    output_file = config.output_dir / f"idxs.{input_file.stem}.npy"
    if output_file.exists():
      print(
          f"Encoded idxs file already exists at {output_file}, skipping BPE encoding for {group}.")
      continue

    print(f"Starting BPE encoding for {config.dataset_name} {group}")
    if encoder is None:
      encoder = BpeEncoder.load(name=config.dataset_name, input_dir=config.output_dir)
    idxs = encoder.encode_file(
        path=input_file,
        num_chunks=config.num_chunks
    )
    print("BPE encoding completed, start saving")
    np.save(output_file, idxs)
    print(f"Saved encoded idxs to {output_file}")


# %%
tiny_story_config = PrepareConfig(
    dataset_name="TinyStoriesV2-GPT4",
    vocab_size=10000,
    special_tokens=["<|endoftext|>"],
    num_chunks=64,
    download_fn=download_tinystory,
    workspace=Path(__file__).parent,
)

owt_config = PrepareConfig(
    dataset_name="owt",
    vocab_size=32000,
    special_tokens=["<|endoftext|>"],
    num_chunks=256,
    download_fn=download_owt,
    workspace=Path(__file__).parent,
)
name_2_config = {
    "TinyStoriesV2-GPT4": tiny_story_config,
    "owt": owt_config,
}
# %%
# bpe_train(config=tiny_story_config)
# bpe_encode(config=tiny_story_config)
# %%


@hydra.main(version_base=None, config_path=str(CONFIG_DIR), config_name="config")
def main(cfg):
  workspace = Path(to_absolute_path(cfg.env.workspace))
  env = EnvConfig(workspace=workspace)
  kwargs = dict(cfg)
  kwargs['env'] = env
  config = Config(**kwargs)

  prepare_config = name_2_config[config.experiment.dataset_name]
  prepare_config.workspace = workspace
  prepare_config.output_dir.mkdir(parents=True, exist_ok=True)

  download_dataset(prepare_config)
  bpe_train(prepare_config)
  bpe_encode(prepare_config)


# %%
if __name__ == "__main__":
  main()
