# %%
from train_bpe import train_bpe, get_tokens_v2
from pathlib import Path
from tests.common import gpt2_bytes_to_unicode
import json
# %%


# %%


def save_result(vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], vocab_path: Path, merges_path: Path):
  gpt2_byte_encoder = gpt2_bytes_to_unicode()
  vocab_display = {
      index: ''.join([gpt2_byte_encoder[b] for b in token]) for index, token in vocab.items()
  }
  merges_display = [(''.join([gpt2_byte_encoder[b] for b in m1]),
                    ''.join([gpt2_byte_encoder[b] for b in m2])) for m1, m2 in merges]
  with open(vocab_path, "w") as f:
    json.dump(vocab_display, f, indent=2, ensure_ascii=False)
  with open(merges_path, "w") as f:
    for m1, m2 in merges_display:
      f.write(f"{m1} {m2}\n")


# %%

def get_longest_token(vocab_path: Path):
  with open(vocab_path, "r") as f:
    vocab_loaded: dict[int, str] = json.load(f)

  max_len = 0
  candidates = []
  for token in vocab_loaded.values():
    if len(token) > max_len:
      max_len = len(token)
      candidates = [token]
    elif len(token) == max_len:
      candidates.append(token)
  print("Max token length:", max_len)
  print("Candidates:", candidates)


# %%
def TinyStoriesV2():
  input_file = Path(__file__).parent.parent / "data/TinyStoriesV2-GPT4-train.txt"
  vocab_size = 10000
  special_tokens = ["<|endoftext|>"]
  num_chunks = 32
  num_processes = 8

  vocab_path = Path(__file__).parent / "TinyStoriesV2_vocab.json"
  merges_path = Path(__file__).parent / "TinyStoriesV2_merges.txt"
  # %%
  vocab, merges = train_bpe(input_file, vocab_size, special_tokens, num_chunks=num_chunks,
                            num_processes=num_processes)
  # save_result(vocab, merges, vocab_path, merges_path)
  # get_longest_token(vocab_path)
# %%


def owt():
  input_file = Path(__file__).parent.parent / "data/owt_train.txt"
  vocab_size = 32000
  special_tokens = ["<|endoftext|>"]
  num_chunks = 128
  num_processes = 16

  vocab_path = Path(__file__).parent / "owt_vocab.json"
  merges_path = Path(__file__).parent / "owt_merges.txt"
  vocab, merges = train_bpe(input_file, vocab_size, special_tokens, num_chunks=num_chunks,
                            num_processes=num_processes)
  # save_result(vocab, merges, vocab_path, merges_path)
  # get_longest_token(vocab_path)

# %%
TinyStoriesV2()
