# %%

from tests.common import gpt2_bytes_to_unicode
import json
from pathlib import Path
from train_bpe import train_bpe
# %%


def diff(a, b):
  print(set(a) - set(b))
  print(set(b) - set(a))


gpt2_byte_encoder: dict[int, str] = gpt2_bytes_to_unicode()


def encode(value: bytes, encoder=gpt2_byte_encoder) -> str:
  return "".join([encoder[b] for b in value])

# %%


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
# %%
from train_bpe import (
    get_tokens,
    init_bpe_tokens,
    init_bpe_pairs,
    BpePair,
    merge_tokens,
    diff_tokens,
    SimplePair
)
special_tokens=["<|endoftext|>"]
tokens = get_tokens(input_path, special_tokens)

bpe_tokens = init_bpe_tokens(tokens)
bpe_pairs = init_bpe_pairs(bpe_tokens)
bpe_pairs_dict = {bp.to_simple(): bp for bp in bpe_pairs}
bpe_pair = bpe_pairs_dict.get(SimplePair(first=b'w', second=b'h'))
assert bpe_pair is not None
b" whether" in [from_.origin for from_ in bpe_pair.froms_]

#%%
def decode_utf8_bytes_to_str_wrong(bytestring: bytes):
  return "".join([bytes([b]).decode("utf-8") for b in bytestring])
bytestring = '中国'.encode('utf-8')
decode_utf8_bytes_to_str_wrong(bytestring)
#%%
ord('中')
chr(20013)
# %%
bytestring = b'\xb0\x80'
bytestring.decode('utf-8')
# %%
import regex
PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
PAT = regex.compile(PATTERN)

for x in PAT.finditer("Hello, world! I'm Breezing. 1234 中国"):
  print(x.group())
PAT.findall("Hello, world! I'm Breezing. 1234 中国")
