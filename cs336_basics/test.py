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
special_tokens = ["<|endoftext|>"]
tokens = get_tokens(input_path, special_tokens)

bpe_tokens = init_bpe_tokens(tokens)
bpe_pairs = init_bpe_pairs(bpe_tokens)
bpe_pairs_dict = {bp.to_simple(): bp for bp in bpe_pairs}
bpe_pair = bpe_pairs_dict.get(SimplePair(first=b'w', second=b'h'))
assert bpe_pair is not None
b" whether" in [from_.origin for from_ in bpe_pair.froms_]

# %%


def decode_utf8_bytes_to_str_wrong(bytestring: bytes):
  return "".join([bytes([b]).decode("utf-8") for b in bytestring])


bytestring = '中国'.encode('utf-8')
decode_utf8_bytes_to_str_wrong(bytestring)
# %%
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
# %%
special_tokens = ["<|endoftext|>", "<|endoftext|><|endoftext|>"]
special_tokens2 = special_tokens[::]
special_tokens2.sort(key=lambda x: len(x), reverse=True)
# %%
import torch
from einops import einsum, reduce, rearrange
d_model = 3
x = torch.Tensor([[1, 2, 3], [4, 5, 6]])
weight = torch.Tensor([1, 1, 1])
eps = 1e-5
norm = reduce(x.pow(2), '... d -> ...', "mean")
rms = torch.sqrt_(norm + eps)
rms = rearrange(rms, '... -> ... 1')
x_normalized = x / rms
result = einsum(x_normalized, weight, "... d_model, d_model -> ... d_model")
# %%
import torch
from einops import einsum, reduce, rearrange, repeat
max_seq_len = 7
d_k = 4
d = d_k // 2
theta = 2
i = torch.arange(1, 1 + max_seq_len)
k = torch.arange(0, d_k // 2, )
theta_k = 1 / (theta ** (2 * k / d_k))
theta_ik = einsum(i, theta_k, "i, k -> i k")  # Shape: (max_seq_len, d_k/2)

cos_val = torch.cos(theta_ik)
sin_val = torch.sin(theta_ik)

rot_submat = torch.zeros(*theta_ik.shape, 2, 2)
rot_submat[..., 0, 0] = cos_val  # [d] → [d,1,1]
rot_submat[..., 0, 1] = -sin_val
rot_submat[..., 1, 0] = sin_val
rot_submat[..., 1, 1] = cos_val

rot_submat_reshape = rearrange(rot_submat, 'i d h w -> i h (d w)')
rot_submat_repeat = repeat(rot_submat_reshape, 'i h d_k -> i (d h) d_k', d=d)
eye = torch.eye(d_k // 2)
block_eye = repeat(eye, 'i j -> i h j w', h=2, w=2)
block_eye = rearrange(block_eye, 'i h j w -> (i h) (j w)')  # 形状: (2l, 2l)

rot_matrix  = None

#%%
import torch
from einops import rearrange, repeat

l = 3
size = 2 * l

eye_l = torch.eye(l)
block_eye_t = repeat(eye_l, 'i j -> i h j w', h=2, w=2)  # 形状: (l, 2, l, 2)
block_eye = rearrange(block_eye_t, 'i h j w -> (i h) (j w)')  # 形状: (2l, 2l)

#%%
import torch
from einops import repeat

x = torch.arange(8).reshape(4, 2)  # tensor([[0, 1], [2, 3], [4, 5], [6, 7]])
x_expanded = repeat(x, 'h w -> h (repeat w)', repeat=2)

print("原始数据:\n", x)
print("扩展后数据:\n", x_expanded)
