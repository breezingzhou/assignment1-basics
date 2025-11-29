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

rot_matrix = None

# %%
import torch
from einops import rearrange, repeat

l = 3
size = 2 * l

eye_l = torch.eye(l)
block_eye_t = repeat(eye_l, 'i j -> i h j w', h=2, w=2)  # 形状: (l, 2, l, 2)
block_eye = rearrange(block_eye_t, 'i h j w -> (i h) (j w)')  # 形状: (2l, 2l)

# %%
import torch
from einops import repeat

x = torch.arange(8).reshape(4, 2)  # tensor([[0, 1], [2, 3], [4, 5], [6, 7]])
x_expanded = repeat(x, 'h w -> h (repeat w)', repeat=2)

print("原始数据:\n", x)
print("扩展后数据:\n", x_expanded)
# %%
import torch
x = torch.tensor([[1, 3, 2], [4, 1, 5]])
max_vals = x.max(dim=0, keepdim=True)

x2 = x - max_vals.values

x3 = x2.exp()
x4 = x3.sum(dim=0, keepdim=True)
x3 / x4

# %%
import torch
from cs336_basics.nn_utils import my_softmax
import torch.nn.functional as F
inputs = torch.tensor(
    [
        [
            [0.1088, 0.1060, 0.6683, 0.5131, 0.0645],
            [0.4538, 0.6852, 0.2520, 0.3792, 0.2675],
            [0.4578, 0.3357, 0.6384, 0.0481, 0.5612],
            [0.9639, 0.8864, 0.1585, 0.3038, 0.0350],
        ],
        [
            [0.3356, 0.9013, 0.7052, 0.8294, 0.8334],
            [0.6333, 0.4434, 0.1428, 0.5739, 0.3810],
            [0.9476, 0.5917, 0.7037, 0.2987, 0.6208],
            [0.8541, 0.1803, 0.2054, 0.4775, 0.8199],
        ],
    ]
)
inputs = inputs * 1000
targets = torch.tensor([[1, 0, 2, 2], [4, 1, 4, 0]])
inputs = inputs.view(-1, inputs.size(-1))
targets = targets.view(-1)
expected = F.cross_entropy(inputs.view(-1, inputs.size(-1)), targets.view(-1))

# q = my_softmax(inputs, dim=-1)

batch_indices = torch.arange(inputs.size(0), device=inputs.device)
# -torch.log(q)[batch_indices, targets].mean()

max_ = inputs.max(dim=-1, keepdim=True)

logsumexp = torch.log(torch.exp(inputs - max_.values).sum(dim=-1, keepdim=True)) + max_.values
(inputs - logsumexp)[batch_indices, targets].mean()
#%%
import numpy as np
import torch
dataset = np.arange(0, 100)
context_length = 7
batch_size = 32
device = "cpu"

data = np.lib.stride_tricks.sliding_window_view(dataset, context_length+1)
choice = np.random.choice(data.shape[0], size=batch_size, replace=False)
choosed_data = data[choice]
r =  torch.from_numpy(choosed_data).to(device)
r[::, :-1], r[:, 1:]
#%%
import torch
from cs336_basics.optimizer import MyAdamW

model = torch.nn.Linear(10, 2)
adamw = MyAdamW(model.parameters(), lr=1e-3, weight_decay=0.01, betas=(0.9, 0.999), eps=1e-8)
adamw.state_dict
