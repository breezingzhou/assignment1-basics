# %%
from collections import Counter
import regex
from cs336_basics.bpe_types import BpeToken, BpePair, SimplePair, Content, ContentType

# $$
PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
PAT = regex.compile(PATTERN)


# %%

# ['a', 'b', 'c'] , SimplePair('a', 'b') -> ['ab', 'c']
def merge_tokens(tokens: list[bytes], target_pair: SimplePair) -> list[bytes]:
  merged_tokens: list[bytes] = []
  i = 0
  while i < len(tokens):
    if i < len(tokens) - 1 and tokens[i:i + 2] == [target_pair.first, target_pair.second]:
      merged_tokens.append(target_pair.concat())
      i += 2
    else:
      merged_tokens.append(tokens[i])
      i += 1
  return merged_tokens

# %%

# ['a', 'ab', 'c'], SimplePair('a', 'b')
# to_add = [SimplePair('ab', 'c'), SimplePair('a', 'ab')]
# to_remove = [SimplePair('a', 'a'), SimplePair('b', 'c')])


def diff_tokens(new_tokens: list[bytes], target_pair: SimplePair):
  to_add: list[SimplePair] = []
  to_remove: list[SimplePair] = []
  concat = target_pair.concat()
  for index, token in enumerate(new_tokens):
    if token != concat:
      continue
    pre_index = index - 1
    post_index = index + 1
    if pre_index >= 0 and new_tokens[pre_index] != concat:
      to_add.append(SimplePair(new_tokens[pre_index], concat))
      to_remove.append(SimplePair(new_tokens[pre_index], target_pair.first))
    if post_index < len(new_tokens):
      to_add.append(SimplePair(concat, new_tokens[post_index]))
      # 如果后一个token不是concat
      if new_tokens[post_index] != concat:
        to_remove.append(SimplePair(target_pair.second, new_tokens[post_index]))
      # 后一个也是合并生成的
      else:
        to_remove.append(SimplePair(target_pair.second, target_pair.first))
  return to_add, to_remove

# %%


def count_apperence(tokens: list[bytes], pair: SimplePair) -> int:
  count = 0
  for first, second in zip(tokens, tokens[1:]):
    if first == pair.first and second == pair.second:
      count += 1
  return count

# %%


def split_special_tokens_training(data: str, special_tokens: list[str]):
  tokens = Counter[str]()
  for content in regex.splititer("|".join(regex.escape(token) for token in special_tokens), data):
    for token in PAT.findall(content):
      tokens[token] += 1
  return tokens


def split_special_tokens_tokenizer(text: str, special_tokens: list[str]) -> list[Content]:
  parts: list[Content] = []
  pattern = "|".join(regex.escape(special_token) for special_token in special_tokens)
  last_pos = 0
  for match_ in regex.finditer(pattern, text):
    if match_.start() > last_pos:
      parts.append(Content.normal(text[last_pos:match_.start()]))
    parts.append(Content.special(match_.group()))
    last_pos = match_.end()
  # 剩余普通文本
  if last_pos < len(text):
    parts.append(Content.normal(text[last_pos:]))
  return parts
