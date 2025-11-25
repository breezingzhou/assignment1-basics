## Problem (unicode1): Understanding Unicode
### What Unicode character does chr(0) return
chr(0) returns '\x00'
### How does this character’s string representation (__repr__()) differ from its printed representation?

* string representation call `__repr__()`  while printed representation call `__str__()`
* string representation is for developer while printed representation is for user

### What happens when this character occurs in text? It may be helpful to play around with the following in your Python interpreter and see if it matches your expectations:
| character                                   | occurs                     |
| ------------------------------------------- | -------------------------- |
| chr(0)                                      | '\x00'                     |
| print(chr(0))                               |                            |
| "this is a test" + chr(0) + "string"        | 'this is a test\x00string' |
| print("this is a test" + chr(0) + "string") | this is a teststring       |



## Problem (unicode2): Unicode Encodings
### What are some reasons to prefer training our tokenizer on UTF-8 encoded bytes, rather than UTF-16 or UTF-32? It may be helpful to compare the output of these encodings for various input strings.
* UTF-8 可变长度1-4字节 对于英文占用1字节 相比另外两个更紧凑 减少存储和计算压力
* UTF-8 更通用 更多的基础软件支持utf-8格式


### Consider the following (incorrect) function, which is intended to decode a UTF-8 byte string into a Unicode string. Why is this function incorrect? Provide an example of an input byte string that yields incorrect results.
```python
def decode_utf8_bytes_to_str_wrong(bytestring: bytes):
  return "".join([bytes([b]).decode("utf-8") for b in bytestring])
```
uft-8编码是变长编码，可变长度1-4字节，有可能多个字节才对应一个unicode字符。 example: b'\xe4\xb8\xad\xe5\x9b\xbd'


### Give a two byte sequence that does not decode to any Unicode character(s).
b'\xb0\x80' utf-8编码长度为2时 首字节范围为(0xC0–0xDF) 次字节范围为(0x80–0xBF)

## Problem(train_bpe_tinystories):BPE Training on TinyStories
### Train a byte-level BPE tokenizer on the TinyStories dataset,using a maximum vocabulary size of 10,000. Make sure to add the TinyStories `<|endoftext|>` special token to thevocabulary. Serialize the resulting vocabulary and merges to disk for further inspection. How many hours and memory did training take? What is the longest token in the vocabulary? Does it make sense?<br>
<b>Resource requirements</b>: ≤ 30minutes (noGPUs) , ≤ 30GB RAM<br>
<b>Hint</b> You should be able to get under 2 minutes for BPE training using multiprocessing during pretokenization and the following twofacts:
* (a) The `<|endoftext|>` token delimits documents in the data files.
* (b) The `<|endoftext|>` token is handled as a special case before the BPE merges are applied.

耗时: 194s  消耗内存：1.17G
最长的tokens为 ["Ġaccomplishment", "Ġdisappointment", "Ġresponsibility"]
最长token是在“压缩频率”和“粒度粗细”之间找到的平衡点。如果最长词过短，说明词被切分的很碎，语义丢失，序列变长。如果最长词过长，词汇表冗余，泛化能力下降。


### Profile your code. What part of the tokenizer training process takes the most time?
BPE Training最耗时的部分是 选择出现频率最高的pair  (max函数)


## Problem(train_bpe_expts_owt):BPE Training on OpenWebText
### Train a byte-level BPE tokenizer on the OpenWebText dataset, using a maximum vocabulary size of 32,000. Serializethe resulting vocabulary and merges to disk for further inspection.What is the longest token in the vocabulary? Does it make sense?<br>
<b>Resource requirements</b>: ≤ 12hours (noGPUs), ≤ 100GB RAM


### Compare and contrast the tokenizer that you get training on TinyStories versus OpenWebText.
