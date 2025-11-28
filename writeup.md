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



## Problem(tokenizer_experiments):Experimentswithtokenizers


## Problem(transformer_accounting):TransformerLMresourceaccounting
### Consider GPT-2 XL,which has the following configuration. Suppose we constructed our model using this configuration. How many trainable parameters would our model have? Assuming each parameter is represented using single-precision floating point, how much memory is required to just load this model?

  * vocab_size : 50,257
  * context_length : 1,024
  * num_layers : 48
  * d_model : 1,600
  * num_heads : 25
  * d_ff : 6,400
总共需要约2.12b的参数,需要内存68GB
| Part                    | params_num |
| ----------------------- | ---------- |
| Token Embedding         | 80411200   |
| Transformer Block       | 40963200   |
| Transformer Block Total | 1966233600 |
| Final LayerNorm         | 1600       |
| LM Head                 | 80411200   |
| Total Parameters        | 2127057600 |


###  Identify the matrix multiplies required to complete a forward pass of our GPT-2XL-shaped model. How many FLOPs do these matrix multiplies require in total? Assume that our input sequence has context_length tokens.

总共需要约4.2t FLOPs
| Component               | FLOPs         |
| ----------------------- | ------------- |
| Token Embedding         | 0             |
| Transformer Block       | 83.9 billion  |
| Transformer Block Total | 4.0 trillion  |
| Final LayerNorm         | 4.9 million   |
| LM Head                 | 164.7 billion |
| Total FLOPs             | 4.2 trillion  |



### Based on your analysis above, which parts of the model require the most FLOPs?

Transformer Blocks

### Repeat your analysis with GPT-2 small (12 layers, 768 d_model, 12 heads), GPT-2 medium(24 layers, 1024 d_model, 16 heads), and GPT-2 large(36 layers, 1280 d_model, 20 heads). As the model size increases, which parts of the Transformer LM take up proportionally more or less of the total FLOPs?

Transformer Blocks的占比越来越多
Final LayerNorm 和 LM Head的占比越来越少
Small Model:
| Component               | FLOPs         | rate                   |
| ----------------------- | ------------- |
| Token Embedding         | 0             | 0.0                    |
| Transformer Block       | 35.0 billion  | 0.0701450544052423     |
| Transformer Block Total | 420.4 billion | 0.8417406528629076     |
| Final LayerNorm         | 2.4 million   | 4.7233606396060085e-06 |
| LM Head                 | 79.0 billion  | 0.15825462377645277    |
| Total FLOPs             | 499.5 billion | 1                      |

Medium Model:
| Component               | FLOPs         | rate                   |
| ----------------------- | ------------- |
| Token Embedding         | 0             | 0.0                    |
| Transformer Block       | 48.9 billion  | 0.03823065885536095    |
| Transformer Block Total | 1.2 trillion  | 0.9175358125286628     |
| Final LayerNorm         | 3.1 million   | 2.4612012138644385e-06 |
| LM Head                 | 105.4 billion | 0.0824617262701234     |
| Total FLOPs             | 1.3 trillion  | 1                      |

Large Model:
| Component               | FLOPs         | rate                  |
| ----------------------- | ------------- |
| Token Embedding         | 0             | 0.0                   |
| Transformer Block       | 63.8 billion  | 0.026270017199063637  |
| Transformer Block Total | 2.3 trillion  | 0.9457206191662909    |
| Final LayerNorm         | 3.9 million   | 1.620005994022178e-06 |
| LM Head                 | 131.7 billion | 0.054277760827715064  |
| Total FLOPs             | 2.4 trillion  | 1                     |

### Take GPT-2 XL and increase the context length to 16,384. How does the total FLOPs for one forward pass change? How do the relative contribution of FLOPs of the model components change?
总FLOPs从4.2t上升到67.1t。相对占比保持不变。

Original Model:
| Component               | FLOPs         | rate                   |
| ----------------------- | ------------- |
| Token Embedding         | 0             | 0.0                    |
| Transformer Block       | 83.9 billion  | 0.020014844629726546   |
| Transformer Block Total | 4.0 trillion  | 0.9607125422268743     |
| Final LayerNorm         | 4.9 million   | 1.1725615897746376e-06 |
| LM Head                 | 164.7 billion | 0.03928628521153597    |
| Total FLOPs             | 4.2 trillion  | 1                      |

Context Length Model:
| Component               | FLOPs         | rate                   |
| ----------------------- | ------------- |
| Token Embedding         | 0             | 0.0                    |
| Transformer Block       | 1.3 trillion  | 0.020014844629726546   |
| Transformer Block Total | 64.4 trillion | 0.9607125422268743     |
| Final LayerNorm         | 78.6 million  | 1.1725615897746376e-06 |
| LM Head                 | 2.6 trillion  | 0.03928628521153597    |
| Total FLOPs             | 67.1 trillion | 1                      |
