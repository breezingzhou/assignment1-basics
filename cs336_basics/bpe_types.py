from dataclasses import dataclass
from enum import Enum


@dataclass
class SimplePair:
  first: bytes
  second: bytes

  def concat(self) -> bytes:
    return self.first + self.second

  def __eq__(self, other: "SimplePair") -> bool:
    return self.first == other.first and self.second == other.second

  def __hash__(self) -> int:
    return hash((self.first, self.second))

  def to_pair(self) -> tuple[bytes, bytes]:
    return (self.first, self.second)


@dataclass
class BpeToken:
  origin: bytes
  count: int
  tokens: list[bytes]

  def __repr__(self) -> str:
    return f"BpeToken({self.origin!r}, count={self.count}, tokens={self.tokens})"

  def __hash__(self) -> int:
    return hash(self.origin)


@dataclass
class BpePair:
  first: bytes
  second: bytes
  count: int
  froms_: set[BpeToken]

  def __lt__(self, other: "BpePair") -> bool:
    if self.count != other.count:
      return self.count < other.count
    if self.first != other.first:
      return self.first < other.first
    return self.second < other.second

  def to_pair(self) -> tuple[bytes, bytes]:
    return (self.first, self.second)

  def to_simple(self) -> SimplePair:
    return SimplePair(self.first, self.second)

  def __repr__(self) -> str:
    return f"BpePair({self.first!r}, {self.second!r}, count={self.count})\n  from_: {self.froms_}"

# %%


class ContentType(Enum):
  NORMAL = "normal"
  SPECIAL = "special"


class Content:
  type: ContentType
  content: str

  def __init__(self, type_: ContentType, content: str):
    self.type = type_
    self.content = content

  def __repr__(self) -> str:
    return f"Content(type={self.type}, content={self.content!r})"

  @classmethod
  def special(cls, content: str) -> "Content":
    return cls(ContentType.SPECIAL, content)

  @classmethod
  def normal(cls, content: str) -> "Content":
    return cls(ContentType.NORMAL, content)
