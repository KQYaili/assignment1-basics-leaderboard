import collections
from typing import Iterable, Iterator
import regex
from twisted import pair


def merge_word(word, p1, p2, merged):
    # 用一个 while 循环遍历 word 的下标
    new_word=[]
    idx=0
    while idx < len(word):
        # 如果当前位置是 p1 且下一个是 p2，把 merged 放进去，下标 + 2
        if idx<len(word) -1 and word[idx]== p1 and word[idx + 1] == p2:
            new_word.append(merged)
            idx+=2
        # 否则把当前元素放进去，下标 + 1
        else:
            new_word.append(word[idx])
            idx+=1
    # 返回组装好的新元组
    return tuple(new_word)

def train_bpe(input_path, vocab_size, special_tokens=None):
    with open(input_path, 'r', encoding='utf-8') as f:
        text = f.read()
    if special_tokens:
        # 1. 对每个特殊 token 内部的特殊字符进行转义，然后用正则的 "|" (或) 串联起来
        pattern = "|".join(regex.escape(tok) for tok in special_tokens)

        # 2. 用这个正则将全文切开成段落列表
        chunks = regex.split(pattern, text)
    else:
        # 如果没有特殊 token，整篇文本就是一个完整的 chunk
        chunks = [text]
    special_tokens = special_tokens or []
    vocab = {i: bytes([i]) for i in range(256)}
    for i, sym in enumerate(special_tokens):
        vocab[256 + i] = sym.encode('utf-8')
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    compiled_pat =regex.compile(PAT)
    word_counts = collections.defaultdict(int)
    merges=[]
    for chunk in chunks:
        for match in compiled_pat.finditer(chunk):
            # 1. 拿到匹配文本
            word_str = match.group(0)
            # 2. 转成单字节元组
            word_tuple = tuple(bytes([b]) for b in word_str.encode("utf-8"))
            # 3. 统计词频
            word_counts[word_tuple] += 1
    while len(vocab) < vocab_size:
        if len(vocab) % 500 == 0:
            print(f"BPE 训练进度: 当前词表大小 {len(vocab)} / {vocab_size}")
        pair_counts = collections.defaultdict(int)
        for (word, count) in word_counts.items():
            for j in range(len(word) - 1):
                pair_counts[(word[j], word[j + 1])] += count
        if not pair_counts:
            break
        best_pair = max(pair_counts.keys(), key=lambda p: (pair_counts[p], p))
        p1, p2 = best_pair
        merged = p1 + p2
        merges.append(best_pair)
        new_word_counts = {}
        for word, count in word_counts.items():
            new_word = merge_word(word, p1, p2, merged)
            new_word_counts[new_word] = count
        word_counts = new_word_counts
        vocab[len(vocab)]=merged
    return vocab, merges

class Tokenizer:
    def __init__(self, vocab, merges, special_tokens=None):
        self.vocab=vocab
        self.bytes_to_id={word:index for index,word in vocab.items()}
        self.pat=regex.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
        self.special_tokens_pat=None
        self.special_tokens = set()
        if special_tokens:
            for token in special_tokens:
                token_en=token.encode('utf-8')
                if token_en not in self.bytes_to_id:
                    new_id = len(self.vocab)
                    self.bytes_to_id[token_en] = new_id
                    self.vocab[new_id]=token_en
            special_tokens_sorted = sorted(special_tokens,key=lambda x: len(x),reverse=True)
            special_tokens_pat="("+"|".join(regex.escape(token) for token in special_tokens_sorted)+")"
            self.special_tokens_pat = regex.compile(special_tokens_pat)
            self.special_tokens = set(special_tokens)
        self.bpe_ranks={items:i for i,items in enumerate(merges)}
        self.cache = {}

    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        pass

    def encode(self, text: str) -> list[int]:
        res=[]
        if self.special_tokens_pat:
            chunks= self.special_tokens_pat.split(text)
        else:
            chunks = [text]
        for chunk in chunks:
            if not chunk:
                continue
            if chunk in self.special_tokens:
                res.append(self.bytes_to_id[chunk.encode('utf-8')])
            else:
                for match in self.pat.finditer(chunk):
                    # 1. 拿到匹配文本
                    word_str = match.group(0)
                    if word_str in self.cache:
                        res.extend(self.cache[word_str])
                        continue
                    # 2. 转成单字节元组
                    word_tuple = tuple(bytes([b]) for b in word_str.encode("utf-8"))
                    while True:
                        rank=None
                        pair=None
                        merge=None
                        for i in range(len(word_tuple) - 1):
                            if (word_tuple[i], word_tuple[i+1]) in self.bpe_ranks:
                                if rank is None or rank>self.bpe_ranks[(word_tuple[i], word_tuple[i+1])]:
                                    rank=self.bpe_ranks[(word_tuple[i], word_tuple[i+1])]
                                    pair=(word_tuple[i], word_tuple[i+1])
                                    merge=word_tuple[i]+word_tuple[i+1]
                        if rank is not None:
                            word_tuple=merge_word(word_tuple, pair[0], pair[1], merge)
                        else:
                            break
                    ids=[self.bytes_to_id[word] for word in word_tuple]
                    res.extend(ids)
                    self.cache[word_str]=ids
        return res

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for chunk in iterable:
            if chunk:
                yield from self.encode(chunk)

    def decode(self, ids: list[int]) -> str:
        return (b"".join([self.vocab[id] for id in ids])).decode("utf-8", errors="replace")