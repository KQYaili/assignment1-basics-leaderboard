import numpy as np
from cs336_basics.tokenizer import train_bpe, Tokenizer
import pickle
import os
from pathlib import Path

# 自动推导项目根目录和 data 目录绝对路径
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"

def tokenize_file(in_path, out_path, tokenizer):
    with open(out_path, 'wb') as out_f:
        with open(in_path, 'r', encoding='utf-8') as f:
            buffer = []
            for token_id in tokenizer.encode_iterable(f):
                buffer.append(token_id)
                if len(buffer)>=100000:
                    arr=np.array(buffer, dtype=np.uint16)
                    out_f.write(arr.tobytes())
                    buffer = []
            if buffer:
                arr=np.array(buffer, dtype=np.uint16)
                out_f.write(arr.tobytes())

if __name__ == "__main__":
    special_tokens = ["<|endoftext|>"]
    if os.path.exists(DATA_DIR/"owt_bpe.pkl"):
        with open(DATA_DIR/"owt_bpe.pkl", "rb") as f:
            vocab, merge = pickle.load(f)
    else:
        print("未找到词表，正在训练 BPE")
        vocab,merge=train_bpe(DATA_DIR/"owt_train_sample.txt", 32000, special_tokens)

        with open(DATA_DIR/"owt_bpe.pkl", 'wb') as f:
            pickle.dump((vocab,merge), f)

    tokenizer = Tokenizer(vocab,merge,special_tokens)
    print("正在编码验证集...")
    tokenize_file(DATA_DIR/"owt_valid.txt", DATA_DIR/"owt_valid.bin", tokenizer)
    print("验证集编码完成！")

    print("正在编码训练集...")
    tokenize_file(DATA_DIR/"owt_train_sample.txt", DATA_DIR/"owt_train.bin", tokenizer)
    print("训练集编码完成！")
