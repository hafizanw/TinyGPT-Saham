import os
import re
import json
import time
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformer_blocks import Block

print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("GPU name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")


# =====================================================================
# 1. KONFIGURASI TOKENIZER
# =====================================================================
# Ganti nilai di bawah untuk mencoba tokenizer yang berbeda.
# Pilihan valid: "char", "word", "subword", "bpe", "wordpiece",
#                "sentencepiece", "tiktoken"
TOKENIZER_TYPE = "subword"

CORPUS_PATH = "corpus_saham_indonesia.txt"
TOKENIZERS_DIR = "tokenizers"
os.makedirs(TOKENIZERS_DIR, exist_ok=True)

# Target ukuran vocabulary untuk tokenizer yang di-training dari corpus
# (subword, bpe, wordpiece, sentencepiece). Untuk char & word, ukuran
# vocab ditentukan otomatis oleh isi corpus, bukan oleh nilai ini.
TARGET_VOCAB_SIZE = 40


def _safe_vocab_size(text, requested_vocab_size, buffer=10):
    """
    Trainer seperti SentencePiece / BPE / WordPiece / Unigram akan
    error jika vocab_size yang diminta lebih kecil dari jumlah karakter
    unik di corpus + token spesial. Fungsi ini otomatis menaikkan
    vocab_size bila corpus terlalu kecil/variatif, supaya training
    tidak crash di corpus kecil.
    """
    n_unique_chars = len(set(text))
    return max(requested_vocab_size, n_unique_chars + buffer)


# =====================================================================
# 2. INTERFACE TOKENIZER SERAGAM
# =====================================================================
# Semua tokenizer di bawah WAJIB menyediakan atribut/metode yang sama:
#   - self.vocab_size  : int
#   - self.ids         : List[int]  (hasil encode seluruh corpus)
#   - self.encode(text): List[int]
#   - self.decode(ids) : str
# Dengan begitu, kode TinyGPT (arsitektur model, training loop, generate)
# TIDAK PERLU TAHU tokenizer mana yang sedang dipakai.

class BaseTokenizer:
    name = "base"

    def encode(self, text):
        raise NotImplementedError

    def decode(self, ids):
        raise NotImplementedError


class CharTokenizer(BaseTokenizer):
    """1. Character Tokenization -- setiap karakter adalah satu token."""
    name = "char"

    def __init__(self, text):
        chars = sorted(set(text))
        if "<unk>" not in chars:
            chars = chars + ["<unk>"]
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for ch, i in self.stoi.items()}
        self.vocab_size = len(self.stoi)
        self.ids = self.encode(text)
        self._save_vocab()

    def encode(self, text):
        unk = self.stoi["<unk>"]
        return [self.stoi.get(ch, unk) for ch in text]

    def decode(self, ids):
        return "".join(self.itos.get(i, "") for i in ids)

    def _save_vocab(self):
        path = os.path.join(TOKENIZERS_DIR, "char_vocab.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.stoi, f, ensure_ascii=False, indent=2)
        print(f"[CharTokenizer] vocab disimpan ke {path}")


class WordTokenizer(BaseTokenizer):
    """2. Word Tokenization -- setiap kata (dan tanda baca) adalah satu token."""
    name = "word"
    _pattern = re.compile(r"\w+|[^\w\s]")

    def __init__(self, text):
        words = self._pattern.findall(text)
        vocab = ["<unk>"] + sorted(set(words))
        self.stoi = {w: i for i, w in enumerate(vocab)}
        self.itos = {i: w for w, i in self.stoi.items()}
        self.vocab_size = len(self.stoi)
        self.ids = self.encode(text)
        self._save_vocab()

    def encode(self, text):
        unk = self.stoi["<unk>"]
        tokens = self._pattern.findall(text)
        return [self.stoi.get(t, unk) for t in tokens]

    def decode(self, ids):
        return " ".join(self.itos.get(i, "<unk>") for i in ids)

    def _save_vocab(self):
        path = os.path.join(TOKENIZERS_DIR, "word_vocab.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.stoi, f, ensure_ascii=False, indent=2)
        print(f"[WordTokenizer] vocab disimpan ke {path}")


class SubwordTokenizer(BaseTokenizer):
    """
    3. Subword Tokenization (umum) -- menggunakan algoritma Unigram
    dari library HuggingFace `tokenizers`, dipasangkan dengan
    Metaspace pre-tokenizer (gaya "_" pengganti spasi, seperti pada
    ALBERT/XLNet) agar spasi bisa direkonstruksi saat decode.
    """
    name = "subword"

    def __init__(self, text, corpus_path=CORPUS_PATH, vocab_size=TARGET_VOCAB_SIZE):
        from tokenizers import Tokenizer
        from tokenizers.models import Unigram
        from tokenizers.trainers import UnigramTrainer
        from tokenizers.pre_tokenizers import Metaspace
        from tokenizers.decoders import Metaspace as MetaspaceDecoder

        vocab_size = _safe_vocab_size(text, vocab_size)

        tok = Tokenizer(Unigram())
        tok.pre_tokenizer = Metaspace()
        tok.decoder = MetaspaceDecoder()
        trainer = UnigramTrainer(
            vocab_size=vocab_size, unk_token="<unk>", special_tokens=["<unk>"]
        )
        tok.train([corpus_path], trainer)

        self._tok = tok
        self.vocab_size = tok.get_vocab_size()
        self.ids = self.encode(text)
        self._save_vocab()

    def encode(self, text):
        return self._tok.encode(text).ids

    def decode(self, ids):
        return self._tok.decode(ids)

    def _save_vocab(self):
        path = os.path.join(TOKENIZERS_DIR, "subword_tokenizer.json")
        self._tok.save(path)
        print(f"[SubwordTokenizer] tokenizer disimpan ke {path}")


class BPETokenizer(BaseTokenizer):
    """
    4. Byte Pair Encoding (BPE) -- gaya GPT-2, menggunakan library
    HuggingFace `tokenizers` dengan ByteLevel pre-tokenizer/decoder
    (beda dari SentencePiece di bawah, yang juga bisa mode "bpe" tapi
    diimplementasikan oleh library `sentencepiece`).
    """
    name = "bpe"

    def __init__(self, text, corpus_path=CORPUS_PATH, vocab_size=TARGET_VOCAB_SIZE):
        from tokenizers import Tokenizer
        from tokenizers.models import BPE
        from tokenizers.trainers import BpeTrainer
        from tokenizers.pre_tokenizers import ByteLevel
        from tokenizers.decoders import ByteLevel as ByteLevelDecoder

        vocab_size = _safe_vocab_size(text, vocab_size)

        tok = Tokenizer(BPE(unk_token="<unk>"))
        tok.pre_tokenizer = ByteLevel(add_prefix_space=False)
        tok.decoder = ByteLevelDecoder()
        trainer = BpeTrainer(vocab_size=vocab_size, special_tokens=["<unk>"])
        tok.train([corpus_path], trainer)

        self._tok = tok
        self.vocab_size = tok.get_vocab_size()
        self.ids = self.encode(text)
        self._save_vocab()

    def encode(self, text):
        return self._tok.encode(text).ids

    def decode(self, ids):
        return self._tok.decode(ids)

    def _save_vocab(self):
        path = os.path.join(TOKENIZERS_DIR, "bpe_tokenizer.json")
        self._tok.save(path)
        print(f"[BPETokenizer] tokenizer disimpan ke {path}")


class WordPieceTokenizer(BaseTokenizer):
    """5. WordPiece -- gaya BERT, menggunakan library HuggingFace `tokenizers`."""
    name = "wordpiece"

    def __init__(self, text, corpus_path=CORPUS_PATH, vocab_size=TARGET_VOCAB_SIZE):
        from tokenizers import Tokenizer
        from tokenizers.models import WordPiece
        from tokenizers.trainers import WordPieceTrainer
        from tokenizers.pre_tokenizers import Whitespace
        from tokenizers.decoders import WordPiece as WordPieceDecoder

        vocab_size = _safe_vocab_size(text, vocab_size)

        tok = Tokenizer(WordPiece(unk_token="[UNK]"))
        tok.pre_tokenizer = Whitespace()
        tok.decoder = WordPieceDecoder()
        trainer = WordPieceTrainer(vocab_size=vocab_size, special_tokens=["[UNK]"])
        tok.train([corpus_path], trainer)

        self._tok = tok
        self.vocab_size = tok.get_vocab_size()
        self.ids = self.encode(text)
        self._save_vocab()

    def encode(self, text):
        return self._tok.encode(text).ids

    def decode(self, ids):
        return self._tok.decode(ids)

    def _save_vocab(self):
        path = os.path.join(TOKENIZERS_DIR, "wordpiece_vocab.txt")
        vocab = sorted(self._tok.get_vocab().items(), key=lambda kv: kv[1])
        with open(path, "w", encoding="utf-8") as f:
            for token_str, _ in vocab:
                f.write(token_str + "\n")
        print(f"[WordPieceTokenizer] vocab disimpan ke {path}")


class SentencePieceTokenizer(BaseTokenizer):
    """
    6. SentencePiece -- library resmi Google, memakai algoritma Unigram
    (mode klasik/default SentencePiece, berbeda dari kelas BPETokenizer
    di atas yang diimplementasikan lewat library `tokenizers`).
    """
    name = "sentencepiece"

    def __init__(self, text, corpus_path=CORPUS_PATH, vocab_size=TARGET_VOCAB_SIZE):
        import sentencepiece as spm

        vocab_size = _safe_vocab_size(text, vocab_size)
        model_prefix = os.path.join(TOKENIZERS_DIR, "sentencepiece")

        spm.SentencePieceTrainer.Train(
            input=corpus_path,
            model_prefix=model_prefix,
            vocab_size=vocab_size,
            model_type="unigram",
        )

        self._sp = spm.SentencePieceProcessor()
        self._sp.load(model_prefix + ".model")
        self.vocab_size = self._sp.get_piece_size()
        self.ids = self.encode(text)
        print(f"[SentencePieceTokenizer] model disimpan ke {model_prefix}.model")

    def encode(self, text):
        return self._sp.encode(text, out_type=int)

    def decode(self, ids):
        return self._sp.decode(ids)


class TiktokenTokenizer(BaseTokenizer):
    """
    7. Tiktoken -- tokenizer BPE dari OpenAI yang SUDAH dilatih
    sebelumnya (pretrained). Tidak ada proses training dari corpus;
    encoding diunduh/di-cache otomatis oleh library `tiktoken`
    (butuh koneksi internet pada pemanggilan pertama).
    """
    name = "tiktoken"

    def __init__(self, text, encoding_name="cl100k_base"):
        import tiktoken

        self._enc = tiktoken.get_encoding(encoding_name)
        self.encoding_name = encoding_name
        self.vocab_size = self._enc.n_vocab
        self.ids = self.encode(text)
        self._save_info()

    def encode(self, text):
        return self._enc.encode(text)

    def decode(self, ids):
        return self._enc.decode(ids)

    def _save_info(self):
        # Vocab tiktoken sangat besar (~100k) dan merupakan aset pretrained
        # milik OpenAI, sehingga kita hanya menyimpan metadata-nya, bukan
        # men-dump seluruh isi vocab.
        path = os.path.join(TOKENIZERS_DIR, "tiktoken_info.json")
        info = {
            "encoding_name": self.encoding_name,
            "vocab_size": self.vocab_size,
            "note": "Tiktoken adalah tokenizer pretrained (tidak dilatih ulang dari corpus).",
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(info, f, indent=2)
        print(f"[TiktokenTokenizer] info disimpan ke {path}")


def build_tokenizer(tokenizer_type, text, corpus_path=CORPUS_PATH):
    """Factory: mengembalikan instance tokenizer sesuai TOKENIZER_TYPE."""
    tokenizer_type = tokenizer_type.lower()
    mapping = {
        "char": lambda: CharTokenizer(text),
        "word": lambda: WordTokenizer(text),
        "subword": lambda: SubwordTokenizer(text, corpus_path),
        "bpe": lambda: BPETokenizer(text, corpus_path),
        "wordpiece": lambda: WordPieceTokenizer(text, corpus_path),
        "sentencepiece": lambda: SentencePieceTokenizer(text, corpus_path),
        "tiktoken": lambda: TiktokenTokenizer(text),
    }
    if tokenizer_type not in mapping:
        raise ValueError(
            f"TOKENIZER_TYPE '{tokenizer_type}' tidak dikenali. "
            f"Pilih salah satu: {', '.join(mapping.keys())}"
        )
    return mapping[tokenizer_type]()


# =====================================================================
# 3. LOAD CORPUS & BANGUN TOKENIZER
# =====================================================================
with open(CORPUS_PATH, "r", encoding="utf-8") as f:
    text = f.read()

print(f"\n>>> Menggunakan TOKENIZER_TYPE = '{TOKENIZER_TYPE}'\n")
tokenizer = build_tokenizer(TOKENIZER_TYPE, text)

data = torch.tensor(tokenizer.ids, dtype=torch.long)
vocab_size = tokenizer.vocab_size

print(data)
print("Vocab size:", vocab_size)


# =====================================================================
# 4. HYPERPARAMETER & ARSITEKTUR MODEL (TIDAK BERUBAH)
# =====================================================================
block_size = 6
embedding_dim = 32
n_heads = 2
n_layers = 2
lr = 1e-3
epochs = 1500


def get_batch(batch_size=16):
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    return x, y


class TinyGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)

        self.position_embedding = nn.Embedding(block_size, embedding_dim)
        self.blocks = nn.Sequential(*[Block(embedding_dim, block_size, n_heads) for _ in range(n_layers)])

        self.ln_f = nn.LayerNorm(embedding_dim)
        self.head = nn.Linear(embedding_dim, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding(idx)

        pos_emb = self.position_embedding(torch.arange(T, device=idx.device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)
        loss = None
        if targets is not None:
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.view(B * T, C), targets.view(B * T))
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            next_idx = torch.multinomial(probs, 1)
            idx = torch.cat((idx, next_idx), dim=1)
        return idx


# =====================================================================
# 5. TRAINING (TIDAK BERUBAH, hanya menambahkan pencatatan waktu/loss)
# =====================================================================
model = TinyGPT()
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

final_loss = None
train_start = time.time()

for step in range(epochs):
    xb, yb = get_batch()
    logits, loss = model(xb, yb)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    final_loss = loss.item()
    if step % 300 == 0:
        print(f"Step {step}, loss={loss.item():.4f}")

train_time = time.time() - train_start


# =====================================================================
# 6. GENERATE TEKS
# =====================================================================
prompt = "Saham Indonesia"
context = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long)

out = model.generate(context, max_new_tokens=100)

print("\nGenerated text:\n")

generated_ids = out[0].tolist()
print(tokenizer.decode(generated_ids))


# =====================================================================
# 7. BENCHMARK TOKENIZER
# =====================================================================
def evaluate_tokenizer(tokenizer, raw_text, final_loss, train_time):
    """
    Menampilkan metrik perbandingan untuk tokenizer yang sedang dipakai:
        1. Vocabulary size
        2. Jumlah token corpus
        3. Compression ratio (karakter per token)
        4. Training loss akhir
        5. Waktu training
    """
    num_tokens = len(tokenizer.ids)
    num_chars = len(raw_text)
    compression_ratio = num_chars / num_tokens if num_tokens > 0 else float("nan")

    print("\n========== BENCHMARK TOKENIZER ==========")
    print(f"Tokenizer             : {TOKENIZER_TYPE}")
    print(f"Vocabulary size        : {tokenizer.vocab_size}")
    print(f"Jumlah token corpus    : {num_tokens}")
    print(f"Jumlah karakter corpus : {num_chars}")
    print(f"Compression ratio      : {compression_ratio:.4f} karakter/token")
    print(f"Training loss akhir    : {final_loss:.4f}")
    print(f"Waktu training         : {train_time:.2f} detik")
    print("==========================================\n")

    return {
        "tokenizer": TOKENIZER_TYPE,
        "vocab_size": tokenizer.vocab_size,
        "num_tokens": num_tokens,
        "num_chars": num_chars,
        "compression_ratio": compression_ratio,
        "final_loss": final_loss,
        "train_time_sec": train_time,
    }


benchmark_result = evaluate_tokenizer(tokenizer, text, final_loss, train_time)