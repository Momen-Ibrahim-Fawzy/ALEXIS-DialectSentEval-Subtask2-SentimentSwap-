"""
Subtask 2 -- near-duplicate retrieval augmentation via BGE-M3 embeddings.

Exact-match retrieval (predict.py Stage 1) only fires on byte-identical sources, but the
EDA found the underlying social-media corpus is full of near-duplicate reposts (minor
spelling/emoji/punctuation variations of the same text -- see EDA section 2's "top
repeated sources" table). For a test source that's *almost* identical to a train/val
source but not byte-for-byte, that train/val pair's target is still a strong candidate.

Design (same low-risk pattern as edit_tagger.py): this is NOT a hard override like
exact-match. The nearest neighbor's target is added as ONE MORE candidate into predict.py's
existing classifier+chrF reranking pool, so it only wins when it's actually good --
it can't make things worse than not having it.

Usage:
  CUDA_VISIBLE_DEVICES=0 conda run -n mo python3 retrieval_augment.py --mode build_index
  (then predict.py imports and uses query_nearest() automatically if the index exists)
"""
import os

import numpy as np
import pandas as pd
import torch

import config as cfg
from data import load_train, load_val

EMBED_MODEL = "BAAI/bge-m3"
INDEX_PATH = os.path.join(cfg.OUTPUT_DIR, "retrieval_index.npz")
SIMILARITY_THRESHOLD = 0.90  # conservative: only propose a candidate for genuinely near-duplicate sources

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_embedder():
    from transformers import AutoModel, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL)
    model = AutoModel.from_pretrained(EMBED_MODEL).to(DEVICE).eval()
    return model, tokenizer


@torch.no_grad()
def embed_texts(texts, model, tokenizer, batch_size=32, max_length=64):
    all_vecs = []
    for i in range(0, len(texts), batch_size):
        batch = [str(t) for t in texts[i:i + batch_size]]
        enc = tokenizer(batch, truncation=True, max_length=max_length, padding=True, return_tensors="pt").to(DEVICE)
        out = model(**enc).last_hidden_state[:, 0]  # BGE-M3 dense embedding = [CLS] representation
        out = torch.nn.functional.normalize(out, dim=-1)
        all_vecs.append(out.cpu().numpy())
    return np.concatenate(all_vecs, axis=0)


def build_index():
    train_df, val_df = load_train(), load_val()
    combined = pd.concat([train_df[["source", "target"]], val_df[["source", "target"]]], ignore_index=True)
    combined = combined.drop_duplicates(subset=["source"]).reset_index(drop=True)

    model, tokenizer = _load_embedder()
    print(f"Embedding {len(combined)} unique train+val sources with {EMBED_MODEL}...")
    embeddings = embed_texts(combined["source"].tolist(), model, tokenizer)

    np.savez(INDEX_PATH, embeddings=embeddings, sources=combined["source"].values, targets=combined["target"].values)
    print(f"Wrote index to {INDEX_PATH} ({embeddings.shape})")
    del model
    torch.cuda.empty_cache()


class RetrievalIndex:
    def __init__(self):
        data = np.load(INDEX_PATH, allow_pickle=True)
        self.embeddings = data["embeddings"]
        self.sources = data["sources"]
        self.targets = data["targets"]
        self.model, self.tokenizer = _load_embedder()

    def query_nearest(self, texts, threshold=SIMILARITY_THRESHOLD):
        """Returns a list the same length as `texts`: the nearest train/val target if its
        cosine similarity exceeds `threshold`, else None."""
        query_vecs = embed_texts(texts, self.model, self.tokenizer)
        sims = query_vecs @ self.embeddings.T  # (N, M), both L2-normalized -> cosine sim
        best_idx = sims.argmax(axis=1)
        best_sim = sims[np.arange(len(texts)), best_idx]
        results = []
        for i in range(len(texts)):
            if best_sim[i] >= threshold:
                results.append(str(self.targets[best_idx[i]]))
            else:
                results.append(None)
        return results, best_sim


def load_index_if_available():
    if not os.path.exists(INDEX_PATH):
        return None
    return RetrievalIndex()


if __name__ == "__main__":
    build_index()
