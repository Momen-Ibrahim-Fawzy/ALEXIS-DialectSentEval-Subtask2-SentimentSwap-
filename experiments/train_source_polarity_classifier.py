"""
Subtask 2 -- train the dedicated source-polarity classifier on ALL train+val labeled
data (7716 rows), for use ONLY at predict.py's Stage 2 (deciding which direction to
swap a test source). See source_polarity_classifier_check.py for the held-out validation:
zero-shot official classifier accuracy 74.31% vs this dedicated classifier 96.72% on the
same held-out split (+22.41pp) -- source polarity is an objective fact about the input,
not something the grading classifier defines, so a classifier fine-tuned specifically on
labeled (source, source_polarity) pairs should (and does) beat a generic zero-shot guess.

The official classifier is still used unchanged for Stage 3+ (reranking/scoring), since
that must match the actual grading criterion.

Usage:
  CUDA_VISIBLE_DEVICES=0 conda run -n mo python3 train_source_polarity_classifier.py
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import os

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

import config as cfg
from data import load_train, load_val

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_MODEL = "UBC-NLP/MARBERTv2"
CKPT_PATH = os.path.join(cfg.CHECKPOINT_DIR, "source_polarity_classifier.pt")


class SourceDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=96):
        self.texts = list(texts)
        self.labels = list(labels)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(str(self.texts[idx]), truncation=True, max_length=self.max_length, padding="max_length")
        item = {k: torch.tensor(v) for k, v in enc.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


class BinaryClassifier(nn.Module):
    def __init__(self, model_name, dropout=0.1):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden, 2)

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None):
        kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
        if token_type_ids is not None:
            kwargs["token_type_ids"] = token_type_ids
        out = self.encoder(**kwargs).last_hidden_state
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (out * mask).sum(1) / mask.sum(1).clamp_min(1e-6)
        return self.classifier(self.dropout(pooled))


def main():
    train_df, val_df = load_train(), load_val()
    full_df = pd.concat([train_df, val_df], ignore_index=True).reset_index(drop=True)
    full_df["label"] = (full_df["source_polarity"] == "Negative").astype(int)
    print(f"Training on {len(full_df)} rows (all train+val, no held-out split -- this is the final classifier)")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    ds = SourceDataset(full_df["source"], full_df["label"], tokenizer)
    loader = DataLoader(ds, batch_size=16, shuffle=True)
    model = BinaryClassifier(BASE_MODEL).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    epochs = 4
    total_steps = len(loader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, int(0.06 * total_steps), total_steps)

    model.train()
    for epoch in range(epochs):
        total_loss, n = 0.0, 0
        for batch in loader:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            labels = batch.pop("labels")
            optimizer.zero_grad()
            logits = model(**batch)
            loss = F.cross_entropy(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
            n += 1
        print(f"[source_polarity_ft] epoch {epoch+1}/{epochs} loss={total_loss/n:.4f}")

    os.makedirs(cfg.CHECKPOINT_DIR, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "model_name": BASE_MODEL}, CKPT_PATH)
    print(f"Saved dedicated source-polarity classifier to {CKPT_PATH}")


if __name__ == "__main__":
    main()
