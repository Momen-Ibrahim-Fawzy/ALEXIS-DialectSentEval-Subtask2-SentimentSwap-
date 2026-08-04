"""
Subtask 2 -- check whether a DEDICATED, fine-tuned source-polarity classifier beats the
zero-shot official CAMeL-Lab classifier at the specific job of detecting a test source's
TRUE starting polarity (used only for Stage 2 of predict.py -- deciding which direction
to swap. NOT for Stage 3+ reranking, which must keep using the official classifier since
that IS the actual grading model).

Why this matters: the official classifier's accuracy at source-polarity detection, as
measured against the real gold source_polarity column on val, is only 74.07% (250/964
wrong). Since target_polarity is defined as the opposite of the TRUE source polarity
(which Codabench holds but we don't see at test time), an incorrect source-polarity guess
means our whole pipeline optimizes toward the WRONG target for that row -- no amount of
generation/reranking quality can recover it. This is a fundamentally different question
from "does this classifier's opinion of my output match the grading criterion" (where the
classifier's opinion IS definitionally correct) -- here we're trying to perceive an
objective fact about the INPUT, and the official classifier is just one imperfect,
zero-shot guess at it. We have 7716 labeled (source, source_polarity) train+val pairs
that have never been used to fine-tune anything for this specific job.

Usage:
  CUDA_VISIBLE_DEVICES=0 conda run -n mo python3 source_polarity_classifier_check.py
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "src"))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

import config as cfg
from data import load_train, load_val
from classifier_utils import PolarityClassifier

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_MODEL = "UBC-NLP/MARBERTv2"  # proven strong for dialectal Arabic (Subtask 1's best backbone)


class SourceDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=96):
        self.texts = list(texts)
        self.labels = list(labels) if labels is not None else None
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(str(self.texts[idx]), truncation=True, max_length=self.max_length, padding="max_length")
        item = {k: torch.tensor(v) for k, v in enc.items()}
        if self.labels is not None:
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
        pooled = (out * mask).sum(1) / mask.sum(1).clamp_min(1e-6)  # mean pooling, proven best in Subtask 1
        return self.classifier(self.dropout(pooled))


def main():
    train_df, val_df = load_train(), load_val()
    full_df = pd.concat([train_df, val_df], ignore_index=True).reset_index(drop=True)
    full_df["label"] = (full_df["source_polarity"] == "Negative").astype(int)  # 0=positive, 1=negative

    gss = GroupShuffleSplit(n_splits=1, test_size=0.10, random_state=cfg.SEED)
    fit_idx, holdout_idx = next(gss.split(full_df, groups=full_df["source"]))
    fit_df = full_df.iloc[fit_idx].reset_index(drop=True)
    holdout_df = full_df.iloc[holdout_idx].reset_index(drop=True)
    print(f"Fit: {len(fit_df)} rows, Holdout: {len(holdout_df)} rows (grouped by exact source text, no leakage)")

    # Baseline: zero-shot official classifier's source-polarity accuracy on the SAME holdout
    official = PolarityClassifier(device=DEVICE)
    official_pred = official.predict_polarity(holdout_df["source"].tolist())
    official_acc = accuracy_score(holdout_df["source_polarity"].tolist(), official_pred)
    print(f"\nZero-shot official classifier accuracy on holdout: {official_acc:.4f}")

    # Fine-tune a dedicated classifier on fit_df
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    train_ds = SourceDataset(fit_df["source"], fit_df["label"], tokenizer)
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    model = BinaryClassifier(BASE_MODEL).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    epochs = 4
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, int(0.06 * total_steps), total_steps)

    model.train()
    for epoch in range(epochs):
        total_loss, n = 0.0, 0
        for batch in train_loader:
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

    model.eval()
    holdout_ds = SourceDataset(holdout_df["source"], None, tokenizer)
    holdout_loader = DataLoader(holdout_ds, batch_size=64)
    all_preds = []
    with torch.no_grad():
        for batch in holdout_loader:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            logits = model(**batch)
            all_preds.extend(logits.argmax(dim=-1).cpu().tolist())
    ft_pred = ["Negative" if p == 1 else "Positive" for p in all_preds]
    ft_acc = accuracy_score(holdout_df["source_polarity"].tolist(), ft_pred)
    print(f"\nFine-tuned dedicated classifier accuracy on holdout: {ft_acc:.4f}")

    print(f"\n=== SUMMARY: zero-shot official={official_acc:.4f}  fine-tuned dedicated={ft_acc:.4f}  "
          f"margin={ft_acc-official_acc:+.4f} ===")
    if ft_acc - official_acc >= 0.03:
        print("Fine-tuned classifier clearly beats zero-shot -- worth training on full data and integrating "
              "into predict.py's Stage 2 (source-polarity detection only, NOT reranking).")
    else:
        print("Fine-tuned classifier did not clearly beat zero-shot -- reconsider.")

    torch.save({"state_dict": model.state_dict(), "model_name": BASE_MODEL}, "outputs/source_polarity_ft_check.pt")


if __name__ == "__main__":
    main()
