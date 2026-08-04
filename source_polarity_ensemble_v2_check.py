"""
Subtask 2 -- check whether ensembling a SECOND, architecturally-different dedicated
source-polarity classifier (CAMeLBERT-DA, dialectal-Arabic-tuned) with the existing
MARBERTv2-based one improves Stage 2 detection accuracy further (currently 96.72% held-out
alone, 97.25% blended with the official zero-shot classifier per v19_classifier_ensemble).

Motivation: v22_residual_gap_check.py showed a persistent ~1.55pp gap between our best
internal StyleAcc proxy and real official StyleAcc that ISN'T explained by the neutral-
marginalization fix (already captured most of its value: v17->v22 cut that specific gap
from 1.81pp to 1.55pp). A structurally different, self-consistently-invisible source of
error: any row where Stage 2 detects the WRONG source polarity gets optimized toward the
wrong target entirely -- and this failure mode can't show up in ANY internal proxy check
built using the SAME (possibly-wrong) detector to both drive generation and grade it. This
is legitimate to improve via ensembling (unlike Stage 3's grading classifier): source
polarity is an objective fact about real, human-written input text, not something defined
by any one classifier's opinion.

Usage:
  CUDA_VISIBLE_DEVICES=0 conda run -n mo python3 source_polarity_ensemble_v2_check.py
"""
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
BACKBONE_A = "UBC-NLP/MARBERTv2"
BACKBONE_B = "CAMeL-Lab/bert-base-arabic-camelbert-da"


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
        pooled = (out * mask).sum(1) / mask.sum(1).clamp_min(1e-6)
        return self.classifier(self.dropout(pooled))


def train_and_predict_probs(model_name, fit_df, holdout_df):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    train_ds = SourceDataset(fit_df["source"], fit_df["label"], tokenizer)
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    model = BinaryClassifier(model_name).to(DEVICE)
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
        print(f"[{model_name}] epoch {epoch+1}/{epochs} loss={total_loss/n:.4f}")

    model.eval()
    holdout_ds = SourceDataset(holdout_df["source"], None, tokenizer)
    holdout_loader = DataLoader(holdout_ds, batch_size=64)
    all_probs = []
    with torch.no_grad():
        for batch in holdout_loader:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            logits = model(**batch)
            all_probs.append(F.softmax(logits, dim=-1).cpu().numpy())
    del model
    torch.cuda.empty_cache()
    return np.concatenate(all_probs, axis=0)


def main():
    train_df, val_df = load_train(), load_val()
    full_df = pd.concat([train_df, val_df], ignore_index=True).reset_index(drop=True)
    full_df["label"] = (full_df["source_polarity"] == "Negative").astype(int)

    gss = GroupShuffleSplit(n_splits=1, test_size=0.10, random_state=cfg.SEED)
    fit_idx, holdout_idx = next(gss.split(full_df, groups=full_df["source"]))
    fit_df = full_df.iloc[fit_idx].reset_index(drop=True)
    holdout_df = full_df.iloc[holdout_idx].reset_index(drop=True)
    print(f"Fit: {len(fit_df)} rows, Holdout: {len(holdout_df)} rows")

    gold = holdout_df["source_polarity"].tolist()

    print(f"\n=== Training dedicated classifier A ({BACKBONE_A}) ===")
    probs_a = train_and_predict_probs(BACKBONE_A, fit_df, holdout_df)
    pred_a = ["Negative" if p[1] >= p[0] else "Positive" for p in probs_a]
    acc_a = accuracy_score(gold, pred_a)
    print(f"Classifier A ({BACKBONE_A}) holdout accuracy: {acc_a:.4f}")

    print(f"\n=== Training dedicated classifier B ({BACKBONE_B}) ===")
    probs_b = train_and_predict_probs(BACKBONE_B, fit_df, holdout_df)
    pred_b = ["Negative" if p[1] >= p[0] else "Positive" for p in probs_b]
    acc_b = accuracy_score(gold, pred_b)
    print(f"Classifier B ({BACKBONE_B}) holdout accuracy: {acc_b:.4f}")

    ensemble_probs = (probs_a + probs_b) / 2
    pred_ens = ["Negative" if p[1] >= p[0] else "Positive" for p in ensemble_probs]
    acc_ens = accuracy_score(gold, pred_ens)
    print(f"\nA+B ensemble holdout accuracy: {acc_ens:.4f}")

    print(f"\nFor reference: dedicated MARBERTv2 alone (prior check) = 0.9672, "
          f"blended with official zero-shot (v19_classifier_ensemble) = 0.9725")
    best_single = max(acc_a, acc_b)
    margin = acc_ens - best_single
    print(f"Margin (A+B ensemble - best single): {margin:+.4f}")
    if margin >= 0.005 and acc_ens > 0.9725:
        print("A+B ensemble beats both the best single dedicated classifier AND the current "
              "deployed blend (0.9725) -- worth deploying for Stage 2.")
    else:
        print("A+B ensemble did NOT clearly beat the current deployed Stage 2 setup -- NULL result.")


if __name__ == "__main__":
    main()
