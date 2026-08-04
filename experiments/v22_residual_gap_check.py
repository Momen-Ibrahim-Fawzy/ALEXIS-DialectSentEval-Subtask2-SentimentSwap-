"""
Subtask 2 -- re-run the argmax-vs-renormalized diagnostic on v22's REAL test predictions
(official StyleAcc=0.9581, up from v17's 0.9436 via two compounding fixes: v21's reranking
signal + v22's DPO reward, both switched from classifier.target_prob() [renormalized,
marginalizes out neutral] to classifier.raw_target_prob() [raw 3-way]) to see how much
proxy-vs-real gap remains. On v17 (before either fix), the gap was +3.21pp (renormalized)
/ +1.81pp (raw argmax). If a meaningful gap still remains on v22, there may be more to
extract from further chasing classifier-fidelity issues; if the gap has mostly closed,
this vein is likely exhausted and effort should go elsewhere.

Usage:
  CUDA_VISIBLE_DEVICES=0 conda run -n mo python3 v22_residual_gap_check.py
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer

import config as cfg
from data import target_polarity_for
from classifier_utils import DedicatedSourcePolarityClassifier

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
V22_PREDICTIONS = "submissions/023_v22_dpo_v5/predictions_diagnostic.csv"
REAL_OFFICIAL_STYLEACC = 95.81


@torch.no_grad()
def raw_probs(model, tokenizer, texts, batch_size=32, max_length=128):
    all_probs = []
    for i in range(0, len(texts), batch_size):
        batch = [str(t) for t in texts[i:i + batch_size]]
        enc = tokenizer(batch, truncation=True, max_length=max_length, padding=True, return_tensors="pt").to(DEVICE)
        logits = model(**enc).logits
        all_probs.append(F.softmax(logits, dim=-1).cpu().numpy())
    return np.concatenate(all_probs, axis=0)


def main():
    preds_df = pd.read_csv(V22_PREDICTIONS)
    print(f"Loaded {len(preds_df)} v22 real test predictions")

    print("Detecting source_polarity for test rows via the dedicated classifier...")
    src_pol_clf = DedicatedSourcePolarityClassifier(device=DEVICE)
    preds_df["source_polarity"] = src_pol_clf.predict_polarity(preds_df["source"].tolist())
    target_pols = [target_polarity_for(p) for p in preds_df["source_polarity"]]
    outputs = preds_df["style"].tolist()

    model_name = cfg.EVAL_CLASSIFIER_MODEL
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name).to(DEVICE).eval()
    id2label = {int(k): v.lower() for k, v in model.config.id2label.items()}

    probs = raw_probs(model, tokenizer, outputs)
    pred_labels = [id2label[i] for i in probs.argmax(axis=1)]
    pos_idx = [i for i, l in id2label.items() if l.startswith("pos")][0]
    neg_idx = [i for i, l in id2label.items() if l.startswith("neg")][0]

    renorm_hits, argmax_hits = [], []
    neutral_argmax_count = 0
    for i, tgt in enumerate(target_pols):
        pos, neg = probs[i, pos_idx], probs[i, neg_idx]
        denom = max(pos + neg, 1e-6)
        renorm_target_prob = (pos / denom) if tgt == "positive" else (neg / denom)
        renorm_hits.append(renorm_target_prob >= 0.5)
        argmax_hits.append(pred_labels[i] == tgt)
        if pred_labels[i] == "neutral":
            neutral_argmax_count += 1

    renorm_acc = float(np.mean(renorm_hits)) * 100
    argmax_acc = float(np.mean(argmax_hits)) * 100
    print(f"\nRows where raw argmax = 'neutral': {neutral_argmax_count}/{len(target_pols)} = {neutral_argmax_count/len(target_pols)*100:.2f}%")
    print(f"\nRenormalized-binary StyleAcc proxy: {renorm_acc:.2f}%")
    print(f"Raw 3-way ARGMAX StyleAcc proxy:    {argmax_acc:.2f}%")
    print(f"\nReal official StyleAcc: {REAL_OFFICIAL_STYLEACC:.2f}%")
    print(f"Renormalized proxy gap: {renorm_acc - REAL_OFFICIAL_STYLEACC:+.2f}pp")
    print(f"Raw argmax proxy gap:   {argmax_acc - REAL_OFFICIAL_STYLEACC:+.2f}pp")
    print(f"\nFor reference, v17 (before either fix): renormalized gap was +3.21pp, raw argmax gap was +1.81pp")


if __name__ == "__main__":
    main()
