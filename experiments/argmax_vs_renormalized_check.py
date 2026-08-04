"""
Subtask 2 -- test whether the gap between our self-assessed StyleAcc proxy (97.57% on
v17's real test outputs, per independent_audit_check.py) and the REAL official StyleAcc
(94.36%) is explained by a methodology mismatch: our PolarityClassifier.binary_probs()
MARGINALIZES OUT the neutral class (renormalizes P(pos)/P(neg) to sum to 1), so a candidate
whose true 3-way argmax is "neutral" can still register as "target achieved" under our
renormalized >=0.5 check, even though a raw-argmax grader (the natural, standard reading of
"the percentage of outputs with the correct target polarity" via a 3-class sentiment
classifier -- confirmed from the actual competition page to be exactly CAMeL-Lab/bert-base-
arabic-camelbert-da-sentiment, same model we use) would count it as wrong.

Computes, for v17's real test outputs, BOTH:
  (a) renormalized-binary target_prob >= 0.5  (what we've measured internally all along)
  (b) raw 3-way argmax == target_polarity      (the more standard grading interpretation)
and compares both to the known real official StyleAcc (94.36%) to see which one the real
grader's number actually looks like.

Usage:
  CUDA_VISIBLE_DEVICES=0 conda run -n mo python3 argmax_vs_renormalized_check.py
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
V17_PREDICTIONS = "submissions/018_v17_dedicated_src_polarity/predictions_diagnostic.csv"


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
    preds_df = pd.read_csv(V17_PREDICTIONS)
    print(f"Loaded {len(preds_df)} v17 real test predictions")

    print("Detecting source_polarity for test rows via the dedicated classifier (matches v17's Stage 2)...")
    src_pol_clf = DedicatedSourcePolarityClassifier(device=DEVICE)
    preds_df["source_polarity"] = src_pol_clf.predict_polarity(preds_df["source"].tolist())
    target_pols = [target_polarity_for(p) for p in preds_df["source_polarity"]]
    outputs = preds_df["style"].tolist()

    model_name = cfg.EVAL_CLASSIFIER_MODEL
    print(f"\nLoading {model_name} directly for raw 3-way scoring...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name).to(DEVICE).eval()
    id2label = {int(k): v.lower() for k, v in model.config.id2label.items()}
    print(f"id2label: {id2label}")

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
    print(f"\nRenormalized-binary StyleAcc (marginalizing out neutral, what we've used all along): {renorm_acc:.2f}%")
    print(f"Raw 3-way ARGMAX StyleAcc (neutral counts as miss):                                    {argmax_acc:.2f}%")
    print(f"\nReal official StyleAcc (ground truth): 94.36%")
    print(f"Renormalized proxy gap: {renorm_acc - 94.36:+.2f}pp")
    print(f"Raw argmax proxy gap:   {argmax_acc - 94.36:+.2f}pp")
    if abs(argmax_acc - 94.36) < abs(renorm_acc - 94.36) - 1.0:
        print("\nRaw 3-way argmax is a MUCH closer match to real official StyleAcc -- this is very likely "
              "the actual grading methodology. Our reranking (which selects via the renormalized-binary "
              "score) has been systematically over-crediting candidates whose true best guess is 'neutral' "
              "-- switching Stage 3 reranking's polarity signal to raw argmax-based scoring is a strong, "
              "well-diagnosed candidate for a real StyleAcc improvement.")
    else:
        print("\nRaw argmax does not clearly explain the gap better than the renormalized proxy -- "
              "the discrepancy likely has some other source (preprocessing, tokenization, etc.).")


if __name__ == "__main__":
    main()
