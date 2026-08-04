"""
Subtask 2 -- cross-validate our own StyleAcc proxy (computed with CAMeL-Lab/bert-base-
arabic-camelbert-da-sentiment, the SAME classifier used throughout Stage 3 reranking)
against a genuinely INDEPENDENT classifier: Subtask 1's own trained sentiment ensemble
(MARBERTv2 + CAMeLBERT-DA + AraBERTv2, different architecture mix, different training
data -- Subtask 1's dialect-sentiment corpus, not Subtask 2's), which has never touched
Subtask 2's pipeline in any way.

Motivation: v20_gated_rerank looked flawless in held-out validation (identical StyleAcc-
proxy to the baseline, +1.06 BLEU/+0.15 chrF) but REGRESSED hard on real Codabench test
(0.9436 -> 0.8933). Root cause: that validation used the SAME classifier for both
selecting candidates and measuring "success" -- self-referential and guaranteed to look
good regardless of real merit. This script establishes a genuinely independent audit
signal for FUTURE Subtask 2 experiments (especially any reranking/selection change), so we
don't repeat that mistake. It also sanity-checks how close our internal proxy already sits
to ground truth: if the independent classifier's assessment of v17's ACTUAL test
predictions lands much closer to the real official StyleAcc (0.9436) than our own
classifier's self-assessment does, that's useful calibration information for how much to
trust internal proxies going forward.

Uses v17's real, already-submitted test predictions (submissions/018_v17_dedicated_src_
polarity/predictions_diagnostic.csv) -- no need to regenerate anything.

Usage:
  CUDA_VISIBLE_DEVICES=0 conda run -n mo python3 independent_audit_check.py
"""
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from data import load_test, target_polarity_for  # noqa: E402
from classifier_utils import PolarityClassifier, DedicatedSourcePolarityClassifier  # noqa: E402

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SUBTASK1_CKPT_DIR = "../../Subtask 1 - Arabic Dialect Sentiment Analysis/System/checkpoints"
SUBTASK1_BACKBONES = ["marbertv2", "camelbert_da", "arabertv2"]
V17_PREDICTIONS = "submissions/018_v17_dedicated_src_polarity/predictions_diagnostic.csv"


@torch.no_grad()
def predict_probs_batched(model, tokenizer, texts, batch_size=32, max_length=128):
    all_probs = []
    for i in range(0, len(texts), batch_size):
        batch = [str(t) for t in texts[i:i + batch_size]]
        enc = tokenizer(batch, truncation=True, max_length=max_length, padding=True, return_tensors="pt").to(DEVICE)
        out = model(**enc)
        all_probs.append(F.softmax(out.logits, dim=-1).cpu().numpy())
    return np.concatenate(all_probs, axis=0)


def main():
    preds_df = pd.read_csv(V17_PREDICTIONS)
    print(f"Loaded {len(preds_df)} v17 real test predictions")

    # test has no gold source_polarity -- reconstruct it the same way v17 did (dedicated
    # classifier alone, per submission 018's tag "v17_dedicated_src_polarity"), so the
    # target-direction we audit against matches what was actually used at generation time.
    print("Detecting source_polarity for test rows via the dedicated classifier (matches v17's Stage 2)...")
    src_pol_clf = DedicatedSourcePolarityClassifier(device=DEVICE)
    preds_df["source_polarity"] = src_pol_clf.predict_polarity(preds_df["source"].tolist())
    target_pols = [target_polarity_for(p) for p in preds_df["source_polarity"]]
    outputs = preds_df["style"].tolist()

    print("\n=== Scoring with OUR classifier (CAMeL-Lab, same one used for reranking) ===")
    our_clf = PolarityClassifier(device=DEVICE)
    our_probs = our_clf.target_prob(outputs, target_pols)
    our_style_acc = float(np.mean([p >= 0.5 for p in our_probs])) * 100
    print(f"Our-classifier StyleAcc proxy on v17's real test outputs: {our_style_acc:.2f}%")
    print(f"(Real official StyleAcc for this exact submission was 94.36%)")

    print("\n=== Scoring with INDEPENDENT classifier (Subtask 1's ensemble) ===")
    id2label = {0: "negative", 1: "neutral", 2: "positive"}
    all_probs = []
    for name in SUBTASK1_BACKBONES:
        ckpt_dir = f"{SUBTASK1_CKPT_DIR}/{name}"
        print(f"Loading {name}...")
        tokenizer = AutoTokenizer.from_pretrained(ckpt_dir)
        model = AutoModelForSequenceClassification.from_pretrained(ckpt_dir).to(DEVICE).eval()
        probs = predict_probs_batched(model, tokenizer, outputs)
        all_probs.append(probs)
        del model
        torch.cuda.empty_cache()

    ensemble_probs = np.mean(all_probs, axis=0)  # (n, 3): [negative, neutral, positive]
    pred_labels = [id2label[i] for i in ensemble_probs.argmax(axis=1)]
    hits = [pred == tgt for pred, tgt in zip(pred_labels, target_pols)]
    independent_style_acc = float(np.mean(hits)) * 100
    print(f"Independent-classifier StyleAcc proxy on v17's real test outputs: {independent_style_acc:.2f}%")

    print(f"\n{'='*70}")
    print(f"Real official StyleAcc (ground truth):        94.36%")
    print(f"Our classifier's self-assessment (biased):    {our_style_acc:.2f}%  (gap: {our_style_acc - 94.36:+.2f}pp)")
    print(f"Independent classifier's assessment:          {independent_style_acc:.2f}%  (gap: {independent_style_acc - 94.36:+.2f}pp)")
    print(f"{'='*70}")
    if abs(independent_style_acc - 94.36) < abs(our_style_acc - 94.36):
        print("\nThe independent classifier is a MORE reliable proxy for real official StyleAcc "
              "than our own reranking classifier's self-assessment -- use it to audit any future "
              "reranking/selection change before trusting a held-out win.")
    else:
        print("\nThe independent classifier is NOT clearly more reliable here -- the gap between our "
              "internal proxies and real official StyleAcc likely reflects genuine test-set difficulty "
              "or distribution shift rather than pure self-referential bias.")


if __name__ == "__main__":
    main()
