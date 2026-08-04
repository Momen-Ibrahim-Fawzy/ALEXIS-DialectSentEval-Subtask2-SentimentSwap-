"""
Subtask 2 -- grid-search the classifier-guided reranking weights
(config.POLARITY_SCORE_WEIGHT / CONTENT_SCORE_WEIGHT) on the validation set.

v1 shipped with an untuned 0.7/0.3 split (polarity confidence / chrF-to-source). This
script generates config.NUM_CANDIDATES candidates per val source exactly the way
predict.py does at test time (including using the *classifier's* predicted polarity,
not the gold `source_polarity` column, so the tuning setup matches real test-time
conditions), then re-scores that same fixed candidate pool under a grid of weight
combinations and reports BLEU/chrF/Sentiment-Style-Accuracy for each -- so we only pay
the (expensive) generation cost once and can compare many rerank weightings cheaply.

Usage:
  CUDA_VISIBLE_DEVICES=1 conda run -n mo python3 tune_reranking.py
"""
import json
import os

import numpy as np
import sacrebleu
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

import config as cfg
from classifier_utils import PolarityClassifier
from data import load_val, prefix_for, target_polarity_for


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def generate_candidates(model, tokenizer, sources, polarities, device,
                         num_candidates=cfg.NUM_CANDIDATES, batch_size=8):
    all_candidates = []
    for i in range(0, len(sources), batch_size):
        batch_src = sources[i:i + batch_size]
        batch_pol = polarities[i:i + batch_size]
        inputs = [prefix_for(p) + s for s, p in zip(batch_src, batch_pol)]
        enc = tokenizer(inputs, truncation=True, max_length=cfg.GENERATOR_MAX_SRC_LEN,
                         padding=True, return_tensors="pt").to(device)
        gen = model.generate(**enc, max_length=cfg.GENERATOR_MAX_TGT_LEN,
                              num_beams=num_candidates, num_return_sequences=num_candidates,
                              early_stopping=True)
        decoded = tokenizer.batch_decode(gen, skip_special_tokens=True)
        for j in range(len(batch_src)):
            cands = decoded[j * num_candidates:(j + 1) * num_candidates]
            all_candidates.append([c if c.strip() else batch_src[j] for c in cands])
    return all_candidates


def score_grid(sources, candidates_per_source, target_polarities, targets, classifier, weight_grid):
    chrf = sacrebleu.CHRF(word_order=2)
    # Precompute per-candidate polarity + content scores ONCE, reuse across the whole grid.
    all_pol_scores, all_content_scores = [], []
    for src, cands, tgt_pol in zip(sources, candidates_per_source, target_polarities):
        pol_probs = classifier.target_prob(cands, [tgt_pol] * len(cands))
        content_scores = [chrf.sentence_score(c, [src]).score / 100.0 for c in cands]
        all_pol_scores.append(pol_probs)
        all_content_scores.append(content_scores)

    results = []
    for pw, cw in weight_grid:
        chosen = []
        for cands, pol_scores, content_scores in zip(candidates_per_source, all_pol_scores, all_content_scores):
            combined = [pw * p + cw * c for p, c in zip(pol_scores, content_scores)]
            best_idx = max(range(len(cands)), key=lambda k: combined[k])
            chosen.append(cands[best_idx])

        bleu = sacrebleu.corpus_bleu(chosen, [targets]).score
        chrf_score = chrf.corpus_score(chosen, [targets]).score
        style_probs = classifier.target_prob(chosen, target_polarities)
        style_acc = float(np.mean([p >= 0.5 for p in style_probs])) * 100
        results.append({
            "polarity_weight": pw, "content_weight": cw,
            "bleu": bleu, "chrf": chrf_score, "sentiment_style_accuracy_pct": style_acc,
        })
        print(f"  pw={pw:.1f} cw={cw:.1f} -> BLEU={bleu:.2f} chrF={chrf_score:.2f} StyleAcc={style_acc:.2f}%")
    return results


def main():
    device = get_device()
    val_df = load_val()

    gen_dir = os.path.join(cfg.CHECKPOINT_DIR, "generator")
    tokenizer = AutoTokenizer.from_pretrained(gen_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(gen_dir).to(device).eval()
    classifier = PolarityClassifier(device=device)

    sources = val_df["source"].tolist()
    targets = val_df["target"].tolist()

    # Mirror predict.py's test-time behavior: detect polarity with the classifier, not
    # the gold column, so the tuned weights are valid for the real (label-less) test file.
    pred_source_polarity = classifier.predict_polarity(sources)
    target_polarities = [target_polarity_for(p) for p in pred_source_polarity]

    print(f"Generating {cfg.NUM_CANDIDATES} candidates for {len(sources)} val rows "
          f"(this is the expensive part; reused for every grid point below)...")
    candidates = generate_candidates(model, tokenizer, sources, pred_source_polarity, device)

    weight_grid = [(1.0, 0.0), (0.9, 0.1), (0.8, 0.2), (0.7, 0.3), (0.6, 0.4), (0.5, 0.5), (0.3, 0.7), (0.0, 1.0)]
    print("\nScoring grid:")
    results = score_grid(sources, candidates, target_polarities, targets, classifier, weight_grid)

    out_path = os.path.join(cfg.OUTPUT_DIR, "reranking_grid_search.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nWrote {out_path}")

    best_style = max(results, key=lambda r: r["sentiment_style_accuracy_pct"])
    print(f"\nBest Sentiment Style Accuracy: pw={best_style['polarity_weight']} cw={best_style['content_weight']} "
          f"-> {best_style['sentiment_style_accuracy_pct']:.2f}% (BLEU={best_style['bleu']:.2f}, chrF={best_style['chrf']:.2f})")


if __name__ == "__main__":
    main()
