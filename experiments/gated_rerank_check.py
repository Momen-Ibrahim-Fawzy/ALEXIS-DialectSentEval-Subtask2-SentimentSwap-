"""
Subtask 2 -- diagnose and fix the gap between the candidate-pool "oracle ceiling" (>=1 of
10 plain-beam candidates hits target polarity for 99.17% of val rows, per
diverse_beam_check.py) and the actual official Sentiment Style Accuracy (94.36%, v17/v19).

Since coverage is already near-ceiling, the gap must be in SELECTION, not generation. The
current reranking (predict.py's rerank()) picks argmax(0.85*pol_prob + 0.15*chrF/100). On
BORDERLINE rows -- two candidates with near-tied polarity scores straddling 0.5 -- simple
algebra shows a chrF gap of just ~11 points (0.15*11/100 > 0.85*0.02) is enough to flip the
pick from a barely-correct-polarity candidate to a barely-wrong one, even though a clearly
correct candidate likely exists elsewhere in the pool (99%+ coverage). This directly
sacrifices Style Accuracy (the SOLE official ranking metric, confirmed from the task spec)
for a small chrF gain on exactly the rows where it matters least to be wrong.

Fix tested here: GATED reranking -- first filter candidates to those with pol_prob >= 0.5;
rank by content score (chrF-to-source) WITHIN that correct-polarity subset; only fall back
to the full weighted-score pool if no candidate clears the gate (rare, given 99%+ coverage).
This should reclaim most of the ceiling-vs-actual gap while barely touching content
preservation, since we're still picking the best-content candidate, just only among ones
that are already polarity-correct.

Uses the REAL deployed primary generator (checkpoints/generator_dpo_v3, per
SUBMISSIONS_LOG.md v17/v19) so results are directly representative, not a proxy backbone.

Usage:
  CUDA_VISIBLE_DEVICES=1 conda run -n mo python3 gated_rerank_check.py
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "src"))

import numpy as np
import sacrebleu
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

import config as cfg
from classifier_utils import PolarityClassifier
from data import load_val, prefix_for, target_polarity_for

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
GEN_DIR = "checkpoints/generator_dpo_v3"
NUM_CANDIDATES = 10


@torch.no_grad()
def generate_plain(model, tokenizer, sources, polarities, batch_size=8):
    all_candidates = []
    for i in range(0, len(sources), batch_size):
        batch_src = sources[i:i + batch_size]
        batch_pol = polarities[i:i + batch_size]
        inputs = [prefix_for(p) + s for s, p in zip(batch_src, batch_pol)]
        enc = tokenizer(inputs, truncation=True, max_length=cfg.GENERATOR_MAX_SRC_LEN,
                         padding=True, return_tensors="pt").to(DEVICE)
        gen = model.generate(**enc, max_length=cfg.GENERATOR_MAX_TGT_LEN, num_beams=NUM_CANDIDATES,
                              num_return_sequences=NUM_CANDIDATES, early_stopping=True)
        decoded = tokenizer.batch_decode(gen, skip_special_tokens=True)
        for j in range(len(batch_src)):
            cands = decoded[j * NUM_CANDIDATES:(j + 1) * NUM_CANDIDATES]
            all_candidates.append([c if c.strip() else batch_src[j] for c in cands])
    return all_candidates


def main():
    val_df = load_val()
    print(f"Evaluating on full val set: {len(val_df)} rows, generator={GEN_DIR}")

    tokenizer = AutoTokenizer.from_pretrained(GEN_DIR)
    model = AutoModelForSeq2SeqLM.from_pretrained(GEN_DIR).to(DEVICE).eval()
    classifier = PolarityClassifier(device=DEVICE)
    chrf = sacrebleu.CHRF(word_order=2)

    sources = val_df["source"].tolist()
    targets = val_df["target"].tolist()
    polarities = val_df["source_polarity"].tolist()
    target_pols = [target_polarity_for(p) for p in polarities]

    print("Generating plain-beam candidates...")
    candidates_per_row = generate_plain(model, tokenizer, sources, polarities)
    del model
    torch.cuda.empty_cache()

    current_picks, gated_picks = [], []
    n_fallback = 0
    for src, cands, tgt_pol in zip(sources, candidates_per_row, target_pols):
        pol_probs = classifier.target_prob(cands, [tgt_pol] * len(cands))
        content_scores = [chrf.sentence_score(c, [src]).score / 100.0 for c in cands]

        combined = [cfg.POLARITY_SCORE_WEIGHT * p + cfg.CONTENT_SCORE_WEIGHT * c
                    for p, c in zip(pol_probs, content_scores)]
        current_idx = max(range(len(cands)), key=lambda k: combined[k])
        current_picks.append((cands[current_idx], pol_probs[current_idx]))

        gated_idx_pool = [k for k in range(len(cands)) if pol_probs[k] >= 0.5]
        if gated_idx_pool:
            gated_idx = max(gated_idx_pool, key=lambda k: content_scores[k])
        else:
            n_fallback += 1
            gated_idx = current_idx
        gated_picks.append((cands[gated_idx], pol_probs[gated_idx]))

    def summarize(picks, label):
        texts = [p[0] for p in picks]
        pol_probs = [p[1] for p in picks]
        style_acc = float(np.mean([p >= 0.5 for p in pol_probs])) * 100
        bleu = sacrebleu.corpus_bleu(texts, [targets]).score
        chrf_score = chrf.corpus_score(texts, [targets]).score
        chrf_to_src = float(np.mean([chrf.sentence_score(t, [s]).score for t, s in zip(texts, sources)]))
        print(f"{label}: StyleAcc(proxy)={style_acc:.2f}%  BLEU-to-gold={bleu:.2f}  chrF-to-gold={chrf_score:.2f}  chrF-to-source={chrf_to_src:.2f}")
        return style_acc, bleu, chrf_score

    print(f"\nGate fallback rate (no candidate cleared pol_prob>=0.5): {n_fallback}/{len(sources)} = {n_fallback/len(sources)*100:.2f}%\n")
    cur_style, cur_bleu, cur_chrf = summarize(current_picks, "CURRENT (weighted 0.85/0.15)")
    gated_style, gated_bleu, gated_chrf = summarize(gated_picks, "GATED   (polarity gate, then content)")

    print(f"\nStyleAcc margin (gated - current): {gated_style - cur_style:+.2f}pp")
    print(f"BLEU margin (gated - current): {gated_bleu - cur_bleu:+.2f}")
    print(f"chrF margin (gated - current): {gated_chrf - cur_chrf:+.2f}")
    if gated_style - cur_style >= 1.0:
        print("\nGated reranking meaningfully improves StyleAcc (the sole official ranking metric) "
              "-- worth integrating into predict.py's rerank() and submitting.")
    else:
        print("\nGated reranking did NOT meaningfully improve StyleAcc on this proxy -- reconsider.")


if __name__ == "__main__":
    main()
