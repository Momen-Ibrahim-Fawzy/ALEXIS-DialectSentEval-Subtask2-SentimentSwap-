"""
Subtask 2 -- re-tune reranking against the CURRENT full candidate pool (generator_dpo_v3
+ fixed multi-word edit-tagger + restored NLLB + near-duplicate retrieval [+ mt5-large if
present]), and test a metric-aware reward SHAPE, not just its weight.

Why this exists: the original tune_reranking.py (a) only pooled mT5-base beams, no
tagger/NLLB/retrieval/mt5-large, and its 0.85/0.15 weights are stale now that the pool
composition changed (tagger coverage was fixed, NLLB was restored after being silently
missing for several submissions); and (b) always combined the classifier's *continuous*
target-polarity probability linearly with chrF. But the official metric only checks
`prob >= 0.5` (README.md: Sentiment Style Accuracy via a fixed classifier threshold) --
pushing an already-safe 0.85 confidence to 0.98 buys nothing on the real metric, yet a
linear reward keeps paying for it, crowding out the content (chrF) term exactly where it
doesn't need to. A SATURATING polarity score (saturates to 1.0 once the classifier is
already confidently past the target threshold) should let content dominate the choice
among safe candidates, which a linear weight grid alone can never discover no matter how
it's tuned -- it's a different reward *shape*, not just a different point on the same
tradeoff line every v3-v8 reward-weight sweep explored.

Usage:
  CUDA_VISIBLE_DEVICES=1 conda run -n mo python3 tune_reranking_v2.py
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import json
import os

import numpy as np
import sacrebleu
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

import config as cfg
from classifier_utils import PolarityClassifier
from data import load_val, prefix_for, target_polarity_for

GENERATOR_DIR = os.path.join(cfg.CHECKPOINT_DIR, "generator_dpo_v3")  # current best (StyleAcc=0.7794/0.7809 official)
SAT_THRESHOLD = 0.5  # the official metric's own threshold (README.md: Sentiment Style Accuracy = pct with prob >= 0.5)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def generate_candidates(model, tokenizer, sources, polarities, device, num_candidates, batch_size=8):
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


@torch.no_grad()
def generate_nllb_candidates(model, tokenizer, sources, polarities, device, num_candidates=3, batch_size=8):
    lang_token_id = tokenizer.convert_tokens_to_ids("arb_Arab")
    all_candidates = []
    for i in range(0, len(sources), batch_size):
        batch_src = sources[i:i + batch_size]
        batch_pol = polarities[i:i + batch_size]
        inputs = [prefix_for(p) + s for s, p in zip(batch_src, batch_pol)]
        enc = tokenizer(inputs, truncation=True, max_length=cfg.GENERATOR_MAX_SRC_LEN,
                         padding=True, return_tensors="pt").to(device)
        gen = model.generate(**enc, forced_bos_token_id=lang_token_id, max_length=cfg.GENERATOR_MAX_TGT_LEN,
                              num_beams=num_candidates, num_return_sequences=num_candidates, early_stopping=True)
        decoded = tokenizer.batch_decode(gen, skip_special_tokens=True)
        for j in range(len(batch_src)):
            all_candidates.append(decoded[j * num_candidates:(j + 1) * num_candidates])
    return all_candidates


def build_full_pool(sources, pred_source_polarity, device):
    """Mirror predict.py's candidate pool exactly (generator_dpo_v3 + tagger + NLLB +
    retrieval [+ mt5-large if trained]), so the tuned weights are valid for the real pool."""
    tokenizer = AutoTokenizer.from_pretrained(GENERATOR_DIR)
    model = AutoModelForSeq2SeqLM.from_pretrained(GENERATOR_DIR).to(device).eval()
    candidates = generate_candidates(model, tokenizer, sources, pred_source_polarity, device, cfg.NUM_CANDIDATES)
    del model
    torch.cuda.empty_cache()

    try:
        import edit_tagger
        tagger_model, tagger_tokenizer, replace_vocab = edit_tagger.load_tagger()
        tagger_candidates = edit_tagger.tag_and_reconstruct(sources, tagger_model, tagger_tokenizer, replace_vocab)
        candidates = [c + [tc] for c, tc in zip(candidates, tagger_candidates)]
        del tagger_model
        torch.cuda.empty_cache()
        print("Pooled: edit-tagger candidate added.")
    except FileNotFoundError:
        print("No edit-tagger checkpoint; skipping.")

    nllb_dir = os.path.join(cfg.CHECKPOINT_DIR, "generator_nllb")
    if os.path.isdir(nllb_dir):
        nllb_tokenizer = AutoTokenizer.from_pretrained(nllb_dir, src_lang="arb_Arab", tgt_lang="arb_Arab")
        nllb_model = AutoModelForSeq2SeqLM.from_pretrained(nllb_dir).to(device).eval()
        nllb_candidates = generate_nllb_candidates(nllb_model, nllb_tokenizer, sources, pred_source_polarity, device)
        candidates = [c + nc for c, nc in zip(candidates, nllb_candidates)]
        del nllb_model
        torch.cuda.empty_cache()
        print("Pooled: NLLB candidates added.")
    else:
        print("No NLLB checkpoint; skipping.")

    large_dir = os.path.join(cfg.CHECKPOINT_DIR, "generator_large")
    if os.path.isdir(large_dir):
        large_tokenizer = AutoTokenizer.from_pretrained(large_dir)
        large_model = AutoModelForSeq2SeqLM.from_pretrained(large_dir).to(device).eval()
        large_candidates = generate_candidates(large_model, large_tokenizer, sources, pred_source_polarity, device, num_candidates=3)
        candidates = [c + lc for c, lc in zip(candidates, large_candidates)]
        del large_model
        torch.cuda.empty_cache()
        print("Pooled: mt5-large candidates added.")
    else:
        print("No mt5-large checkpoint yet; skipping (retune again once it lands).")

    import retrieval_augment
    retrieval_index = retrieval_augment.load_index_if_available()
    if retrieval_index is not None:
        nn_candidates, _ = retrieval_index.query_nearest(sources)
        candidates = [c + ([nc] if nc is not None else []) for c, nc in zip(candidates, nn_candidates)]
        print("Pooled: near-duplicate retrieval candidate added.")

    return candidates


def score_grid(sources, candidates_per_source, target_polarities, targets, classifier, weight_grid, shape="linear"):
    chrf = sacrebleu.CHRF(word_order=2)
    all_pol_scores, all_content_scores = [], []
    for src, cands, tgt_pol in zip(sources, candidates_per_source, target_polarities):
        pol_probs = classifier.target_prob(cands, [tgt_pol] * len(cands))
        if shape == "saturating":
            pol_probs = [min(p / SAT_THRESHOLD, 1.0) for p in pol_probs]
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
        results.append({"shape": shape, "polarity_weight": pw, "content_weight": cw,
                         "bleu": bleu, "chrf": chrf_score, "sentiment_style_accuracy_pct": style_acc})
        print(f"  [{shape}] pw={pw:.2f} cw={cw:.2f} -> BLEU={bleu:.2f} chrF={chrf_score:.2f} StyleAcc={style_acc:.2f}%")
    return results


def main():
    device = get_device()
    val_df = load_val()
    classifier = PolarityClassifier(device=device)

    sources = val_df["source"].tolist()
    targets = val_df["target"].tolist()
    pred_source_polarity = classifier.predict_polarity(sources)
    target_polarities = [target_polarity_for(p) for p in pred_source_polarity]

    print(f"Building full candidate pool for {len(sources)} val rows (generator_dpo_v3 + tagger + NLLB + "
          f"retrieval [+ mt5-large if present])...")
    candidates = build_full_pool(sources, pred_source_polarity, device)
    pool_sizes = [len(c) for c in candidates]
    print(f"Pool size per row: min={min(pool_sizes)} max={max(pool_sizes)} mean={np.mean(pool_sizes):.1f}")

    weight_grid = [(1.0, 0.0), (0.95, 0.05), (0.9, 0.1), (0.85, 0.15), (0.8, 0.2), (0.7, 0.3), (0.6, 0.4), (0.5, 0.5)]

    print("\n=== Linear (current) reward shape ===")
    linear_results = score_grid(sources, candidates, target_polarities, targets, classifier, weight_grid, shape="linear")

    print("\n=== Saturating (metric-aware) reward shape ===")
    sat_results = score_grid(sources, candidates, target_polarities, targets, classifier, weight_grid, shape="saturating")

    all_results = linear_results + sat_results
    out_path = os.path.join(cfg.OUTPUT_DIR, "reranking_grid_search_v2.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\nWrote {out_path}")

    # Current best real submission (v9) is StyleAcc=0.7809; among configs that meet/beat
    # that on val, report the one with the best BLEU/chrF (the actual goal: keep StyleAcc,
    # win back BLEU/chrF -- not just chase StyleAcc further up the same old tradeoff).
    current_best_style_acc = 78.09
    contenders = [r for r in all_results if r["sentiment_style_accuracy_pct"] >= current_best_style_acc]
    if contenders:
        best = max(contenders, key=lambda r: r["bleu"] + r["chrf"])
        print(f"\nBest BLEU+chrF among configs with val StyleAcc >= {current_best_style_acc}: {best}")
    else:
        best = max(all_results, key=lambda r: r["sentiment_style_accuracy_pct"])
        print(f"\nNo config reached {current_best_style_acc} StyleAcc on val; best overall: {best}")


if __name__ == "__main__":
    main()
