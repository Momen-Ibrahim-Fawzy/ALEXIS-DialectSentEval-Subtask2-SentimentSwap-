"""
Subtask 2 -- inference / submission builder.

Pipeline (see ../EDA/EDA_REPORT.md section 11 for the reasoning behind each stage):

  1. Exact-match retrieval: if a test `source` is byte-identical to a train/val `source`
     (~23% of test rows per the EDA), reuse the human-written `target` directly -- no
     model involved, gold-quality output for free. When a source has multiple distinct
     human targets, pick the most frequent one.

  2. Test-time polarity detection: the released test file has no `source_polarity`
     column, so run the OFFICIAL evaluation classifier (CAMeL-Lab/bert-base-arabic-
     camelbert-da-sentiment) on every remaining source to decide which direction to swap.

  3. Generation + classifier-guided reranking: beam-search `NUM_CANDIDATES` diverse
     candidates from the fine-tuned generator, PLUS (if trained) one more candidate from
     the minimal-edit tagging model (edit_tagger.py), then score every candidate by a
     weighted combination of (a) the official classifier's confidence that it reached the
     *target* polarity, and (b) chrF similarity to the source (content preservation,
     mirrors the official secondary metric), and keep the best-scoring candidate.

  4. Emoji safety net: deterministically flip any left-over source-polarity emoji using
     the lexicon mined from training data (lexicons.py), if the model didn't already.

Output: outputs/predictions.xlsx with columns **id**, **style** (per the Codabench
"Submission Guidelines": "The XLSX file should contain two columns: id ... and style
... The name of both ZIP and .xlsx files should be predictions"). Also zips it.

Usage:
  CUDA_VISIBLE_DEVICES=1 conda run -n mo python3 predict.py
"""
import os
import zipfile
from collections import Counter

import pandas as pd
import sacrebleu
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

import config as cfg
from classifier_utils import PolarityClassifier
from data import load_test, load_train, load_val, prefix_for, target_polarity_for
from lexicons import apply_emoji_safety_net, apply_word_safety_net, load_emoji_flip_lexicon, load_word_flip_lexicon


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_retrieval_lookup(train_df, val_df):
    """source -> most-frequent target text seen for that exact source, pooling train+val."""
    combined = pd.concat([train_df[["source", "target"]], val_df[["source", "target"]]], ignore_index=True)
    lookup = {}
    for src, group in combined.groupby("source"):
        lookup[src] = group["target"].value_counts().idxmax()
    return lookup


@torch.no_grad()
def generate_candidates(model, tokenizer, sources, polarities, device,
                         num_candidates=cfg.NUM_CANDIDATES, batch_size=8):
    """Beam-search generation returning the top `num_candidates` beams per input.

    NOTE: this transformers version gates *diverse* (grouped) beam search behind a
    `trust_remote_code=True` remote-code plugin fetched from the Hub at call time --
    an extra network dependency we'd rather not add given the flaky CDN on this box.
    Plain beam search's top-N beams already give useful, if less diverse, candidates for
    the classifier+chrF reranking step below, which is where most of the benefit comes
    from anyway."""
    all_candidates = []
    for i in range(0, len(sources), batch_size):
        batch_src = sources[i:i + batch_size]
        batch_pol = polarities[i:i + batch_size]
        inputs = [prefix_for(p) + s for s, p in zip(batch_src, batch_pol)]
        enc = tokenizer(inputs, truncation=True, max_length=cfg.GENERATOR_MAX_SRC_LEN,
                         padding=True, return_tensors="pt").to(device)
        gen = model.generate(
            **enc,
            max_length=cfg.GENERATOR_MAX_TGT_LEN,
            num_beams=num_candidates,
            num_return_sequences=num_candidates,
            early_stopping=True,
        )
        decoded = tokenizer.batch_decode(gen, skip_special_tokens=True)
        for j in range(len(batch_src)):
            all_candidates.append(decoded[j * num_candidates:(j + 1) * num_candidates])
    return all_candidates


def generate_nllb_candidates(model, tokenizer, sources, polarities, device, num_candidates=3, batch_size=8):
    """Same idea as generate_candidates but for the NLLB backbone (src_lang=tgt_lang=
    arb_Arab, needs forced_bos_token_id). Fewer beams than mT5's pool (3 vs
    NUM_CANDIDATES=10): NLLB is here purely to add cross-model diversity, not to be the
    primary candidate source."""
    lang_token_id = tokenizer.convert_tokens_to_ids("arb_Arab")
    all_candidates = []
    for i in range(0, len(sources), batch_size):
        batch_src = sources[i:i + batch_size]
        batch_pol = polarities[i:i + batch_size]
        inputs = [prefix_for(p) + s for s, p in zip(batch_src, batch_pol)]
        enc = tokenizer(inputs, truncation=True, max_length=cfg.GENERATOR_MAX_SRC_LEN,
                         padding=True, return_tensors="pt").to(device)
        gen = model.generate(
            **enc, forced_bos_token_id=lang_token_id, max_length=cfg.GENERATOR_MAX_TGT_LEN,
            num_beams=num_candidates, num_return_sequences=num_candidates, early_stopping=True,
        )
        decoded = tokenizer.batch_decode(gen, skip_special_tokens=True)
        for j in range(len(batch_src)):
            all_candidates.append(decoded[j * num_candidates:(j + 1) * num_candidates])
    return all_candidates


def rerank(sources, candidates_per_source, target_polarities, classifier):
    """Pick, for each source, the candidate maximizing a weighted blend of
    (classifier confidence in the target polarity) and (chrF similarity to source).

    Polarity signal is raw_target_prob() (RAW 3-way softmax probability, NOT marginalizing
    out neutral) rather than target_prob(). Confirmed from the actual competition page: the
    official Sentiment Style Accuracy metric is "the percentage of outputs with the correct
    target polarity" via this exact classifier -- the natural reading is a raw 3-way argmax
    match (neutral counts as a miss). The old renormalized-binary signal marginalized
    neutral out entirely, so it could rate a secretly-neutral candidate as confidently
    "positive" or "negative" purely from the pos/neg ratio, over-crediting it. Diagnosed on
    v17's real test outputs (argmax_vs_renormalized_check.py): raw argmax's gap to the real
    official StyleAcc (94.36%) was +1.81pp vs. +3.21pp for the renormalized proxy -- a real
    but modest effect (only ~1.55% of outputs have a 'neutral' true argmax). This is still
    a CONTINUOUS score (not a hard gate), unlike v20_gated_rerank's hard-threshold mistake
    (see git history / SUBMISSIONS_LOG for that regression) -- it only changes how
    confidently we estimate "target polarity achieved" per candidate, using the more
    faithful (raw, non-renormalized) reading of the confirmed grading methodology."""
    chrf = sacrebleu.CHRF(word_order=2)
    best = []
    for src, cands, tgt_pol in zip(sources, candidates_per_source, target_polarities):
        cands = [c if c.strip() else src for c in cands]  # never allow an empty generation
        pol_probs = classifier.raw_target_prob(cands, [tgt_pol] * len(cands))
        content_scores = [chrf.sentence_score(c, [src]).score / 100.0 for c in cands]
        combined = [
            cfg.POLARITY_SCORE_WEIGHT * p + cfg.CONTENT_SCORE_WEIGHT * c
            for p, c in zip(pol_probs, content_scores)
        ]
        best_idx = max(range(len(cands)), key=lambda k: combined[k])
        best.append(cands[best_idx])
    return best


def main(generator_dir=None, output_subdir=None):
    device = get_device()
    print(f"Using device: {device}")

    train_df, val_df, test_df = load_train(), load_val(), load_test()
    retrieval_lookup = build_retrieval_lookup(train_df, val_df)
    emoji_lexicon = load_emoji_flip_lexicon(train_df)
    word_lexicon = load_word_flip_lexicon()

    gen_dir = generator_dir or os.path.join(cfg.CHECKPOINT_DIR, "generator")
    if not os.path.isdir(gen_dir):
        raise RuntimeError(f"No trained generator found under {gen_dir}. Run train_generator.py --mode final first.")
    tokenizer = AutoTokenizer.from_pretrained(gen_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(gen_dir).to(device).eval()

    classifier = PolarityClassifier(device=device)

    # Optional: dedicated source-polarity classifier (train_source_polarity_classifier.py),
    # fine-tuned on the 7716 labeled train+val (source, source_polarity) pairs -- used ONLY
    # for Stage 2 below (deciding which direction to swap). The official classifier's
    # zero-shot accuracy at this specific job, measured against real gold labels, is only
    # 74.31%; this dedicated one reaches 96.72% on the same held-out split. See
    # classifier_utils.py's module docstring for why this is a sound, different-in-kind
    # fix from "ensemble the reranking signal" (rejected earlier): source polarity is an
    # objective input fact, not something the grading classifier defines. Falls back to
    # the official classifier if this hasn't been trained yet.
    source_polarity_classifier = classifier
    try:
        from classifier_utils import DedicatedSourcePolarityClassifier
        source_polarity_classifier = DedicatedSourcePolarityClassifier(device=device)
        print("Loaded dedicated source-polarity classifier (96.72% held-out accuracy) for Stage 2.")
    except FileNotFoundError:
        print("No dedicated source-polarity classifier found; falling back to the official classifier "
              "zero-shot for Stage 2 (run train_source_polarity_classifier.py to enable).")

    # Optional: edit-tagging model (edit_tagger.py) adds one more minimal-edit candidate
    # per row into the reranking pool below -- see that module's docstring. Purely
    # additive: if its checkpoint doesn't exist yet, this whole system still works
    # exactly as before (mT5 candidates only).
    tagger_bundle = None
    try:
        import edit_tagger
        tagger_bundle = edit_tagger.load_tagger()
        print("Loaded edit-tagging model; its candidate will be added to the reranking pool.")
    except FileNotFoundError:
        print("No edit-tagging checkpoint found; skipping (run edit_tagger.py --mode train to enable).")

    # Optional: near-duplicate retrieval (retrieval_augment.py) adds the nearest train/val
    # target as one more candidate when a test source is highly similar (but not
    # byte-identical) to a training source -- see that module's docstring.
    import retrieval_augment
    retrieval_index = retrieval_augment.load_index_if_available()
    if retrieval_index is not None:
        print("Loaded near-duplicate retrieval index; its candidate will be added to the reranking pool.")
    else:
        print("No retrieval index found; skipping (run retrieval_augment.py to enable).")

    # Optional: NLLB second generator (train_nllb.py) -- architecturally different
    # backbone, adds up to 3 more candidates per row so reranking has genuinely
    # different-model options, not just different beams of the same mT5 model.
    nllb_bundle = None
    nllb_dir = os.path.join(cfg.CHECKPOINT_DIR, "generator_nllb")
    if os.path.isdir(nllb_dir):
        nllb_tokenizer = AutoTokenizer.from_pretrained(nllb_dir, src_lang="arb_Arab", tgt_lang="arb_Arab")
        nllb_model = AutoModelForSeq2SeqLM.from_pretrained(nllb_dir).to(device).eval()
        nllb_bundle = (nllb_model, nllb_tokenizer)
        print("Loaded NLLB second generator; its candidates will be added to the reranking pool.")
    else:
        print("No NLLB checkpoint found; skipping (run train_nllb.py to enable).")

    # Optional: larger mT5 backbone (mt5-large vs the base pipeline's mt5-base) -- same
    # tokenizer/prefix scheme as the primary generator, so it reuses generate_candidates
    # directly, just with fewer beams since it's here for capacity/diversity, not as the
    # primary candidate source.
    large_bundle = None
    large_dir = os.path.join(cfg.CHECKPOINT_DIR, "generator_large")
    if os.path.isdir(large_dir):
        large_tokenizer = AutoTokenizer.from_pretrained(large_dir)
        large_model = AutoModelForSeq2SeqLM.from_pretrained(large_dir).to(device).eval()
        large_bundle = (large_model, large_tokenizer)
        print("Loaded mt5-large second generator; its candidates will be added to the reranking pool.")
    else:
        print("No mt5-large checkpoint found; skipping (run train_generator.py --out_name generator_large to enable).")

    sources = test_df["source"].tolist()
    outputs = [None] * len(sources)
    method = [None] * len(sources)

    # Stage 1: exact-match retrieval
    to_generate_idx = []
    for i, src in enumerate(sources):
        if src in retrieval_lookup:
            outputs[i] = retrieval_lookup[src]
            method[i] = "retrieval"
        else:
            to_generate_idx.append(i)
    print(f"Exact-match retrieval covered {len(sources) - len(to_generate_idx)}/{len(sources)} test rows.")

    if to_generate_idx:
        gen_sources = [sources[i] for i in to_generate_idx]

        # Stage 2: test-time source-polarity detection. Blends the dedicated classifier
        # with the official classifier's zero-shot guess (50/50) when both are available --
        # held-out validated 96.72% (dedicated alone) -> 97.25% (blend) on the same split,
        # a small, real, essentially-free improvement (see classifier_utils.blended_source_polarity).
        # Falls back to the dedicated classifier alone, then the official classifier alone,
        # if either isn't available.
        if source_polarity_classifier is not classifier:
            from classifier_utils import blended_source_polarity
            pred_source_polarity = blended_source_polarity(source_polarity_classifier, classifier, gen_sources)
            del source_polarity_classifier
            torch.cuda.empty_cache()
        else:
            pred_source_polarity = source_polarity_classifier.predict_polarity(gen_sources)
        target_polarities = [target_polarity_for(p) for p in pred_source_polarity]

        # Stage 3: generate + classifier-guided rerank
        # Negative->Positive is a documented harder direction for sentiment style transfer
        # (euphemistic source phrasing needs more sentence-structure change than the
        # reverse -- see arxiv.org/html/2312.14708), and we measured the same asymmetry
        # on our own val set (chrF 84.27 Pos->Neg vs 78.94 Neg->Pos, StyleAcc 96.06% vs
        # 94.37%). Give the harder direction more beam-search budget -- the same lever
        # that worked for mt5-large's pool contribution, just targeted instead of uniform.
        HARD_DIRECTION_EXTRA_BEAMS = 3
        neg_idx = [i for i, p in enumerate(pred_source_polarity) if p == "Negative"]
        pos_idx = [i for i, p in enumerate(pred_source_polarity) if p != "Negative"]
        print(f"Generating {cfg.NUM_CANDIDATES + HARD_DIRECTION_EXTRA_BEAMS} candidates for "
              f"{len(neg_idx)} Negative->Positive rows (harder direction) and {cfg.NUM_CANDIDATES} for "
              f"{len(pos_idx)} Positive->Negative rows...")
        candidates = [None] * len(gen_sources)
        if neg_idx:
            neg_sources = [gen_sources[i] for i in neg_idx]
            neg_pol = [pred_source_polarity[i] for i in neg_idx]
            neg_cands = generate_candidates(model, tokenizer, neg_sources, neg_pol, device,
                                             num_candidates=cfg.NUM_CANDIDATES + HARD_DIRECTION_EXTRA_BEAMS)
            for local_i, global_i in enumerate(neg_idx):
                candidates[global_i] = neg_cands[local_i]
        if pos_idx:
            pos_sources = [gen_sources[i] for i in pos_idx]
            pos_pol = [pred_source_polarity[i] for i in pos_idx]
            pos_cands = generate_candidates(model, tokenizer, pos_sources, pos_pol, device,
                                             num_candidates=cfg.NUM_CANDIDATES)
            for local_i, global_i in enumerate(pos_idx):
                candidates[global_i] = pos_cands[local_i]

        if tagger_bundle is not None:
            tagger_model, tagger_tokenizer, replace_vocab = tagger_bundle
            tagger_candidates = edit_tagger.tag_and_reconstruct(gen_sources, tagger_model, tagger_tokenizer, replace_vocab)
            candidates = [cands + [tc] for cands, tc in zip(candidates, tagger_candidates)]

        if nllb_bundle is not None:
            nllb_model, nllb_tokenizer = nllb_bundle
            nllb_candidates = generate_nllb_candidates(nllb_model, nllb_tokenizer, gen_sources, pred_source_polarity, device)
            candidates = [cands + nc for cands, nc in zip(candidates, nllb_candidates)]

        if large_bundle is not None:
            large_model, large_tokenizer = large_bundle
            large_candidates = generate_candidates(large_model, large_tokenizer, gen_sources, pred_source_polarity,
                                                     device, num_candidates=6)
            candidates = [cands + lc for cands, lc in zip(candidates, large_candidates)]

        if retrieval_index is not None:
            nn_candidates, nn_sims = retrieval_index.query_nearest(gen_sources, threshold=0.85)
            n_added = sum(1 for c in nn_candidates if c is not None)
            print(f"Near-duplicate retrieval proposed a candidate for {n_added}/{len(gen_sources)} rows "
                  f"(similarity >= {retrieval_augment.SIMILARITY_THRESHOLD}).")
            candidates = [cands + ([nc] if nc is not None else []) for cands, nc in zip(candidates, nn_candidates)]

        best = rerank(gen_sources, candidates, target_polarities, classifier)

        # Stage 4: emoji + word-level lexicon safety nets
        best = [apply_emoji_safety_net(src, cand, emoji_lexicon) for src, cand in zip(gen_sources, best)]
        best = [apply_word_safety_net(src, cand, word_lexicon) for src, cand in zip(gen_sources, best)]

        for local_i, global_i in enumerate(to_generate_idx):
            outputs[global_i] = best[local_i]
            method[global_i] = "generated"

    method_counts = Counter(method)
    print(f"\nMethod breakdown: {dict(method_counts)}")

    out_dir = os.path.join(cfg.OUTPUT_DIR, output_subdir) if output_subdir else cfg.OUTPUT_DIR
    os.makedirs(out_dir, exist_ok=True)

    out_df = pd.DataFrame({"id": test_df["id"], "style": outputs})
    xlsx_path = os.path.join(out_dir, "predictions.xlsx")
    out_df.to_excel(xlsx_path, index=False)
    print(f"Wrote {xlsx_path}")

    # Keep a diagnostic copy including the source + method, for our own error analysis
    # (not part of the submission).
    diag_df = pd.DataFrame({"id": test_df["id"], "source": sources, "style": outputs, "method": method})
    diag_df.to_csv(os.path.join(out_dir, "predictions_diagnostic.csv"), index=False)

    zip_path = os.path.join(out_dir, "predictions.zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(xlsx_path, arcname="predictions.xlsx")
    print(f"Wrote {zip_path}  <-- upload this file to Codabench")
    return out_dir


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--generator_dir", default=None,
                         help="override checkpoints/generator, e.g. checkpoints/generator_dpo")
    parser.add_argument("--output_subdir", default=None,
                         help="write predictions under outputs/<this>/ instead of outputs/ directly")
    args = parser.parse_args()
    main(generator_dir=args.generator_dir, output_subdir=args.output_subdir)
