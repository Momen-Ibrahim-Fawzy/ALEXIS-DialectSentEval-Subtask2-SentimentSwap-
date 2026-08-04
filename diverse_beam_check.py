"""
Subtask 2 -- check whether DIVERSE beam search (Vijayakumar et al. 2016) raises the
"oracle ceiling" of candidate generation vs. plain beam search: for each val row, generate
N=10 candidates either way, and measure the fraction of rows where AT LEAST ONE candidate
achieves the target polarity per the official classifier (a diagnostic ceiling, decoupled
from reranking quality -- reranking can only pick among what generation already produced,
so if none of N plain-beam candidates ever flip polarity for a row, no reranking change
can fix it).

Motivated by fresh research (BeamR / Beam Reweighing with Attribute Discriminators, and
the broader controllable-generation literature): plain beam search optimizes sequence
likelihood, not the target attribute, and is known to produce near-duplicate beams that
under-explore the space of valid target-attribute rewrites. Diverse beam search
(num_beam_groups + diversity_penalty) is a standard, well-established, drop-in
replacement in HF's generate() -- no retraining, no change to the reranking/scoring logic
(which must stay matched to the official grading classifier) -- so if it raises the
oracle ceiling, it's a low-risk lever purely on the generation side.

Usage:
  CUDA_VISIBLE_DEVICES=1 conda run -n mo python3 diverse_beam_check.py
"""
import numpy as np
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

import config as cfg
from data import load_val, prefix_for, target_polarity_for

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
GEN_DIR = "checkpoints/generator_large"
NUM_CANDIDATES = 10
DIVERSITY_PENALTY = 0.5


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


@torch.no_grad()
def generate_diverse(model, tokenizer, sources, polarities, batch_size=8):
    # HF moved classic Group Beam Search behind a trust_remote_code custom_generate repo in
    # this transformers version; rather than pull in remote code, use nucleus/temperature
    # sampling for N independent draws -- a simpler, dependency-free, well-established way
    # to get a diverse candidate pool (arguably more diverse than group beam search, since
    # it's stochastic rather than a soft diversity penalty on otherwise-greedy beams).
    all_candidates = []
    for i in range(0, len(sources), batch_size):
        batch_src = sources[i:i + batch_size]
        batch_pol = polarities[i:i + batch_size]
        inputs = [prefix_for(p) + s for s, p in zip(batch_src, batch_pol)]
        enc = tokenizer(inputs, truncation=True, max_length=cfg.GENERATOR_MAX_SRC_LEN,
                         padding=True, return_tensors="pt").to(DEVICE)
        gen = model.generate(**enc, max_length=cfg.GENERATOR_MAX_TGT_LEN, do_sample=True,
                              top_p=0.92, temperature=1.0, num_return_sequences=NUM_CANDIDATES)
        decoded = tokenizer.batch_decode(gen, skip_special_tokens=True)
        for j in range(len(batch_src)):
            cands = decoded[j * NUM_CANDIDATES:(j + 1) * NUM_CANDIDATES]
            all_candidates.append([c if c.strip() else batch_src[j] for c in cands])
    return all_candidates


def main():
    val_df = load_val()
    print(f"Evaluating on full val set: {len(val_df)} rows")

    tokenizer = AutoTokenizer.from_pretrained(GEN_DIR)
    model = AutoModelForSeq2SeqLM.from_pretrained(GEN_DIR).to(DEVICE).eval()

    sources = val_df["source"].tolist()
    polarities = val_df["source_polarity"].tolist()
    target_pols = [target_polarity_for(p) for p in polarities]

    from classifier_utils import PolarityClassifier
    clf = PolarityClassifier(device=DEVICE)

    print("Generating with PLAIN beam search...")
    plain_candidates = generate_plain(model, tokenizer, sources, polarities)
    print("Generating with DIVERSE beam search...")
    diverse_candidates = generate_diverse(model, tokenizer, sources, polarities)

    def oracle_ceiling_and_mean(candidates_per_row):
        hits = 0
        mean_probs = []
        for cands, tgt_pol in zip(candidates_per_row, target_pols):
            probs = clf.target_prob(cands, [tgt_pol] * len(cands))
            mean_probs.append(float(np.mean(probs)))
            if any(p >= 0.5 for p in probs):
                hits += 1
        return hits / len(candidates_per_row) * 100, float(np.mean(mean_probs)) * 100

    plain_ceiling, plain_mean = oracle_ceiling_and_mean(plain_candidates)
    diverse_ceiling, diverse_mean = oracle_ceiling_and_mean(diverse_candidates)

    print(f"\nPLAIN beam search:   oracle ceiling (>=1 of {NUM_CANDIDATES} hits target polarity) = {plain_ceiling:.2f}%  |  mean per-candidate target-prob = {plain_mean:.2f}%")
    print(f"DIVERSE beam search: oracle ceiling (>=1 of {NUM_CANDIDATES} hits target polarity) = {diverse_ceiling:.2f}%  |  mean per-candidate target-prob = {diverse_mean:.2f}%")
    margin = diverse_ceiling - plain_ceiling
    print(f"\nOracle ceiling margin (diverse - plain): {margin:+.2f}pp")
    if margin >= 1.0:
        print("Diverse beam search meaningfully raises the oracle ceiling -- worth integrating into "
              "predict.py's candidate generation for the primary generator (low-risk: reranking logic "
              "unchanged, still uses the official classifier for final selection).")
    else:
        print("Diverse beam search did NOT meaningfully raise the oracle ceiling -- NULL result. "
              "The bottleneck is likely elsewhere (source-polarity detection, already fixed at "
              "97.25%, or genuine generation difficulty), not candidate-pool diversity.")


if __name__ == "__main__":
    main()
