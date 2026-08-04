"""
Subtask 2 -- empirically check whether BGE-M3 semantic similarity-to-source is a BETTER
proxy for the thing we actually want (chrF-to-GOLD-TARGET, unobservable at test time)
than the current reranking content signal (chrF-to-source), on val data where gold
targets exist. Research motivation: for sentiment/formality style transfer specifically,
semantic-embedding metrics (BERTScore-like) correlate better with human judgment of
content preservation than surface n-gram overlap (BLEU/chrF) -- but the OFFICIAL grading
metric IS chrF/BLEU, not BERTScore, so this must be verified as a better PROXY for that
graded quantity, not just "more human-like" in the abstract, before touching reranking.

Usage:
  CUDA_VISIBLE_DEVICES=0 conda run -n mo python3 content_proxy_check.py
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import numpy as np
import sacrebleu
import torch
import torch.nn.functional as F
from scipy.stats import pearsonr, spearmanr
from transformers import AutoModel, AutoModelForSeq2SeqLM, AutoTokenizer

import config as cfg
from data import load_val, prefix_for

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def generate_candidates(model, tokenizer, sources, polarities, device, num_candidates=10, batch_size=8):
    all_candidates = []
    for i in range(0, len(sources), batch_size):
        batch_src = sources[i:i + batch_size]
        batch_pol = polarities[i:i + batch_size]
        inputs = [prefix_for(p) + s for s, p in zip(batch_src, batch_pol)]
        enc = tokenizer(inputs, truncation=True, max_length=cfg.GENERATOR_MAX_SRC_LEN,
                         padding=True, return_tensors="pt").to(device)
        gen = model.generate(**enc, max_length=cfg.GENERATOR_MAX_TGT_LEN, num_beams=num_candidates,
                              num_return_sequences=num_candidates, early_stopping=True)
        decoded = tokenizer.batch_decode(gen, skip_special_tokens=True)
        for j in range(len(batch_src)):
            cands = decoded[j * num_candidates:(j + 1) * num_candidates]
            all_candidates.append([c if c.strip() else batch_src[j] for c in cands])
    return all_candidates


@torch.no_grad()
def embed_texts(texts, model, tokenizer, batch_size=64, max_length=64):
    all_vecs = []
    for i in range(0, len(texts), batch_size):
        batch = [str(t) for t in texts[i:i + batch_size]]
        enc = tokenizer(batch, truncation=True, max_length=max_length, padding=True, return_tensors="pt").to(DEVICE)
        out = model(**enc).last_hidden_state[:, 0]
        out = F.normalize(out, dim=-1)
        all_vecs.append(out.cpu().numpy())
    return np.concatenate(all_vecs, axis=0)


def main():
    val_df = load_val()
    N_ROWS = 200  # subsample for speed -- this is a correlation diagnostic, not a submission
    val_df = val_df.sample(N_ROWS, random_state=cfg.SEED).reset_index(drop=True)

    gen_dir = "checkpoints/generator_dpo_v3"
    tokenizer = AutoTokenizer.from_pretrained(gen_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(gen_dir).to(DEVICE).eval()

    sources = val_df["source"].tolist()
    targets = val_df["target"].tolist()
    polarities = val_df["source_polarity"].tolist()

    print(f"Generating 10 candidates each for {len(sources)} val rows...")
    candidates = generate_candidates(model, tokenizer, sources, polarities, DEVICE)
    del model
    torch.cuda.empty_cache()

    embed_tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-m3")
    embed_model = AutoModel.from_pretrained("BAAI/bge-m3").to(DEVICE).eval()

    chrf = sacrebleu.CHRF(word_order=2)

    chrf_to_source_all, sim_to_source_all, chrf_to_target_all = [], [], []
    for src, tgt, cands in zip(sources, targets, candidates):
        cand_embs = embed_texts(cands, embed_model, embed_tokenizer)
        src_emb = embed_texts([src], embed_model, embed_tokenizer)[0]
        sims = cand_embs @ src_emb
        for c, sim in zip(cands, sims):
            chrf_to_source_all.append(chrf.sentence_score(c, [src]).score)
            chrf_to_target_all.append(chrf.sentence_score(c, [tgt]).score)
            sim_to_source_all.append(float(sim))

    chrf_to_source_all = np.array(chrf_to_source_all)
    sim_to_source_all = np.array(sim_to_source_all)
    chrf_to_target_all = np.array(chrf_to_target_all)

    r_chrf, p1 = pearsonr(chrf_to_source_all, chrf_to_target_all)
    r_sim, p2 = pearsonr(sim_to_source_all, chrf_to_target_all)
    rho_chrf, _ = spearmanr(chrf_to_source_all, chrf_to_target_all)
    rho_sim, _ = spearmanr(sim_to_source_all, chrf_to_target_all)

    print(f"\n{len(chrf_to_source_all)} (candidate, row) pairs analyzed")
    print(f"\nCorrelation with chrF-to-GOLD-TARGET (the thing we actually want to predict):")
    print(f"  chrF-to-source     (current proxy):  Pearson r={r_chrf:.4f}  Spearman rho={rho_chrf:.4f}")
    print(f"  BGE-M3 sim-to-source (candidate):     Pearson r={r_sim:.4f}  Spearman rho={rho_sim:.4f}")

    if r_sim > r_chrf + 0.03:
        print("\nBGE-M3 similarity is a MEANINGFULLY BETTER proxy -- worth testing in reranking.")
    elif r_sim > r_chrf:
        print("\nBGE-M3 similarity is a marginally better proxy -- small potential upside, low priority.")
    else:
        print("\nchrF-to-source remains AT LEAST AS GOOD a proxy -- no evidence to switch reranking's content signal.")


if __name__ == "__main__":
    main()
