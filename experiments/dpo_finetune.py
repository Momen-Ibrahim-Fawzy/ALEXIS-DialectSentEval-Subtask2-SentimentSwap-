"""
Subtask 2 -- DPO-style preference fine-tuning on top of the MLE-fine-tuned generator.

Rationale (see Subtask 1's experiment battery for the motivating precedent): the single
strongest technique there (FGM, +0.013-0.017 F1 over every architecture-only change) was
a *direct* training-time intervention on the actual failure mode, not a post-hoc
selection step. Our current Subtask 2 pipeline is the inverse: standard MLE fine-tuning
toward the single gold reference, then all the "does this actually match the grading
criteria" work happens only at inference time via reranking (classifier confidence +
chrF). This stage bakes that same reward signal into TRAINING itself:

  1. For each training example, sample K diverse candidates from the current
     (MLE-fine-tuned) policy model, plus the gold reference itself, forming a
     preference pool.
  2. Score every candidate in the pool by the same reward used for reranking (classifier
     confidence in the target polarity + chrF to source) -- so training and inference
     optimize the same thing.
  3. Take the highest-reward pool member as "chosen" and the lowest as "rejected", and
     take one DPO gradient step per example: push the policy's log-probability of the
     chosen sequence up relative to a frozen reference copy of the pre-DPO model, and
     the rejected sequence's down, exactly as in Rafailov et al. 2023.

Because gold is always in the pool, "chosen" is often gold itself (safe, grounded);
"rejected" is always a genuinely low-reward sample, giving a real contrastive signal.
This differs from (and is intended to complement, not replace) the plain MLE checkpoint:
predict.py can point at either checkpoints/generator (MLE-only) or
checkpoints/generator_dpo (this script's output), and both get submitted so we get real
signal on whether this helped rather than assuming it did.

Usage:
  CUDA_VISIBLE_DEVICES=1 conda run -n mo python3 dpo_finetune.py
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import copy
import os
import random

import numpy as np
import sacrebleu
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, get_linear_schedule_with_warmup

import config as cfg
from classifier_utils import PolarityClassifier
from data import load_train, load_val, prefix_for, target_polarity_for

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# v1 (2500/2 epochs) reached preference_accuracy ~0.54; v2 (4500/4 epochs) reached 0.619
# and was STILL climbing at the last epoch (0.522 -> 0.560 -> 0.605 -> 0.619, loss still
# dropping too, no plateau) -- and the real test result improved correspondingly
# (StyleAcc 0.7675 -> 0.7721, BLEU/chrF essentially unchanged). Since neither loss nor
# preference accuracy had plateaued, push epochs further again.
DPO_SUBSET_SIZE = 4500
NUM_SAMPLES = 4          # sampled candidates per example, plus gold = pool size 5
DPO_EPOCHS = 7
DPO_LR = 1e-5             # much smaller than the 3e-4 used for the original MLE fine-tune
DPO_BATCH_SIZE = 4
DPO_BETA = 0.1            # standard DPO temperature (Rafailov et al. 2023)


def seed_everything(seed=cfg.SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def sample_candidates(model, tokenizer, sources, polarities, device, num_samples=NUM_SAMPLES, batch_size=8):
    all_candidates = []
    for i in range(0, len(sources), batch_size):
        batch_src = sources[i:i + batch_size]
        batch_pol = polarities[i:i + batch_size]
        inputs = [prefix_for(p) + s for s, p in zip(batch_src, batch_pol)]
        enc = tokenizer(inputs, truncation=True, max_length=cfg.GENERATOR_MAX_SRC_LEN,
                         padding=True, return_tensors="pt").to(device)
        gen = model.generate(
            **enc, max_length=cfg.GENERATOR_MAX_TGT_LEN, do_sample=True, top_p=0.92, temperature=1.0,
            num_return_sequences=num_samples,
        )
        decoded = tokenizer.batch_decode(gen, skip_special_tokens=True)
        for j in range(len(batch_src)):
            cands = decoded[j * num_samples:(j + 1) * num_samples]
            all_candidates.append([c if c.strip() else batch_src[j] for c in cands])
    return all_candidates


@torch.no_grad()
def sample_nllb_candidates(sources, polarities, device, num_samples=2, batch_size=8):
    """Optional: if the NLLB second generator (train_nllb.py) exists, add its candidates
    into the DPO preference pool too -- not just at inference-time reranking (predict.py)
    but into the preference signal itself, so the policy can learn from a genuinely
    different model's phrasing, not only its own samples + gold."""
    nllb_dir = os.path.join(cfg.CHECKPOINT_DIR, "generator_nllb")
    if not os.path.isdir(nllb_dir):
        return None
    nllb_tokenizer = AutoTokenizer.from_pretrained(nllb_dir, src_lang="arb_Arab", tgt_lang="arb_Arab")
    nllb_model = AutoModelForSeq2SeqLM.from_pretrained(nllb_dir).to(device).eval()
    lang_id = nllb_tokenizer.convert_tokens_to_ids("arb_Arab")
    all_candidates = []
    for i in range(0, len(sources), batch_size):
        batch_src = sources[i:i + batch_size]
        batch_pol = polarities[i:i + batch_size]
        inputs = [prefix_for(p) + s for s, p in zip(batch_src, batch_pol)]
        enc = nllb_tokenizer(inputs, truncation=True, max_length=cfg.GENERATOR_MAX_SRC_LEN,
                              padding=True, return_tensors="pt").to(device)
        gen = nllb_model.generate(**enc, forced_bos_token_id=lang_id, max_length=cfg.GENERATOR_MAX_TGT_LEN,
                                   num_beams=num_samples, num_return_sequences=num_samples, early_stopping=True)
        decoded = nllb_tokenizer.batch_decode(gen, skip_special_tokens=True)
        for j in range(len(batch_src)):
            cands = decoded[j * num_samples:(j + 1) * num_samples]
            all_candidates.append([c if c.strip() else batch_src[j] for c in cands])
    del nllb_model
    torch.cuda.empty_cache()
    return all_candidates


SEMANTIC_WEIGHT = 0.15  # on top of the usual polarity/chrF reward -- see build_preference_pairs


@torch.no_grad()
def compute_semantic_scores(sources, pools, device):
    """BGE-M3 cosine similarity between each source and every candidate in its pool --
    a MEANING-based content signal, complementary to chrF's character-overlap: a fluent
    synonym substitution keeps meaning but can score low on chrF, and this reward term
    is meant to catch that case (see dpo_finetune module docstring)."""
    import retrieval_augment as ra
    model, tokenizer = ra._load_embedder()
    flat_texts, spans = [], []
    for src, pool in zip(sources, pools):
        start = len(flat_texts)
        flat_texts.append(src)
        flat_texts.extend(pool)
        spans.append((start, start + 1 + len(pool)))
    embeddings = ra.embed_texts(flat_texts, model, tokenizer, batch_size=64)
    del model
    torch.cuda.empty_cache()

    all_scores = []
    for start, end in spans:
        src_vec = embeddings[start]
        cand_vecs = embeddings[start + 1:end]
        sims = cand_vecs @ src_vec  # both L2-normalized -> cosine similarity, in [-1, 1]
        all_scores.append(((sims + 1) / 2).tolist())  # rescale to [0, 1] to match other reward terms
    return all_scores


def build_preference_pairs(sources, polarities, targets, sampled, classifier, nllb_sampled=None, device=None):
    """Returns (chosen_list, rejected_list): for each example, the highest- and
    lowest-reward member of {gold target} u {mT5-sampled candidates} u
    {NLLB-sampled candidates, if available}."""
    chrf = sacrebleu.CHRF(word_order=2)
    chosen, rejected = [], []
    nllb_sampled = nllb_sampled or [[] for _ in sources]
    pools = [[gold] + cands + nllb_cands for gold, cands, nllb_cands in zip(targets, sampled, nllb_sampled)]
    semantic_scores = compute_semantic_scores(sources, pools, device) if device is not None else [None] * len(sources)

    for src, pol, gold, cands, nllb_cands, pool, sem_scores in zip(
        sources, polarities, targets, sampled, nllb_sampled, pools, semantic_scores
    ):
        target_pol = target_polarity_for(pol)
        # raw_target_prob(), NOT target_prob(): see predict.py's rerank() docstring / v21's
        # SUBMISSIONS_LOG entry. target_prob() marginalizes out neutral and over-credits
        # secretly-neutral candidates; using it as the DPO reward would teach the generator
        # to prefer outputs that game that renormalized signal, not the real (raw 3-way
        # argmax) grading criterion.
        pol_scores = classifier.raw_target_prob(pool, [target_pol] * len(pool))
        content_scores = [chrf.sentence_score(c, [src]).score / 100.0 for c in pool]
        if sem_scores is not None:
            rewards = [cfg.POLARITY_SCORE_WEIGHT * p + cfg.CONTENT_SCORE_WEIGHT * c + SEMANTIC_WEIGHT * s
                       for p, c, s in zip(pol_scores, content_scores, sem_scores)]
        else:
            rewards = [cfg.POLARITY_SCORE_WEIGHT * p + cfg.CONTENT_SCORE_WEIGHT * c
                       for p, c in zip(pol_scores, content_scores)]
        best_idx = max(range(len(pool)), key=lambda k: rewards[k])
        worst_idx = min(range(len(pool)), key=lambda k: rewards[k])
        chosen.append(pool[best_idx])
        rejected.append(pool[worst_idx])
    return chosen, rejected


class PreferenceDataset(Dataset):
    def __init__(self, sources, polarities, chosen, rejected, tokenizer):
        self.sources = sources
        self.polarities = polarities
        self.chosen = chosen
        self.rejected = rejected
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.sources)

    def __getitem__(self, idx):
        inp = prefix_for(self.polarities[idx]) + str(self.sources[idx])
        enc = self.tokenizer(inp, truncation=True, max_length=cfg.GENERATOR_MAX_SRC_LEN, padding="max_length")
        chosen_enc = self.tokenizer(text_target=str(self.chosen[idx]), truncation=True,
                                     max_length=cfg.GENERATOR_MAX_TGT_LEN, padding="max_length")
        rejected_enc = self.tokenizer(text_target=str(self.rejected[idx]), truncation=True,
                                       max_length=cfg.GENERATOR_MAX_TGT_LEN, padding="max_length")
        item = {f"input_{k}": torch.tensor(v) for k, v in enc.items()}
        item.update({f"chosen_{k}": torch.tensor(v) for k, v in chosen_enc.items()})
        item.update({f"rejected_{k}": torch.tensor(v) for k, v in rejected_enc.items()})
        return item


def sequence_logprob(model, input_ids, attention_mask, label_ids):
    labels = label_ids.clone()
    labels[labels == model.config.pad_token_id] = -100
    out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    logits = out.logits.float()  # upcast before log_softmax: bf16 has ~2-3 decimal digits,
    # nowhere near enough precision for a sum over ~96 tokens whose difference (policy vs
    # ref, chosen vs rejected) is exactly the DPO training signal -- computing this in bf16
    # is catastrophic cancellation, not a minor rounding issue (verified: it silently kept
    # preference_accuracy at random chance for a full 7-epoch run before this fix).
    logp = F.log_softmax(logits, dim=-1)
    token_logp = torch.gather(logp, 2, labels.clamp(min=0).unsqueeze(-1)).squeeze(-1)  # (B, T)
    mask = (labels != -100).float()
    return (token_logp * mask).sum(dim=1)  # (B,) summed sequence log-prob


def dpo_loss(policy, ref, batch):
    input_ids, attn = batch["input_input_ids"].to(DEVICE), batch["input_attention_mask"].to(DEVICE)
    chosen_ids = batch["chosen_input_ids"].to(DEVICE)
    rejected_ids = batch["rejected_input_ids"].to(DEVICE)

    policy_chosen = sequence_logprob(policy, input_ids, attn, chosen_ids)
    policy_rejected = sequence_logprob(policy, input_ids, attn, rejected_ids)
    with torch.no_grad():
        ref_chosen = sequence_logprob(ref, input_ids, attn, chosen_ids)
        ref_rejected = sequence_logprob(ref, input_ids, attn, rejected_ids)

    policy_logratio = policy_chosen - policy_rejected
    ref_logratio = ref_chosen - ref_rejected
    logits = DPO_BETA * (policy_logratio - ref_logratio)
    loss = -F.logsigmoid(logits).mean()
    acc = (logits > 0).float().mean().item()
    return loss, acc


def main(out_name="generator_dpo", generator_dir=None, batch_size=DPO_BATCH_SIZE):
    seed_everything()
    gen_dir = generator_dir or os.path.join(cfg.CHECKPOINT_DIR, "generator")
    if not os.path.isdir(gen_dir):
        raise RuntimeError(f"No MLE-fine-tuned generator at {gen_dir}; run train_generator.py --mode final first.")

    # NOTE: an earlier version of this function loaded both models in bf16 to save memory
    # for the mt5-large case (two full 1.2B-param copies resident at once). That was a real
    # bug, not just a memory optimization: AdamW's DPO_LR=1e-5 update on a typical weight
    # magnitude is a ~0.02% relative change, well below bf16's ~0.39% relative-precision
    # floor (7 mantissa bits) -- every weight update silently rounded to zero, so the
    # policy never moved from the frozen reference (verified: preference_accuracy stayed
    # pinned at random chance ~0.48-0.50 across a full 7-epoch run, identical whether or
    # not the loss computation itself was upcast to fp32, since the actual PARAMETER
    # STORAGE was the bottleneck, not the loss math). Parameters need fp32 for small-LR
    # update precision; 8-bit AdamW (bitsandbytes, already proven working in this
    # environment for the NLLB-1.3B fix) controls the optimizer-state memory that bf16 was
    # originally trying to save instead.
    tokenizer = AutoTokenizer.from_pretrained(gen_dir)
    policy = AutoModelForSeq2SeqLM.from_pretrained(gen_dir).to(DEVICE)
    ref = AutoModelForSeq2SeqLM.from_pretrained(gen_dir).to(DEVICE)
    ref.eval()
    for p in ref.parameters():
        p.requires_grad_(False)

    classifier = PolarityClassifier(device=DEVICE)

    train_df = load_train()
    val_df = load_val()
    full_df = train_df.sample(min(DPO_SUBSET_SIZE, len(train_df)), random_state=cfg.SEED).reset_index(drop=True)
    print(f"DPO preference-pair mining on {len(full_df)} training examples "
          f"({NUM_SAMPLES} sampled candidates + gold each)...")

    policy.eval()
    sampled = sample_candidates(policy, tokenizer, full_df["source"].tolist(), full_df["source_polarity"].tolist(), DEVICE)
    nllb_sampled = sample_nllb_candidates(full_df["source"].tolist(), full_df["source_polarity"].tolist(), DEVICE)
    if nllb_sampled is not None:
        print("Including NLLB-generated candidates in the preference pool too.")
    chosen, rejected = build_preference_pairs(
        full_df["source"].tolist(), full_df["source_polarity"].tolist(), full_df["target"].tolist(), sampled,
        classifier, nllb_sampled=nllb_sampled, device=DEVICE,
    )
    n_gold_chosen = sum(1 for c, g in zip(chosen, full_df["target"]) if c == g)
    print(f"Preference pairs built. Gold was the chosen (highest-reward) example in "
          f"{n_gold_chosen}/{len(full_df)} ({n_gold_chosen/len(full_df):.1%}) cases "
          f"(the rest: a sampled candidate scored even higher than gold).")

    ds = PreferenceDataset(full_df["source"].tolist(), full_df["source_polarity"].tolist(), chosen, rejected, tokenizer)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

    try:
        import bitsandbytes as bnb
        optimizer = bnb.optim.AdamW8bit(policy.parameters(), lr=DPO_LR, weight_decay=0.0)
    except ImportError:
        optimizer = torch.optim.AdamW(policy.parameters(), lr=DPO_LR, weight_decay=0.0)
    total_steps = len(loader) * DPO_EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, int(0.06 * total_steps), total_steps)

    policy.train()
    for epoch in range(DPO_EPOCHS):
        total_loss, total_acc, n = 0.0, 0.0, 0
        for batch in loader:
            optimizer.zero_grad()
            loss, acc = dpo_loss(policy, ref, batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
            total_acc += acc
            n += 1
        print(f"[dpo] epoch {epoch+1}/{DPO_EPOCHS} loss={total_loss/n:.4f} "
              f"preference_accuracy={total_acc/n:.3f} (fraction of batches where policy correctly prefers chosen)")

    out_dir = os.path.join(cfg.CHECKPOINT_DIR, out_name)
    policy.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    print(f"Saved DPO-tuned generator to {out_dir}")

    # Quick val sanity check so we know before submitting whether this plausibly helped.
    print("\nSanity-checking on val (beam=4, no reranking pool)...")
    policy.eval()
    preds = []
    for i in range(0, len(val_df), 32):
        batch_src = val_df["source"].iloc[i:i + 32].tolist()
        batch_pol = val_df["source_polarity"].iloc[i:i + 32].tolist()
        inputs = [prefix_for(p) + s for s, p in zip(batch_src, batch_pol)]
        enc = tokenizer(inputs, truncation=True, max_length=cfg.GENERATOR_MAX_SRC_LEN, padding=True, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            gen = policy.generate(**enc, max_length=cfg.GENERATOR_MAX_TGT_LEN, num_beams=4)
        preds.extend(tokenizer.batch_decode(gen, skip_special_tokens=True))
    refs = val_df["target"].tolist()
    bleu = sacrebleu.corpus_bleu(preds, [refs]).score
    chrf_score = sacrebleu.CHRF(word_order=2).corpus_score(preds, [refs]).score
    target_pols = [target_polarity_for(p) for p in val_df["source_polarity"]]
    style_probs = classifier.target_prob(preds, target_pols)
    style_acc = float(np.mean([p >= 0.5 for p in style_probs])) * 100
    print(f"DPO-tuned val metrics: BLEU={bleu:.2f} chrF={chrf_score:.2f} StyleAcc={style_acc:.2f}%")

    import json
    with open(os.path.join(cfg.OUTPUT_DIR, f"dpo_val_metrics_{out_name}.json"), "w", encoding="utf-8") as f:
        json.dump({"bleu": bleu, "chrf": chrf_score, "sentiment_style_accuracy_pct": style_acc,
                    "gold_chosen_fraction": n_gold_chosen / len(full_df)}, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_name", default="generator_dpo", help="checkpoint dir name under checkpoints/")
    parser.add_argument("--generator_dir", default=None, help="override checkpoints/generator, e.g. checkpoints/generator_large")
    parser.add_argument("--batch_size", type=int, default=DPO_BATCH_SIZE)
    args = parser.parse_args()
    main(out_name=args.out_name, generator_dir=args.generator_dir, batch_size=args.batch_size)
