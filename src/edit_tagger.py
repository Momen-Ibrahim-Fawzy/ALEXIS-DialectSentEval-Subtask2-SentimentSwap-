"""
Subtask 2 -- edit-tagging model (LaserTagger/FELIX-style minimal-edit generation).

Motivation: the EDA (Sections 4-5, 7) found the sentiment swap is overwhelmingly a
*minimal-edit* operation -- median character-level similarity between source/target is
high, most pairs keep word count within +/-2, and the dominant edit pattern is a short
antonym/negation word swap. A full seq2seq model (our mT5 baseline) has to *learn* to
mostly-copy; a tagging model encodes that prior directly: for each source WORD, predict
KEEP / DELETE / REPLACE-with-a-specific-mined-word, then reconstruct the sentence. This
can't hallucinate new content and trivially preserves everything it doesn't explicitly
edit -- ideal for the chrF/BLEU content-preservation component of the metric.

Design choice: rather than a standalone system with its own fallback logic, the tagger's
reconstructed sentence is added as ONE MORE CANDIDATE into predict.py's existing
classifier+chrF reranking pool alongside the mT5 beam candidates. This reuses all
existing infrastructure and is low-risk (reranking already picks the best of several
candidates, so an extra candidate that "abstains" on hard examples never hurts overall,
it's simply out-competed by the other candidates for whichever fraction of the val set
was needed to learn the reranking's behavior).

Simplifications made for this to fit a tractable training scope (documented for the
paper): only SINGLE-SOURCE-WORD replacement spans are tagged (i.e. "replace this one
word with mined phrase X"); training examples whose required edit needs a true
insertion (target words with no source counterpart at all) are excluded from the
tagger's training set, since a per-source-token tagging scheme can't represent those --
those examples simply aren't available to the tagger and fall through to mT5 candidates
at inference. Word tokenization is whitespace-based (str.split()) for lossless
reconstruction via ' '.join().

Usage:
  CUDA_VISIBLE_DEVICES=0 conda run -n mo python3 edit_tagger.py --mode train
  CUDA_VISIBLE_DEVICES=0 conda run -n mo python3 edit_tagger.py --mode eval_val
"""
import argparse
import json
import os
import random
from collections import Counter
from difflib import SequenceMatcher

import numpy as np
import pandas as pd
import sacrebleu
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

import config as cfg
from data import load_train, load_val, target_polarity_for

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TAGGER_MODEL_NAME = "UBC-NLP/MARBERTv2"  # dialectal-Arabic-pretrained encoder, already cached/proven in Subtask 1
TAG_VOCAB_PATH = os.path.join(cfg.OUTPUT_DIR, "edit_tag_vocab.json")
TAGGER_CKPT_PATH = os.path.join(cfg.CHECKPOINT_DIR, "edit_tagger.pt")

KEEP, DELETE = 0, 1  # REPLACE tags start at 2


def seed_everything(seed=cfg.SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def merge_opcodes(opcodes):
    """Merge consecutive non-'equal' opcodes (replace/delete/insert) into single blocks
    spanning [i1,i2) source words / [j1,j2) target words. This lets a REPLACE be applied
    across a multi-word source span (tagged on its first word, remainder DELETE) and also
    absorbs inserts that are adjacent to a replace/delete into the same span -- e.g. a
    delete immediately followed by an insert (opcodes emit these separately) becomes one
    replace block. A block with zero source words (i2==i1, a pure insertion with no
    adjacent edit) has no source word to anchor a tag on and stays unrepresentable."""
    blocks = []
    cur = None
    for tag, i1, i2, j1, j2 in opcodes:
        if tag == "equal":
            if cur is not None:
                blocks.append(tuple(cur))
                cur = None
            continue
        if cur is None:
            cur = [i1, i2, j1, j2]
        else:
            cur[1] = i2
            cur[3] = j2
    if cur is not None:
        blocks.append(tuple(cur))
    return blocks


def mine_tag_vocab(pairs_df, top_k=800, min_count=2):
    """Mine (source_phrase -> target_phrase) replacement pairs via alignment, across ALL
    train+val pairs. Unlike the original single-source-word scheme, this allows
    multi-word source spans (merge_opcodes) so phrase-level swaps -- not just single-word
    antonym substitutions -- are representable; coverage on train+val jumps from ~8% to
    ~90% of pairs being *structurally* representable (though vocab size still caps how
    many are representable by a *learnable* fixed tag set -- see edit_tagger's docstring
    for the coverage trade-off measured before this change)."""
    counter = Counter()
    for src, tgt in zip(pairs_df["source"], pairs_df["target"]):
        sw, tw = str(src).split(), str(tgt).split()
        sm = SequenceMatcher(a=sw, b=tw)
        for i1, i2, j1, j2 in merge_opcodes(sm.get_opcodes()):
            if i2 - i1 == 0:
                continue  # pure insertion, no source anchor
            src_phrase = " ".join(sw[i1:i2])
            target_phrase = " ".join(tw[j1:j2])
            counter[(src_phrase, target_phrase)] += 1
    # keep each source phrase's single most common replacement
    best = {}
    for (src_p, tgt_p), c in sorted(counter.items(), key=lambda kv: -kv[1]):
        if src_p not in best and c >= min_count:
            best[src_p] = tgt_p
    # cap vocab size to the most frequent replacements
    top_items = sorted(best.items(), key=lambda kv: -counter[(kv[0], kv[1])])[:top_k]
    replace_vocab = [w for _, w in top_items]  # target words/phrases, tag id = 2 + index
    src_to_tagvalue = {src: tgt for src, tgt in top_items}
    return replace_vocab, src_to_tagvalue


def derive_tags(source, target, src_to_tagvalue, replace_vocab_index):
    """Returns (word_list, tag_list) or None if this pair needs a true insertion (no
    adjacent source words to anchor it) or a replacement span outside our mined
    vocabulary. A multi-word replace span is tagged REPLACE on its first source word and
    DELETE on the rest, so reconstruct() emits the mined target phrase exactly once."""
    sw, tw = str(source).split(), str(target).split()
    sm = SequenceMatcher(a=sw, b=tw)
    tags = [None] * len(sw)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            for i in range(i1, i2):
                tags[i] = KEEP
    for i1, i2, j1, j2 in merge_opcodes(sm.get_opcodes()):
        if i2 - i1 == 0:
            return None  # true insertion -- can't be represented per-source-token
        src_phrase = " ".join(sw[i1:i2])
        target_phrase = " ".join(tw[j1:j2])
        if src_to_tagvalue.get(src_phrase) != target_phrase:
            return None  # this specific span isn't in our mined vocabulary
        tags[i1] = 2 + replace_vocab_index[target_phrase]
        for i in range(i1 + 1, i2):
            tags[i] = DELETE
    if any(t is None for t in tags):
        return None
    return sw, tags


def reconstruct(words, tags, replace_vocab):
    out = []
    for w, t in zip(words, tags):
        if t == KEEP:
            out.append(w)
        elif t == DELETE:
            continue
        else:
            out.append(replace_vocab[t - 2])
    return " ".join(out)


class TaggingDataset(Dataset):
    def __init__(self, examples, tokenizer, max_length=96):
        self.examples = examples  # list of (words, tags)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        words, tags = self.examples[idx]
        enc = self.tokenizer(words, is_split_into_words=True, truncation=True,
                              max_length=self.max_length, padding="max_length")
        word_ids = enc.word_ids()
        label_ids = []
        prev_word_idx = None
        for wid in word_ids:
            if wid is None or wid == prev_word_idx:
                label_ids.append(-100)
            else:
                label_ids.append(tags[wid])
            prev_word_idx = wid
        item = {k: torch.tensor(v) for k, v in enc.items() if k != "overflow_to_sample_mapping"}
        item["labels"] = torch.tensor(label_ids, dtype=torch.long)
        return item


class TaggerModel(nn.Module):
    def __init__(self, model_name, num_tags, dropout=0.1):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden, num_tags)

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None):
        kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
        if token_type_ids is not None:
            kwargs["token_type_ids"] = token_type_ids
        hidden = self.encoder(**kwargs).last_hidden_state
        logits = self.classifier(self.dropout(hidden))
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100)
        return loss, logits


def build_training_examples(df, src_to_tagvalue, replace_vocab_index):
    examples = []
    for src, tgt in zip(df["source"], df["target"]):
        derived = derive_tags(src, tgt, src_to_tagvalue, replace_vocab_index)
        if derived is not None:
            examples.append(derived)
    return examples


def train_tagger(epochs=8, batch_size=16, lr=3e-5):
    seed_everything()
    train_df, val_df = load_train(), load_val()
    full_df = pd.concat([train_df, val_df], ignore_index=True)

    replace_vocab, src_to_tagvalue = mine_tag_vocab(full_df)
    replace_vocab_index = {w: i for i, w in enumerate(replace_vocab)}
    num_tags = 2 + len(replace_vocab)
    print(f"Mined replacement vocab: {len(replace_vocab)} entries -> {num_tags} tag classes")

    examples = build_training_examples(full_df, src_to_tagvalue, replace_vocab_index)
    print(f"{len(examples)}/{len(full_df)} pairs ({len(examples)/len(full_df):.1%}) are fully representable "
          f"by the tag vocabulary and usable for tagger training")

    os.makedirs(os.path.dirname(TAG_VOCAB_PATH), exist_ok=True)
    with open(TAG_VOCAB_PATH, "w", encoding="utf-8") as f:
        json.dump({"replace_vocab": replace_vocab}, f, ensure_ascii=False, indent=2)

    tokenizer = AutoTokenizer.from_pretrained(TAGGER_MODEL_NAME)
    model = TaggerModel(TAGGER_MODEL_NAME, num_tags).to(DEVICE)

    ds = TaggingDataset(examples, tokenizer)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = len(loader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, int(0.06 * total_steps), total_steps)

    model.train()
    for epoch in range(epochs):
        total_loss, n_batches = 0.0, 0
        for batch in loader:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            labels = batch.pop("labels")
            optimizer.zero_grad()
            loss, _ = model(**batch, labels=labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
            n_batches += 1
        print(f"[edit_tagger] epoch {epoch+1}/{epochs} loss={total_loss/n_batches:.4f}")

    os.makedirs(os.path.dirname(TAGGER_CKPT_PATH), exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "num_tags": num_tags, "model_name": TAGGER_MODEL_NAME}, TAGGER_CKPT_PATH)
    print(f"Saved tagger to {TAGGER_CKPT_PATH}")
    return model, tokenizer, replace_vocab


def load_tagger():
    with open(TAG_VOCAB_PATH, encoding="utf-8") as f:
        replace_vocab = json.load(f)["replace_vocab"]
    ckpt = torch.load(TAGGER_CKPT_PATH, map_location=DEVICE, weights_only=False)
    model = TaggerModel(ckpt["model_name"], ckpt["num_tags"]).to(DEVICE)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(ckpt["model_name"])
    return model, tokenizer, replace_vocab


@torch.no_grad()
def tag_and_reconstruct(sources, model, tokenizer, replace_vocab, batch_size=32):
    """Returns one candidate string per source (best-effort; may equal the source
    unchanged if the model tags everything KEEP)."""
    model.eval()
    outputs = []
    for i in range(0, len(sources), batch_size):
        batch_sources = sources[i:i + batch_size]
        word_lists = [str(s).split() for s in batch_sources]
        enc = tokenizer(word_lists, is_split_into_words=True, truncation=True, max_length=96,
                         padding=True, return_tensors="pt").to(DEVICE)
        _, logits = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"],
                           token_type_ids=enc.get("token_type_ids"))
        preds = logits.argmax(dim=-1).cpu().numpy()
        for b, words in enumerate(word_lists):
            word_ids = enc.word_ids(batch_index=b)
            tags = [None] * len(words)
            seen = set()
            for pos, wid in enumerate(word_ids):
                if wid is None or wid in seen:
                    continue
                seen.add(wid)
                tags[wid] = int(preds[b, pos])
            tags = [t if t is not None else KEEP for t in tags]
            outputs.append(reconstruct(words, tags, replace_vocab))
    return outputs


def eval_on_val():
    """Diagnostic: how good are the tagger's candidates on their own (before being
    pooled into reranking)? Only meaningful on the subset of val the tagger can actually
    attempt (i.e. every example, since at inference it always produces *some* output,
    even if just KEEP-everything)."""
    from classifier_utils import PolarityClassifier

    model, tokenizer, replace_vocab = load_tagger()
    val_df = load_val()
    preds = tag_and_reconstruct(val_df["source"].tolist(), model, tokenizer, replace_vocab)

    changed = sum(1 for s, p in zip(val_df["source"], preds) if s != p)
    print(f"Tagger changed {changed}/{len(val_df)} ({changed/len(val_df):.1%}) val sources")

    chrf = sacrebleu.CHRF(word_order=2)
    bleu = sacrebleu.corpus_bleu(preds, [val_df["target"].tolist()]).score
    chrf_score = chrf.corpus_score(preds, [val_df["target"].tolist()]).score

    classifier = PolarityClassifier(device=DEVICE)
    target_pols = [target_polarity_for(p) for p in val_df["source_polarity"]]
    style_probs = classifier.target_prob(preds, target_pols)
    style_acc = float(np.mean([p >= 0.5 for p in style_probs])) * 100

    metrics = {"bleu": bleu, "chrf": chrf_score, "sentiment_style_accuracy_pct": style_acc, "pct_changed": changed / len(val_df) * 100}
    print("Tagger-only (no reranking pool) val metrics:", metrics)
    with open(os.path.join(cfg.OUTPUT_DIR, "edit_tagger_val_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "eval_val"], required=True)
    parser.add_argument("--epochs", type=int, default=8)
    args = parser.parse_args()
    if args.mode == "train":
        train_tagger(epochs=args.epochs)
    else:
        eval_on_val()
