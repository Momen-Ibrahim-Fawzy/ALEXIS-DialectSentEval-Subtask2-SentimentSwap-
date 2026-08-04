"""
Wrapper around the OFFICIAL evaluation classifier
(CAMeL-Lab/bert-base-arabic-camelbert-da-sentiment).

Used for one purpose now (see EDA_REPORT.md section 11):
  1. Classifier-guided reranking of generated candidates at inference time (does a
     candidate actually flip to the target polarity, and how confidently) -- this MUST
     stay the official classifier since it IS the actual grading model; there's no "more
     correct" alternative to compare a generated candidate's polarity against.

NOTE ON SOURCE-POLARITY DETECTION: this classifier was ALSO originally used zero-shot to
guess a test source's starting polarity (the released test file has no source_polarity
column), on the theory that using the "exact grading model" keeps things self-consistent.
That reasoning doesn't actually hold: source polarity is an OBJECTIVE fact about the
input (Codabench holds the true hidden label and defines target = opposite of it), not
something the grading classifier defines -- unlike the reranking use above, where the
classifier's opinion of the OUTPUT is definitionally correct. Measured zero-shot accuracy
against real gold source_polarity (val set) was only 74.31%, i.e. up to ~1/4 of rows
could be optimized toward the wrong target entirely before generation even starts. See
train_source_polarity_classifier.py / DedicatedSourcePolarityClassifier below: a
classifier fine-tuned on the 7716 labeled (source, source_polarity) train+val pairs we
already have reaches 96.72% on the same held-out split (+22.41pp). Use THAT for source-
polarity detection; keep PolarityClassifier (this class) for reranking/scoring only.

NOTE: the model's raw id2label is {0: positive, 1: negative, 2: neutral}, i.e. it is
NOT strictly binary. Since MA'AKS `source_polarity` is only ever Positive/Negative
(see EDA section 3), we always make binary decisions by comparing only the
positive-vs-negative logits/probabilities and ignoring the neutral class -- this avoids
ties/indecision from the neutral class leaking into a task that has no neutral label.
"""
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer

import config as cfg

_POS_LABEL, _NEG_LABEL = "positive", "negative"


class PolarityClassifier:
    def __init__(self, model_name=cfg.EVAL_CLASSIFIER_MODEL, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device)
        self.model.eval()
        self.id2label = {int(k): v.lower() for k, v in self.model.config.id2label.items()}
        self.pos_idx = [i for i, l in self.id2label.items() if l.startswith("pos")][0]
        self.neg_idx = [i for i, l in self.id2label.items() if l.startswith("neg")][0]

    @torch.no_grad()
    def binary_probs(self, texts, batch_size=32):
        """Returns a tensor of shape (N, 2): renormalized [P(positive), P(negative)],
        marginalizing out the neutral class."""
        all_probs = []
        for i in range(0, len(texts), batch_size):
            batch = [str(t) for t in texts[i:i + batch_size]]
            enc = self.tokenizer(batch, truncation=True, max_length=128, padding=True, return_tensors="pt").to(self.device)
            logits = self.model(**enc).logits
            full_probs = F.softmax(logits, dim=-1)
            pos = full_probs[:, self.pos_idx]
            neg = full_probs[:, self.neg_idx]
            denom = (pos + neg).clamp_min(1e-6)
            all_probs.append(torch.stack([pos / denom, neg / denom], dim=1).cpu())
        return torch.cat(all_probs, dim=0)

    def predict_polarity(self, texts, batch_size=32):
        """Binary Positive/Negative label per text (string, matches MA'AKS casing)."""
        probs = self.binary_probs(texts, batch_size=batch_size)
        return ["Positive" if p[0] >= p[1] else "Negative" for p in probs]

    def target_prob(self, texts, target_polarities, batch_size=32):
        """P(classifier predicts `target_polarity`) for each (text, target_polarity) pair,
        marginalizing out neutral. NOTE: kept for backward compatibility with existing
        checks/dev-metrics, but Stage 3 reranking uses raw_target_prob() instead -- see
        that method's docstring for why (this renormalized version over-credits outputs
        whose true 3-way argmax is 'neutral')."""
        probs = self.binary_probs(texts, batch_size=batch_size)
        out = []
        for p, tgt in zip(probs, target_polarities):
            out.append(p[0].item() if str(tgt).lower().startswith("pos") else p[1].item())
        return out

    @torch.no_grad()
    def raw_target_prob(self, texts, target_polarities, batch_size=32):
        """P(classifier predicts `target_polarity`) computed from the RAW 3-way softmax,
        WITHOUT marginalizing out neutral. Confirmed from the actual competition page: the
        official Sentiment Style Accuracy metric is "the percentage of outputs with the
        correct target polarity" per this exact classifier -- the natural reading of that
        is a raw 3-way argmax match (neutral counts as a miss), not a renormalized binary
        decision. Diagnosed via argmax_vs_renormalized_check.py on v17's real test outputs:
        raw argmax's gap to real official StyleAcc (94.36%) was +1.81pp, vs. +3.21pp for the
        renormalized binary proxy used elsewhere in this file -- driven by the ~1.55% of
        outputs whose true best guess is 'neutral' but got inflated to 'correct' by
        marginalization. Used for Stage 3 reranking's polarity signal specifically; Stage 2
        (source-polarity detection on real, cleanly-labeled human text) is unaffected by
        this distinction and keeps using target_prob()/binary_probs()."""
        all_probs = []
        for i in range(0, len(texts), batch_size):
            batch = [str(t) for t in texts[i:i + batch_size]]
            enc = self.tokenizer(batch, truncation=True, max_length=128, padding=True, return_tensors="pt").to(self.device)
            logits = self.model(**enc).logits
            all_probs.append(F.softmax(logits, dim=-1).cpu())
        full_probs = torch.cat(all_probs, dim=0)
        out = []
        for i, tgt in enumerate(target_polarities):
            idx = self.pos_idx if str(tgt).lower().startswith("pos") else self.neg_idx
            out.append(full_probs[i, idx].item())
        return out


class _MeanPoolBinaryHead(nn.Module):
    def __init__(self, model_name, dropout=0.1):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden, 2)

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None):
        kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
        if token_type_ids is not None:
            kwargs["token_type_ids"] = token_type_ids
        out = self.encoder(**kwargs).last_hidden_state
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (out * mask).sum(1) / mask.sum(1).clamp_min(1e-6)
        return self.classifier(self.dropout(pooled))


class DedicatedSourcePolarityClassifier:
    """Fine-tuned on the 7716 labeled (source, source_polarity) train+val pairs -- use
    ONLY for Stage 2 source-polarity detection in predict.py, never for reranking/scoring
    (see module docstring for why those are different questions). 96.72% held-out
    accuracy vs the zero-shot official classifier's 74.31% on the same split."""

    CKPT_PATH = os.path.join(cfg.CHECKPOINT_DIR, "source_polarity_classifier.pt")

    def __init__(self, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = torch.load(self.CKPT_PATH, map_location=self.device, weights_only=False)
        self.tokenizer = AutoTokenizer.from_pretrained(ckpt["model_name"])
        self.model = _MeanPoolBinaryHead(ckpt["model_name"]).to(self.device)
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()

    @torch.no_grad()
    def binary_probs(self, texts, batch_size=32):
        """Returns (N,2) [P(positive), P(negative)] -- same convention as
        PolarityClassifier.binary_probs, so the two can be blended directly."""
        all_probs = []
        for i in range(0, len(texts), batch_size):
            batch = [str(t) for t in texts[i:i + batch_size]]
            enc = self.tokenizer(batch, truncation=True, max_length=96, padding=True, return_tensors="pt").to(self.device)
            logits = self.model(**enc)
            probs = F.softmax(logits, dim=-1).cpu()  # index 0=positive, 1=negative (training label convention)
            all_probs.append(probs)
        return torch.cat(all_probs, dim=0)

    def predict_polarity(self, texts, batch_size=32):
        """Binary Positive/Negative label per text (string, matches MA'AKS casing) --
        same interface as PolarityClassifier.predict_polarity."""
        probs = self.binary_probs(texts, batch_size=batch_size)
        return ["Positive" if p[0] >= p[1] else "Negative" for p in probs]


def blended_source_polarity(dedicated, official, texts, official_weight=0.5, batch_size=32):
    """Ensembles DedicatedSourcePolarityClassifier + the official classifier's zero-shot
    guess for Stage 2 source-polarity detection: held-out validated 96.72% (dedicated
    alone) -> 97.25% (50/50 blend) on the same split -- see src_polarity_ensemble_check.py.
    Small, real, essentially-free improvement, only for Stage 2 (never for reranking)."""
    dedicated_probs = dedicated.binary_probs(texts, batch_size=batch_size)
    official_probs = official.binary_probs(texts, batch_size=batch_size)
    blended = (1 - official_weight) * dedicated_probs + official_weight * official_probs
    return ["Positive" if p[0] >= p[1] else "Negative" for p in blended]
