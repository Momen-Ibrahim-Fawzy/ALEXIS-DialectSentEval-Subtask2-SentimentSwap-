<p align="center">
  <img src="assets/ALEXIS_Logo.png" alt="ALEXIS team logo" width="170"/>
  &nbsp;&nbsp;&nbsp;&nbsp;
  <img src="assets/AI-Moment.png" alt="AI Moment" width="220"/>
</p>

# ALEXIS — Subtask 2: Arabic Sentiment Swap (DialectSentEval 2026)

Rewrites an Arabic sentence so it expresses the opposite sentiment polarity while
preserving its topic/meaning. This is the system code behind our DialectSentEval 2026
Subtask 2 system-description paper.

## Approach

**Generator**: `google/mt5-base`, fully fine-tuned (not zero/few-shot prompted). Two
Arabic-specific T5 checkpoints were tried first and rejected for concrete, reproducible
reasons (see `config.py`): `UBC-NLP/AraT5v2-base-1024`'s tokenizer ships in a legacy
format the installed `tokenizers` library can't parse, and `UBC-NLP/AraT5-base`'s
published weights are themselves broken (garbage output even zero-shot, verified
directly). mT5-base was verified to produce sane output and covers Arabic among its 101
pretraining languages.

The EDA and subsequent experimentation drove the rest of the design:

1. **The released test file has no `source_polarity` column** (unlike train/val), so it
   must be detected at inference time. We first tried the official grading model
   zero-shot for this (`classifier_utils.PolarityClassifier`, wrapping
   `CAMeL-Lab/bert-base-arabic-camelbert-da-sentiment`) — self-consistent with how
   submissions are scored, but only 74.31% accurate at this specific job when measured
   against real gold labels. This was the single largest bottleneck in the whole system
   (Sentiment Style Accuracy jumped 0.784→0.944 in one step once fixed). The deployed
   system instead uses `classifier_utils.DedicatedSourcePolarityClassifier` — fine-tuned
   on all 7,716 labeled (source, polarity) pairs, reaching 96.72% alone — blended 50/50
   with the official classifier's vote via `blended_source_polarity()` (97.25%), since the
   two err differently. The official classifier is kept unchanged for Stage 3+ reranking/
   scoring (below), since that must match the actual grading criterion — only source-side
   polarity detection was replaced, not the classifier defining "correct" output polarity.

2. **Exact-match retrieval shortcut** (`predict.py::build_retrieval_lookup`). A share of
   test sources are byte-identical to a train/val source with a known human-written
   target — used directly, no generation involved.

3. **Minimal-edit generation, not free paraphrasing.** EDA: the large majority of pairs
   keep word count within ±2 and median character-level similarity is high. mT5 is fully
   fine-tuned (rather than prompted) because the MA'AKS paper's own AceGPT/JAIS/Llama-3
   benchmarking found fine-tuning to be the strong regime for this exact task.

4. **Checkpoint selection by generated-text chrF, not eval_loss.** An early run selected
   the "best" checkpoint by teacher-forced eval_loss and got *worse* real generation
   quality (BLEU/chrF/StyleAcc) than a later, higher-loss checkpoint — teacher-forced loss
   and free-running generation quality don't track each other reliably here. `train_generator.py`'s
   `compute_metrics` now decodes actual beam-search generations and scores chrF every eval
   pass; `metric_for_best_model="chrf"` selects on that instead.

5. **Multi-candidate classifier-guided reranking.** At inference, `predict.py` beam-searches
   candidates from the mT5 generator, **plus** one candidate each from the edit-tagging
   model and the near-duplicate retrieval index (both described below, both
   additive/optional — the system works with just mT5 if their artifacts don't exist).
   Every candidate is scored by a weighted blend of (a) the official classifier's
   confidence the *target* polarity was reached, and (b) chrF similarity to the source
   (content preservation), and the best-scoring one is kept.

6. **Emoji + word-level lexicon safety nets** (`lexicons.py`). Force-flips any left-over
   source-polarity emoji or short antonym word using lexicons mined from the training data,
   if the model didn't already handle it — deterministic, high-precision, can't hurt an
   already-correct generation (only fires when the source-side item is present and the
   target-side counterpart isn't).

7. **Edit-tagging model** (`edit_tagger.py`) — a LaserTagger/FELIX-style minimal-edit
   generator: a MARBERTv2 token-classification head predicts KEEP / DELETE /
   REPLACE-with-a-specific-mined-word for each source word, then the sentence is
   reconstructed directly from the tags. Motivated by finding #3 above (this is
   overwhelmingly a minimal-edit task) — a tagging model encodes that prior structurally
   instead of hoping a seq2seq model learns to mostly-copy, and by construction can't
   hallucinate new content. Only representable for a subset of pairs (single-word
   antonym-style replacements), so it's added as one extra candidate into the reranking
   pool rather than a standalone system — it wins reranking only when its restricted edit
   is actually the right one, and never makes things worse otherwise.

8. **Near-duplicate retrieval augmentation** (`retrieval_augment.py`) — extends the
   exact-match idea using `BAAI/bge-m3` sentence embeddings: for a test source that's
   *not* byte-identical to any train/val source but is a close near-duplicate (cosine
   similarity ≥0.90 — the underlying social-media corpus has many minor-variant reposts),
   the near-duplicate's target is added as another reranking candidate.

## Files

| File | Purpose |
|---|---|
| `config.py` | paths, model choices, generation/reranking hyperparameters |
| `data.py` | dataset loading, T5 input/target formatting |
| `classifier_utils.py` | official-classifier wrapper: polarity detection + candidate scoring |
| `lexicons.py` | mined emoji-flip and word-flip lexicons + safety-net post-processing |
| `edit_tagger.py` | minimal-edit tagging model (`--mode {train, eval_val}`) |
| `retrieval_augment.py` | BGE-M3 near-duplicate retrieval index (`build_index`) |
| `train_generator.py` | `--mode {dev, final}` fine-tunes the mT5 generator |
| `train_nllb.py`, `mt5_xl_run.py`, `mt5_xl_lora_dev.py` | alternative generator backbones evaluated in the ablations |
| `dpo_finetune.py`, `dpo_v6_more_epochs.py` | preference-optimization fine-tuning stage |
| `tune_reranking.py`, `tune_reranking_v2.py` | grid-searches the reranking weight blend on val |
| `predict.py` | full pipeline → `outputs/predictions.xlsx` + `outputs/predictions.zip` |
| `log_submission.py` | snapshots each submission (predictions + config + metrics) |
| `SUBMISSIONS_LOG.md` | full experiment log: every technique tried, val/official metrics |
| `*_check.py` | individual ablation experiments referenced in `SUBMISSIONS_LOG.md` and in the paper's ablation table |

Model checkpoints, raw run outputs (including the retrieval index), and the released
shared-task data are not included in this repository (see `.gitignore`) —
`train_generator.py --mode final` and `edit_tagger.py --mode train` regenerate
checkpoints, `retrieval_augment.py` regenerates the retrieval index, and `predict.py`
writes `outputs/predictions.zip`.

## How to run

```bash
pip install -r requirements.txt
cd System   # this repo's root, once cloned
python3 train_generator.py --mode dev --epochs 16  # find the true chrF-best epoch
python3 train_generator.py --mode final --epochs N # trains + saves what predict.py uses
python3 edit_tagger.py --mode train                 # optional: enables the tagger candidate
python3 retrieval_augment.py                        # optional: enables near-dup retrieval candidate
python3 predict.py                                  # -> outputs/predictions.zip
python3 log_submission.py --tag <name> --note "..."  # snapshot before uploading
```

A CUDA GPU is strongly recommended (set `CUDA_VISIBLE_DEVICES` for your setup); see
`requirements.txt` for the exact package versions used in our experiments.

## Data

This repository does not redistribute the DialectSentEval 2026 shared-task dataset.
`config.py` expects the released train/val/test files under `../Data/`; obtain them from
the official shared task page.

## Submission format

Per the Codabench "Submission Guidelines" for the Evaluation (Final) phase: a ZIP
containing one `.xlsx` file, **both named `predictions`**, with columns **id** and
**style** (the generated text). `predict.py` writes exactly that to
`outputs/predictions.xlsx` / `outputs/predictions.zip`.

Official evaluation (per the task page): Sentiment Style Accuracy via
`CAMeL-Lab/bert-base-arabic-camelbert-da-sentiment` (Codabench labels this "Sentiment
Preservation" in the results panel — same metric), plus BLEU/chrF for content
preservation against the source.

## Related work

- Mughaus, Abudalfa, Luqman et al., *"MA'AKS: manually-curated parallel dataset for Arabic
  text sentiment swap"*, Language Resources and Evaluation (2026) — the dataset paper;
  code/prompts at github.com/sabudalfa/ArabicTextSentimentSwap. Benchmarked AceGPT, JAIS
  and Llama-3 zero-shot/few-shot/fine-tuned; fine-tuning was consistently the strongest
  regime, which is why this system fully fine-tunes rather than prompts.
- Classic text-style-transfer literature (PPLM and successors) establishes
  classifier-guided decoding as a standard technique for controllable sentiment
  generation; the reranking stage here is a beam-search-time instantiation of the same idea.
- Malmi et al. 2019 ("Encode, Tag, Realize"/LaserTagger) and Mallinson et al. 2020
  (FELIX): minimal-edit tagging as an alternative to full seq2seq generation for
  text-editing tasks where most of the input should be copied — the direct inspiration
  for `edit_tagger.py`, simplified to single-source-word replacement spans (see that
  file's docstring for the exact simplifications made).

## Citation

If you use this code, please cite our DialectSentEval 2026 system-description paper
(citation to be added once published).
