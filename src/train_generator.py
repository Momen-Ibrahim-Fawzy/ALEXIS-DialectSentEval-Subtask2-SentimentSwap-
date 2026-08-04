"""
Subtask 2 -- fine-tune the seq2seq sentiment-swap generator (see config.GENERATOR_MODEL).

Why this architecture (see ../EDA/EDA_REPORT.md sections 4-5 and 8, and the research
notes in System/README.md): the swap is empirically a *minimal-edit* operation (median
character-level similarity between source/target is high, most pairs keep word count
within +/-2) applied to noisy social-media Arabic (emoji/hashtags/elongation). A T5-style
encoder-decoder that is fully fine-tuned on in-domain pairs is copy-friendly (attention can
learn to reproduce most of the input verbatim) and, per the MA'AKS paper's own benchmarking,
outperforms zero/few-shot prompting of generic LLMs on exactly this kind of task -- fine-tuning
was their strong setting, not zero-shot.

Two modes:
  dev    Train on `train` only, evaluate every epoch on `val` (BLEU/chrF + the official
         Sentiment Style Accuracy computed via CAMeL-Lab/bert-base-arabic-camelbert-da-sentiment).
         Checkpoints go to outputs/tmp_dev/ and are not required for the final submission --
         this run exists purely to pick the number of epochs / sanity-check the approach.
  final  Train on train+val combined (more data for the actual submission) for the epoch
         count chosen from the `dev` run, and persist to checkpoints/generator/.

Usage:
  CUDA_VISIBLE_DEVICES=1 conda run -n mo python3 train_generator.py --mode dev
  CUDA_VISIBLE_DEVICES=1 conda run -n mo python3 train_generator.py --mode final --epochs 8
"""
import argparse
import json
import os
import random
import shutil

import numpy as np
import sacrebleu
import torch
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed,
)

import config as cfg
from data import SwapSeq2SeqDataset, load_train, load_val, prefix_for, target_polarity_for


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    set_seed(seed)


def build_model_and_tokenizer(model_name=cfg.GENERATOR_MODEL):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # use_safetensors=False: UBC-NLP/AraT5-base only ships pytorch_model.bin, and letting
    # transformers attempt an on-the-fly safetensors auto-conversion hits a remote HF
    # conversion service that this box's flaky network resets. We already have the .bin
    # cached locally, so just load it directly.
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, use_safetensors=False)
    return model, tokenizer


def make_compute_metrics(tokenizer):
    """chrF on the actually-*generated* (free-running, beam-search) text, not the
    teacher-forced eval_loss. These turned out NOT to track each other in practice here
    (an 8-epoch run selecting on eval_loss picked epoch 3, which scored *worse* on every
    real generation metric -- BLEU/chrF/StyleAcc -- than the later epochs a fixed
    6-epoch run had used). chrF is cheap enough to compute every eval pass and is a much
    more direct proxy for what we actually care about."""
    chrf = sacrebleu.CHRF(word_order=2)

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        return {"chrf": chrf.corpus_score(decoded_preds, [decoded_labels]).score}

    return compute_metrics


def run_dev(epochs):
    seed_everything(cfg.SEED)
    train_df = load_train()
    val_df = load_val()

    model, tokenizer = build_model_and_tokenizer()
    train_ds = SwapSeq2SeqDataset(train_df["source"], train_df["source_polarity"], train_df["target"], tokenizer)
    val_ds = SwapSeq2SeqDataset(val_df["source"], val_df["source_polarity"], val_df["target"], tokenizer)

    collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    out_dir = os.path.join(cfg.OUTPUT_DIR, "tmp_dev")

    args = Seq2SeqTrainingArguments(
        output_dir=out_dir,
        per_device_train_batch_size=cfg.BATCH_SIZE,
        per_device_eval_batch_size=cfg.EVAL_BATCH_SIZE,
        gradient_accumulation_steps=cfg.GRAD_ACCUM_STEPS,
        gradient_checkpointing=True,
        learning_rate=cfg.LEARNING_RATE,
        num_train_epochs=epochs,
        weight_decay=cfg.WEIGHT_DECAY,
        warmup_ratio=cfg.WARMUP_RATIO,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        # Select the best checkpoint by generated-text chrF, not eval_loss -- see
        # make_compute_metrics() docstring for why loss-based selection was misleading.
        load_best_model_at_end=True,
        metric_for_best_model="chrf",
        greater_is_better=True,
        predict_with_generate=True,
        generation_max_length=cfg.GENERATOR_MAX_TGT_LEN,
        generation_num_beams=4,
        # T5-family models are well known to produce NaN losses under fp16 (LayerNorm/
        # logit overflow); bf16 (native on this box's A100s) avoids that entirely.
        bf16=torch.cuda.is_available(),
        logging_steps=50,
        report_to=[],
        seed=cfg.SEED,
    )
    trainer = Seq2SeqTrainer(
        model=model, args=args, train_dataset=train_ds, eval_dataset=val_ds,
        data_collator=collator, processing_class=tokenizer,
        compute_metrics=make_compute_metrics(tokenizer),
    )
    trainer.train()

    # Generate on val and compute BLEU/chrF + (approximate) Sentiment Style Accuracy.
    print("\nGenerating on val for qualitative + automatic metrics...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    preds = []
    for i in range(0, len(val_df), 32):
        batch_src = val_df["source"].iloc[i:i + 32].tolist()
        batch_pol = val_df["source_polarity"].iloc[i:i + 32].tolist()
        inputs = [prefix_for(p) + s for s, p in zip(batch_src, batch_pol)]
        enc = tokenizer(inputs, truncation=True, max_length=cfg.GENERATOR_MAX_SRC_LEN,
                         padding=True, return_tensors="pt").to(device)
        with torch.no_grad():
            gen = model.generate(**enc, max_length=cfg.GENERATOR_MAX_TGT_LEN, num_beams=4)
        preds.extend(tokenizer.batch_decode(gen, skip_special_tokens=True))

    refs = val_df["target"].tolist()
    bleu = sacrebleu.corpus_bleu(preds, [refs]).score
    chrf = sacrebleu.CHRF(word_order=2).corpus_score(preds, [refs]).score

    try:
        from classifier_utils import PolarityClassifier
        clf = PolarityClassifier(device=device)
        target_pols = [target_polarity_for(p) for p in val_df["source_polarity"]]
        target_probs = clf.target_prob(preds, target_pols)
        style_acc = float(np.mean([p >= 0.5 for p in target_probs])) * 100
    except Exception as e:
        print(f"[warn] could not compute Sentiment Style Accuracy: {e}")
        style_acc = None

    metrics = {"bleu": bleu, "chrf": chrf, "sentiment_style_accuracy_pct": style_acc}
    print("\nDev-set metrics:", metrics)
    with open(os.path.join(cfg.OUTPUT_DIR, "dev_generation_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    sample_out = val_df[["source", "source_polarity", "target"]].copy()
    sample_out["prediction"] = preds
    sample_out.to_csv(os.path.join(cfg.OUTPUT_DIR, "dev_predictions_sample.csv"), index=False)
    print(f"Wrote {os.path.join(cfg.OUTPUT_DIR, 'dev_predictions_sample.csv')}")

    # `dev` mode's checkpoints (incl. optimizer state, several GB for a model this size)
    # are only needed transiently to pick an epoch count -- not required for the final
    # submission model, which trains fresh in `final` mode. Free the disk immediately.
    shutil.rmtree(out_dir, ignore_errors=True)
    print(f"Cleaned up temporary checkpoint dir {out_dir}")


def run_final(epochs, model_name=cfg.GENERATOR_MODEL, out_name="generator", batch_size=cfg.BATCH_SIZE,
              grad_accum=cfg.GRAD_ACCUM_STEPS, optim="adamw_torch"):
    seed_everything(cfg.SEED)
    train_df = load_train()
    val_df = load_val()
    import pandas as pd
    full_df = pd.concat([train_df, val_df], ignore_index=True)
    print(f"Training final generator ({model_name}) on {len(full_df)} pairs (train+val combined)")

    model, tokenizer = build_model_and_tokenizer(model_name)
    full_ds = SwapSeq2SeqDataset(full_df["source"], full_df["source_polarity"], full_df["target"], tokenizer)
    collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    out_dir = os.path.join(cfg.OUTPUT_DIR, f"tmp_final_gen_{out_name}")

    args = Seq2SeqTrainingArguments(
        output_dir=out_dir,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=cfg.EVAL_BATCH_SIZE,
        gradient_accumulation_steps=grad_accum,
        gradient_checkpointing=True,
        learning_rate=cfg.LEARNING_RATE,
        num_train_epochs=epochs,
        weight_decay=cfg.WEIGHT_DECAY,
        warmup_ratio=cfg.WARMUP_RATIO,
        eval_strategy="no",
        save_strategy="no",
        predict_with_generate=True,
        # T5-family models are well known to produce NaN losses under fp16 (LayerNorm/
        # logit overflow); bf16 (native on this box's A100s) avoids that entirely.
        bf16=torch.cuda.is_available(),
        # 8-bit AdamW (bitsandbytes) cuts optimizer-state memory ~4x -- needed for models
        # meaningfully bigger than mt5-large (e.g. mt5-xl, 3.7B params), where plain fp32
        # AdamW states alone (~2*4 bytes/param) would be ~30GB, likely exceeding available
        # headroom on this shared, contended GPU even before weights/gradients/activations.
        optim=optim,
        logging_steps=50,
        report_to=[],
        seed=cfg.SEED,
    )
    trainer = Seq2SeqTrainer(model=model, args=args, train_dataset=full_ds, data_collator=collator, processing_class=tokenizer)
    trainer.train()

    final_dir = os.path.join(cfg.CHECKPOINT_DIR, out_name)
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"Saved final generator to {final_dir}")

    shutil.rmtree(out_dir, ignore_errors=True)
    print(f"Cleaned up temporary checkpoint dir {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["dev", "final"], required=True)
    parser.add_argument("--epochs", type=int, default=cfg.NUM_EPOCHS)
    parser.add_argument("--model_name", type=str, default=cfg.GENERATOR_MODEL)
    parser.add_argument("--out_name", type=str, default="generator")
    parser.add_argument("--batch_size", type=int, default=cfg.BATCH_SIZE)
    parser.add_argument("--grad_accum", type=int, default=cfg.GRAD_ACCUM_STEPS)
    parser.add_argument("--optim", type=str, default="adamw_torch", help="e.g. adamw_bnb_8bit for large models")
    args = parser.parse_args()

    if args.mode == "dev":
        run_dev(args.epochs)
    else:
        run_final(args.epochs, model_name=args.model_name, out_name=args.out_name,
                   batch_size=args.batch_size, grad_accum=args.grad_accum, optim=args.optim)
