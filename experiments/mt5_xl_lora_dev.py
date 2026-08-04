"""
Subtask 2 -- dev-mode check: can a LoRA-adapted google/mt5-xl (3.7B params) be trained at
all on this shared box, and if so, is it worth pursuing as a candidate-pool contributor
(the established, validated pattern here -- see SUBMISSIONS_LOG.md v10/v11 for mt5-large --
is that NEW generators get ADDED to the reranking pool, not swapped in as the sole
generator)?

Why LoRA instead of full fine-tuning: three consecutive full-fine-tune attempts OOM'd with
an identical root cause -- fp32 weights (14.8GB) + fp32 grads (14.8GB) + 8-bit AdamW states
(7.4GB) already sums to ~37GB before any activations, against only ~42GB free per GPU
(both GPUs stably carry ~37-38GB of other-tenant usage). Freezing the base in bf16 (7.4GB,
no grad buffers needed for frozen params) and training only small LoRA adapter matrices
removes almost all of that footprint. mT5's training LR here (3e-4) is far above the LR
(1e-5) that caused the earlier bf16-weight-UNDERFLOW bug during DPO -- and moot anyway
since the base is frozen and never updated; only the (kept-precise) adapter params get
gradient updates.

This is a dev-mode check (train on `train` only, eval each epoch on `val`) to get a real
BLEU/chrF/StyleAcc signal cheaply before committing to a `final` (train+val) run and a
pool-integration change to predict.py.

Usage:
  CUDA_VISIBLE_DEVICES=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_TOKEN=... \
    conda run -n mo python3 mt5_xl_lora_dev.py
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import json
import os

import numpy as np
import sacrebleu
import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

import config as cfg
from data import SwapSeq2SeqDataset, load_train, load_val, prefix_for, target_polarity_for
from train_generator import make_compute_metrics, seed_everything

MODEL_NAME = "google/mt5-xl"
OUT_DIR = os.path.join(cfg.OUTPUT_DIR, "tmp_mt5_xl_lora_dev")


def main():
    seed_everything(cfg.SEED)
    train_df = load_train()
    val_df = load_val()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL_NAME, use_safetensors=False, torch_dtype=torch.bfloat16
    )
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()  # required so grad checkpointing works with a frozen base

    lora_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q", "v"],  # T5-style attention query/value projections
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    train_ds = SwapSeq2SeqDataset(train_df["source"], train_df["source_polarity"], train_df["target"], tokenizer)
    val_ds = SwapSeq2SeqDataset(val_df["source"], val_df["source_polarity"], val_df["target"], tokenizer)
    collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    args = Seq2SeqTrainingArguments(
        output_dir=OUT_DIR,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=2,
        learning_rate=1e-3,  # LoRA adapters conventionally use a higher LR than full FT
        num_train_epochs=4,
        weight_decay=cfg.WEIGHT_DECAY,
        warmup_ratio=cfg.WARMUP_RATIO,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="chrf",
        greater_is_better=True,
        predict_with_generate=True,
        generation_max_length=cfg.GENERATOR_MAX_TGT_LEN,
        generation_num_beams=4,
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

    print("\nGenerating on val for BLEU/chrF/StyleAcc...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    preds = []
    for i in range(0, len(val_df), 16):
        batch_src = val_df["source"].iloc[i:i + 16].tolist()
        batch_pol = val_df["source_polarity"].iloc[i:i + 16].tolist()
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
    print("\nmT5-XL LoRA dev-set metrics:", metrics)
    print("For comparison, current best generator (mt5-large pooled, v17 recipe): "
          "official StyleAcc=0.9436, BLEU=57.95, chrF=73.03 (see SUBMISSIONS_LOG.md)")
    with open(os.path.join(cfg.OUTPUT_DIR, "mt5_xl_lora_dev_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
