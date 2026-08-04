"""
Subtask 2 -- second generator backbone: NLLB-200-distilled-600M, fine-tuned as a
monolingual Arabic->Arabic "translator" (src_lang=tgt_lang=arb_Arab) conditioned on the
same instructional prefix mT5 uses.

Motivation: every Subtask 2 experiment so far (reranking weight tuning, DPO) has moved
along the SAME axis -- how aggressively the selection/reward favors polarity-flip
confidence over content preservation -- and that axis has a clear, consistent,
monotonic trade-off (more polarity focus = higher Sentiment Style Accuracy, lower
BLEU/chrF; confirmed across v1/v2/v3_dpo_tuned). A second, architecturally different
generator is a genuinely different lever: if NLLB's candidates are good, pooling them
into the SAME reranking mechanism as mT5's gives the reranker strictly more/better
options to choose from, which can push the Pareto frontier itself outward (find
candidates that are simultaneously good on both axes) rather than only sliding along
the frontier mT5 alone defines.

Usage:
  CUDA_VISIBLE_DEVICES=0 conda run -n mo python3 train_nllb.py
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import os
import random

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed,
)

import config as cfg
from data import load_train, load_val, prefix_for

NLLB_MODEL = "facebook/nllb-200-distilled-600M"


def seed_everything(seed=cfg.SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    set_seed(seed)


class NLLBSeq2SeqDataset(Dataset):
    def __init__(self, sources, polarities, targets, tokenizer, max_src_len=cfg.GENERATOR_MAX_SRC_LEN,
                 max_tgt_len=cfg.GENERATOR_MAX_TGT_LEN):
        self.sources = list(sources)
        self.polarities = list(polarities)
        self.targets = list(targets) if targets is not None else None
        self.tokenizer = tokenizer
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len

    def __len__(self):
        return len(self.sources)

    def __getitem__(self, idx):
        text = prefix_for(self.polarities[idx]) + str(self.sources[idx])
        enc = self.tokenizer(text, truncation=True, max_length=self.max_src_len, padding="max_length")
        item = {k: torch.tensor(v) for k, v in enc.items()}
        if self.targets is not None:
            tgt_enc = self.tokenizer(text_target=str(self.targets[idx]), truncation=True,
                                      max_length=self.max_tgt_len, padding="max_length")
            item["labels"] = torch.tensor(tgt_enc["input_ids"])
        return item


def build_model_and_tokenizer(model_name=NLLB_MODEL):
    tokenizer = AutoTokenizer.from_pretrained(model_name, src_lang="arb_Arab", tgt_lang="arb_Arab")
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    model.generation_config.forced_bos_token_id = tokenizer.convert_tokens_to_ids("arb_Arab")
    return model, tokenizer


def run_final(epochs=6, model_name=NLLB_MODEL, out_name="generator_nllb",
              batch_size=cfg.BATCH_SIZE, grad_accum=cfg.GRAD_ACCUM_STEPS):
    seed_everything()
    train_df = load_train()
    val_df = load_val()
    full_df = train_df._append(val_df, ignore_index=True) if hasattr(train_df, "_append") else \
        __import__("pandas").concat([train_df, val_df], ignore_index=True)

    model, tokenizer = build_model_and_tokenizer(model_name)
    full_ds = NLLBSeq2SeqDataset(full_df["source"], full_df["source_polarity"], full_df["target"], tokenizer)
    collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    out_dir = os.path.join(cfg.OUTPUT_DIR, f"tmp_nllb_{out_name}")

    args = Seq2SeqTrainingArguments(
        output_dir=out_dir,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        gradient_checkpointing=True,
        learning_rate=cfg.LEARNING_RATE,
        num_train_epochs=epochs,
        weight_decay=cfg.WEIGHT_DECAY,
        warmup_ratio=cfg.WARMUP_RATIO,
        save_strategy="no",
        bf16=torch.cuda.is_available(),
        # 8-bit AdamW (bitsandbytes): plain AdamW keeps optimizer moments in fp32 regardless
        # of bf16 training, a FIXED cost of ~3x model size in bytes that OOMs at the very
        # first optimizer.step() (before any batch even runs) once the model is large enough
        # -- this is what killed nllb-200-distilled-1.3B on this shared, contended box even
        # at batch_size=1. 8-bit optimizer states cut that fixed cost ~4x.
        optim="adamw_bnb_8bit",
        logging_steps=50,
        report_to=[],
        seed=cfg.SEED,
    )
    trainer = Seq2SeqTrainer(model=model, args=args, train_dataset=full_ds, data_collator=collator, processing_class=tokenizer)
    trainer.train()

    final_dir = os.path.join(cfg.CHECKPOINT_DIR, out_name)
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"Saved final NLLB generator to {final_dir}")

    import shutil
    shutil.rmtree(out_dir, ignore_errors=True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--model_name", type=str, default=NLLB_MODEL)
    parser.add_argument("--out_name", type=str, default="generator_nllb")
    parser.add_argument("--batch_size", type=int, default=cfg.BATCH_SIZE)
    parser.add_argument("--grad_accum", type=int, default=cfg.GRAD_ACCUM_STEPS)
    args = parser.parse_args()
    run_final(epochs=args.epochs, model_name=args.model_name, out_name=args.out_name,
               batch_size=args.batch_size, grad_accum=args.grad_accum)
