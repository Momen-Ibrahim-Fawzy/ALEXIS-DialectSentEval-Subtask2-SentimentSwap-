"""
One-off wrapper for the mT5-XL final-generator training attempt. Overrides
GENERATOR_MAX_SRC_LEN/TGT_LEN 96->64 in-process only (NOT persisted to config.py, which
other scripts/the deployed pipeline depend on at 96) to shave activation memory, on top of
GPU1 (currently ~42GB free vs GPU0's ~69GB already occupied by other tenants) and 8-bit
AdamW, after two prior attempts on a more contended device OOM'd by an identical, stable
~490MB shortfall.

Usage:
  CUDA_VISIBLE_DEVICES=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_TOKEN=... \
    conda run -n mo python3 mt5_xl_run.py
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import config as cfg

cfg.GENERATOR_MAX_SRC_LEN = 64
cfg.GENERATOR_MAX_TGT_LEN = 64

from train_generator import run_final

if __name__ == "__main__":
    run_final(
        epochs=3,
        model_name="google/mt5-xl",
        out_name="generator_xl",
        batch_size=1,
        grad_accum=8,
        optim="adamw_bnb_8bit",
    )
