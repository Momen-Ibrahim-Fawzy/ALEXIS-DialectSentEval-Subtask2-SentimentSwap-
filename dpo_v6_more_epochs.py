"""
One-off wrapper: DPO training with the corrected raw_target_prob() reward (same as v5),
but more epochs (12 instead of 7). v5's preference_accuracy was still climbing at epoch 7
(0.722->0.739, not plateaued) -- this exact pattern ("DPO hasn't plateaued, add more
epochs") is precisely what drove real wins earlier in this project's history (v6_dpo_v2
-> v7_dpo_v3, each a genuine epoch-count-driven improvement).

Usage:
  CUDA_VISIBLE_DEVICES=1 conda run -n mo python3 dpo_v6_more_epochs.py
"""
import dpo_finetune as dpo

dpo.DPO_EPOCHS = 12

if __name__ == "__main__":
    dpo.main(out_name="generator_dpo_v6", generator_dir=None, batch_size=dpo.DPO_BATCH_SIZE)
