"""
Subtask 2 -- check whether ensembling the dedicated source-polarity classifier (96.72%
held-out accuracy) with the official classifier's zero-shot guess (74.31%) for Stage 2
source-polarity detection beats the dedicated classifier alone. They're trained very
differently (fine-tuned on this exact domain's 7716 labeled pairs vs. generic zero-shot),
so may make different errors; averaging their confidence could push past 96.72%.

Usage:
  CUDA_VISIBLE_DEVICES=0 conda run -n mo python3 src_polarity_ensemble_check.py
"""
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GroupShuffleSplit

import config as cfg
from data import load_train, load_val
from classifier_utils import PolarityClassifier, _MeanPoolBinaryHead
from transformers import AutoTokenizer

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# IMPORTANT: use the checkpoint trained on only 90% (source_polarity_ft_check.pt), NOT the
# final deployed one (checkpoints/source_polarity_classifier.pt, trained on 100% of data)
# -- the deployed one has already seen this holdout during its final training pass, which
# would make any "held-out" evaluation of it meaningless (trivially ~100%, pure leakage).
HELDOUT_CKPT_PATH = "outputs/source_polarity_ft_check.pt"


def main():
    train_df, val_df = load_train(), load_val()
    full_df = pd.concat([train_df, val_df], ignore_index=True).reset_index(drop=True)

    # Same held-out split used to validate the dedicated classifier
    gss = GroupShuffleSplit(n_splits=1, test_size=0.10, random_state=cfg.SEED)
    _, holdout_idx = next(gss.split(full_df, groups=full_df["source"]))
    holdout_df = full_df.iloc[holdout_idx].reset_index(drop=True)
    print(f"Holdout: {len(holdout_df)} rows")

    official = PolarityClassifier(device=DEVICE)
    ckpt = torch.load(HELDOUT_CKPT_PATH, map_location=DEVICE, weights_only=False)
    dedicated_tokenizer = AutoTokenizer.from_pretrained(ckpt["model_name"])
    dedicated_model = _MeanPoolBinaryHead(ckpt["model_name"]).to(DEVICE)
    dedicated_model.load_state_dict(ckpt["state_dict"])
    dedicated_model.eval()

    official_probs = official.binary_probs(holdout_df["source"].tolist()).numpy()  # (N,2) [pos, neg]
    official_pred = ["Positive" if p[0] >= p[1] else "Negative" for p in official_probs]

    # Dedicated classifier: get logits->probs directly for ensembling
    import torch.nn.functional as F
    all_probs = []
    for i in range(0, len(holdout_df), 32):
        batch = [str(t) for t in holdout_df["source"].iloc[i:i + 32].tolist()]
        enc = dedicated_tokenizer(batch, truncation=True, max_length=96, padding=True, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            logits = dedicated_model(**enc)
        probs = F.softmax(logits, dim=-1).cpu()  # index 0=positive, 1=negative (see training script)
        all_probs.append(probs)
    dedicated_probs = torch.cat(all_probs, dim=0).numpy()  # (N,2) [pos, neg]
    dedicated_pred = ["Positive" if p[0] >= p[1] else "Negative" for p in dedicated_probs]

    gold = holdout_df["source_polarity"].tolist()
    official_acc = accuracy_score(gold, official_pred)
    dedicated_acc = accuracy_score(gold, dedicated_pred)
    print(f"Official zero-shot accuracy: {official_acc:.4f}")
    print(f"Dedicated classifier accuracy: {dedicated_acc:.4f}")

    for w in [0.1, 0.2, 0.3, 0.5]:
        ensemble_probs = (1 - w) * dedicated_probs + w * official_probs
        ensemble_pred = ["Positive" if p[0] >= p[1] else "Negative" for p in ensemble_probs]
        acc = accuracy_score(gold, ensemble_pred)
        print(f"Ensemble (dedicated weight={1-w:.1f}, official weight={w:.1f}): {acc:.4f}")

    print(f"\nBest single: dedicated alone={dedicated_acc:.4f}")


if __name__ == "__main__":
    main()
