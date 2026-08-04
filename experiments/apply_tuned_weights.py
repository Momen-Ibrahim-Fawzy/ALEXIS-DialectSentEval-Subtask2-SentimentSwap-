"""
Reads outputs/reranking_grid_search.json (written by tune_reranking.py) and rewrites
config.py's POLARITY_SCORE_WEIGHT / CONTENT_SCORE_WEIGHT constants in place with the
best-scoring combination, so predict.py picks it up automatically on its next run.

Selection criterion: Sentiment Style Accuracy is listed first/primary in the task page's
metric description ("ranking will be based on... Sentiment Style Accuracy... Content
Preservation: BLEU and chrF"), so among grid points within 1.5 points of the best chrF
(i.e. not sacrificing meaningful content-preservation), pick the one with the highest
Style Accuracy. This avoids the degenerate pw=1.0/cw=0.0 corner (pure polarity-chasing
with no content-preservation pressure) unless it's genuinely far ahead.

Usage:
  conda run -n mo python3 apply_tuned_weights.py
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import json
import os
import re

import config as cfg

GRID_PATH = os.path.join(cfg.OUTPUT_DIR, "reranking_grid_search.json")
CONFIG_PATH = os.path.join(cfg.SYSTEM_DIR, "config.py")


def main():
    if not os.path.exists(GRID_PATH):
        print(f"No {GRID_PATH} found; run tune_reranking.py first. Leaving config.py unchanged.")
        return

    with open(GRID_PATH, encoding="utf-8") as f:
        results = json.load(f)

    best_chrf = max(r["chrf"] for r in results)
    candidates = [r for r in results if r["chrf"] >= best_chrf - 1.5]
    best = max(candidates, key=lambda r: r["sentiment_style_accuracy_pct"])

    print(f"Grid search had {len(results)} points. Best chrF={best_chrf:.2f}. Among points within 1.5 "
          f"chrF of that, best Sentiment Style Accuracy: pw={best['polarity_weight']} cw={best['content_weight']} "
          f"-> BLEU={best['bleu']:.2f} chrF={best['chrf']:.2f} StyleAcc={best['sentiment_style_accuracy_pct']:.2f}%")

    if best["polarity_weight"] == cfg.POLARITY_SCORE_WEIGHT and best["content_weight"] == cfg.CONTENT_SCORE_WEIGHT:
        print("Already using these weights in config.py -- no change needed.")
        return

    with open(CONFIG_PATH, encoding="utf-8") as f:
        text = f.read()
    text = re.sub(r"POLARITY_SCORE_WEIGHT = [\d.]+", f"POLARITY_SCORE_WEIGHT = {best['polarity_weight']}", text)
    text = re.sub(r"CONTENT_SCORE_WEIGHT = [\d.]+", f"CONTENT_SCORE_WEIGHT = {best['content_weight']}", text)
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"Updated {CONFIG_PATH}: POLARITY_SCORE_WEIGHT={best['polarity_weight']}, "
          f"CONTENT_SCORE_WEIGHT={best['content_weight']}")


if __name__ == "__main__":
    main()
