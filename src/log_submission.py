"""
Submission tracker for Subtask 2.

Every time you upload outputs/predictions.zip to Codabench, snapshot it:

  conda run -n mo python3 log_submission.py --tag v1_mt5_retrieval --note "mt5-base + retrieval + rerank + emoji safety net"

This copies the current predictions.xlsx/.zip plus the full config + local dev-set
metrics (BLEU/chrF/approx. Sentiment Style Accuracy) into submissions/<NNN>_<tag>/, and
adds a row to SUBMISSIONS_LOG.md. Once Codabench reports the official score, record it:

  conda run -n mo python3 log_submission.py --record v1_mt5_retrieval \
      --sentiment_style_accuracy 0.61 --bleu 28.4 --chrf 55.2

This keeps a permanent, reproducible link between "exact system config" <-> "exact
predictions file" <-> "official leaderboard score" for writing up the system-description
paper later.
"""
import argparse
import glob
import json
import os
import re
import shutil
from datetime import datetime, timezone

import config as cfg

SUB_DIR = os.path.join(cfg.SYSTEM_DIR, "submissions")
LOG_PATH = os.path.join(cfg.SYSTEM_DIR, "SUBMISSIONS_LOG.md")
os.makedirs(SUB_DIR, exist_ok=True)


def next_index():
    existing = [d for d in os.listdir(SUB_DIR) if os.path.isdir(os.path.join(SUB_DIR, d))]
    nums = [int(d.split("_")[0]) for d in existing if d[:3].isdigit()]
    return (max(nums) + 1) if nums else 1


def config_snapshot():
    return {
        "generator_model": cfg.GENERATOR_MODEL,
        "eval_classifier_model": cfg.EVAL_CLASSIFIER_MODEL,
        "generator_max_src_len": cfg.GENERATOR_MAX_SRC_LEN,
        "generator_max_tgt_len": cfg.GENERATOR_MAX_TGT_LEN,
        "seed": cfg.SEED,
        "batch_size": cfg.BATCH_SIZE,
        "grad_accum_steps": cfg.GRAD_ACCUM_STEPS,
        "effective_batch_size": cfg.BATCH_SIZE * cfg.GRAD_ACCUM_STEPS,
        "learning_rate": cfg.LEARNING_RATE,
        "num_epochs": cfg.NUM_EPOCHS,
        "weight_decay": cfg.WEIGHT_DECAY,
        "warmup_ratio": cfg.WARMUP_RATIO,
        "num_candidates": cfg.NUM_CANDIDATES,
        "polarity_score_weight": cfg.POLARITY_SCORE_WEIGHT,
        "content_score_weight": cfg.CONTENT_SCORE_WEIGHT,
    }


def load_json_if_exists(path):
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return None


def parse_predict_log(log_text):
    """Extract the ACTUAL active pipeline configuration from a predict.py run's stdout --
    which candidate pool sources loaded, exact-match coverage, beam allocation. This is
    what makes Section 2 of the report true for a SPECIFIC submission instead of a
    generic template that goes stale the moment any pool component changes."""
    if not log_text:
        return None
    info = {"pool_components": [], "exact_match": None, "beam_allocation": None, "source_polarity_detector": None}
    for line in log_text.splitlines():
        line = line.strip()
        if line.startswith("Loaded "):
            info["pool_components"].append(line)
        m = re.search(r"Exact-match retrieval covered (\d+)/(\d+)", line)
        if m:
            info["exact_match"] = f"{m.group(1)}/{m.group(2)} test rows ({int(m.group(1))/int(m.group(2)):.1%})"
        m = re.search(r"Generating (\d+) candidates for (\d+) (\S+) rows.*and (\d+) for (\d+) (\S+) rows", line)
        if m:
            info["beam_allocation"] = (f"{m.group(1)} beams for {m.group(2)} {m.group(3)} rows, "
                                        f"{m.group(4)} beams for {m.group(5)} {m.group(6)} rows")
        elif re.search(r"Generating (\d+) candidates each for (\d+) rows", line):
            m2 = re.search(r"Generating (\d+) candidates each for (\d+) rows", line)
            info["beam_allocation"] = f"uniform {m2.group(1)} beams for all {m2.group(2)} generated rows"
        if "dedicated source-polarity classifier" in line.lower():
            info["source_polarity_detector"] = "dedicated (fine-tuned on 7716 labeled train+val pairs)"
    if info["source_polarity_detector"] is None and info["pool_components"]:
        info["source_polarity_detector"] = "official CAMeL-Lab classifier, zero-shot"
    return info if info["pool_components"] or info["beam_allocation"] else None


def find_predict_log_for_submission(entry_dir_name, timestamp_iso):
    """Best-effort match of a submission to the predict.py log that produced it, by
    output_subdir naming convention. Returns raw log text or None if no confident match."""
    tag_part = entry_dir_name.split("_", 1)[1] if "_" in entry_dir_name else entry_dir_name
    candidates = glob.glob(os.path.join(cfg.OUTPUT_DIR, "predict_*.log"))
    # Prefer a log whose filename shares the same version token (e.g. "v17") as the tag
    m = re.match(r"(v\d+)_", tag_part)
    version_token = m.group(1) if m else None
    if version_token:
        matches = [p for p in candidates if re.search(rf"predict_{version_token}(_|\.)", os.path.basename(p))]
        matches = [p for p in matches if "retry" not in os.path.basename(p)] or matches
        if matches:
            with open(sorted(matches)[-1], encoding="utf-8", errors="replace") as f:
                return f.read()
    return None


def append_log_row(entry_dir_name, tag, note, dev_summary):
    header = "| Dir | Tag | Date (UTC) | Note | Dev BLEU / chrF / StyleAcc% | Official Sentiment Style Acc | Official BLEU | Official chrF |\n"
    sep = "|---|---|---|---|---|---|---|---|\n"
    if not os.path.exists(LOG_PATH):
        with open(LOG_PATH, "w", encoding="utf-8") as f:
            f.write("# Subtask 2 — Submission Log\n\n")
            f.write(header)
            f.write(sep)
    row = (
        f"| `{entry_dir_name}` | {tag} | {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M')} UTC "
        f"| {note} | {dev_summary} | pending | pending | pending |\n"
    )
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(row)


def update_log_row(tag, metrics):
    if not os.path.exists(LOG_PATH):
        print("No SUBMISSIONS_LOG.md yet -- nothing to update.")
        return
    with open(LOG_PATH, encoding="utf-8") as f:
        lines = f.readlines()
    updated = False
    for i, line in enumerate(lines):
        if line.startswith("|") and f"| {tag} |" in line:
            parts = line.split("|")
            # columns: ['', Dir, Tag, Date, Note, DevSummary, StyleAcc, BLEU, chrF, '\n']
            parts[6] = f" {metrics.get('sentiment_style_accuracy', 'pending')} "
            parts[7] = f" {metrics.get('bleu', 'pending')} "
            parts[8] = f" {metrics.get('chrf', 'pending')} "
            lines[i] = "|".join(parts)
            updated = True
            break
    if updated:
        with open(LOG_PATH, "w", encoding="utf-8") as f:
            f.writelines(lines)
        print(f"Updated SUBMISSIONS_LOG.md row for tag '{tag}'.")
    else:
        print(f"No row found for tag '{tag}' in SUBMISSIONS_LOG.md.")


def write_report_md(entry_dir, data):
    """Self-contained markdown report: what produced this predictions file, the exact
    system/training configuration, local validation numbers, and the official Codabench
    result (once recorded) -- written for direct reuse in a system-description paper.

    Section 2 is built from the ACTUAL predict.py log for this submission when available
    (parsed pool components + beam allocation actually used), not a fixed template -- an
    earlier version of this function hardcoded a "mt5-base + official classifier"
    description for every submission regardless of what was actually pooled/detected,
    which went silently wrong the moment DPO, mt5-large, NLLB, the tagger, retrieval
    augmentation, direction-aware beams, or the dedicated source-polarity classifier were
    added. The Note field (always submission-specific and written at snapshot time) is
    treated as the authoritative source when no log can be matched, rather than papering
    over the gap with a generic description that might be flatly wrong for that run."""
    cfg_snap = data["config"]
    dev = data.get("dev_generation_metrics") or {}
    official = data.get("official_score")
    pipeline = data.get("pipeline_from_log")

    lines = []
    lines.append(f"# Subtask 2 Submission Report — `{data['tag']}`")
    lines.append("")
    lines.append(f"**Date (UTC):** {data['timestamp_utc']}  ")
    lines.append(f"**Note (authoritative, submission-specific description of what's different about this run):** "
                  f"{data['note'] or '(none)'}")
    lines.append("")

    lines.append("## 1. Task")
    lines.append("")
    lines.append("DialectSentEval 2026 Subtask 2 — Arabic Sentiment Swap. Rewrite an Arabic sentence to invert "
                  "its sentiment polarity while preserving its topic/meaning. Dataset: MA'AKS Extended, 9,647 "
                  "social-media (source, target, source_polarity) pairs — 6,752 train / 964 val / 1,931 test. "
                  "The released test file has **no `source_polarity` column**, unlike train/val. Official "
                  "metrics: Sentiment Style Accuracy (via `CAMeL-Lab/bert-base-arabic-camelbert-da-sentiment`) "
                  "plus BLEU/chrF for content preservation against the source.")
    lines.append("")

    lines.append("## 2. System description (this specific submission)")
    lines.append("")
    lines.append("**Base architecture (unchanging across all submissions):** seq2seq generator(s), fully "
                  "fine-tuned end-to-end on the in-domain MA'AKS pairs, conditioned on a source-polarity-specific "
                  "instruction prefix (`حول من ايجابي الى سلبي:` / `حول من سلبي الى ايجابي:`); a pool of one or "
                  "more candidates per test row is generated and reranked by classifier-confidence + chrF-to-"
                  "source. **What varies between submissions is which generator checkpoint, which additional "
                  "candidate sources are pooled, the source-polarity detector, and beam allocation -- see below "
                  "for what was ACTUALLY active in this specific run, and the Note above for why.**")
    lines.append("")
    if pipeline:
        lines.append("**Actual pipeline configuration for this submission** (parsed from its `predict.py` run log):")
        lines.append("")
        if pipeline.get("pool_components"):
            for comp in pipeline["pool_components"]:
                lines.append(f"- {comp}")
        if pipeline.get("exact_match"):
            lines.append(f"- Exact-match retrieval coverage: {pipeline['exact_match']}")
        if pipeline.get("source_polarity_detector"):
            lines.append(f"- Source-polarity detection (Stage 2): {pipeline['source_polarity_detector']}")
        if pipeline.get("beam_allocation"):
            lines.append(f"- Beam allocation: {pipeline['beam_allocation']}")
        lines.append("")
        lines.append(f"Reranking score: `{cfg_snap['polarity_score_weight']} * P(classifier predicts target "
                      f"polarity) + {cfg_snap['content_score_weight']} * chrF(candidate, source)`; the official "
                      f"grading classifier ([`{cfg_snap['eval_classifier_model']}`]"
                      f"(https://huggingface.co/{cfg_snap['eval_classifier_model']})) is always used for this "
                      f"scoring step regardless of what detects source polarity, since it must match the actual "
                      f"grading criterion.")
    else:
        lines.append("**No matching `predict.py` log was found for this submission** (this is expected for "
                      "earlier submissions before per-run logs were consistently kept) -- **the Note above is "
                      "the authoritative description of this submission's exact configuration.** Do not assume "
                      "the generic architecture description above applies in full; several submissions in this "
                      "project's history use a DPO-tuned generator, an additional pooled backbone (mt5-large "
                      "and/or NLLB), the edit-tagger, near-duplicate retrieval, or a dedicated source-polarity "
                      "classifier instead of the plain MLE mt5-base + official-classifier baseline.")
    lines.append("")
    lines.append("**Emoji + word-level safety nets:** force-flip any left-over source-polarity emoji/word using "
                  "lexicons mined from the training data, if the reranked candidate didn't already flip it -- "
                  "present in every submission from `v2` onward (see Note for whether this specific one predates "
                  "or postdates that).")
    lines.append("")

    lines.append("## 3. Training configuration")
    lines.append("")
    lines.append("These are the base MLE fine-tuning hyperparameters (`train_generator.py`, `config.py`). "
                  "**If this submission's generator was DPO-tuned, a different-sized backbone (mt5-large), or a "
                  "different auxiliary model (NLLB, the edit-tagger, the dedicated source-polarity classifier), "
                  "its own training used different hyperparameters (see the corresponding training script and "
                  "the Note above) -- this table does not describe those.**")
    lines.append("")
    lines.append("| Hyperparameter | Value |")
    lines.append("|---|---|")
    lines.append(f"| Max source / target length | {cfg_snap['generator_max_src_len']} / "
                  f"{cfg_snap['generator_max_tgt_len']} sub-word tokens |")
    lines.append(f"| Per-device batch size | {cfg_snap['batch_size']} |")
    lines.append(f"| Gradient accumulation steps | {cfg_snap['grad_accum_steps']} |")
    lines.append(f"| Effective batch size | {cfg_snap['effective_batch_size']} |")
    lines.append(f"| Learning rate | {cfg_snap['learning_rate']} |")
    lines.append(f"| Epochs | {cfg_snap['num_epochs']} |")
    lines.append(f"| Weight decay | {cfg_snap['weight_decay']} |")
    lines.append(f"| Warmup ratio | {cfg_snap['warmup_ratio']} |")
    lines.append(f"| Precision | bf16 (T5-family models are known to produce NaN losses under fp16; "
                  f"verified empirically on this exact checkpoint during development) |")
    lines.append(f"| Memory optimizations | gradient checkpointing enabled (shared, memory-constrained GPU) |")
    lines.append(f"| Final-model training data | train + val combined, 7,716 pairs "
                  f"(`train_generator.py --mode final`); no held-out split for the deployed model |")
    lines.append("")

    lines.append("## 4. Local validation")
    lines.append("")
    lines.append("**This submission's specific validation methodology and numbers (held-out check, dev-set "
                  "metrics, or ablation) are described in the Note above** -- this project used different "
                  "validation approaches for different techniques (dev-set generation metrics for early MLE "
                  "generator choices; held-out classifier accuracy for the source-polarity fix; val-set grid "
                  "search for reranking weights, later found unreliable and abandoned -- see SUBMISSIONS_LOG.md "
                  "for that history) rather than one fixed protocol, so a single table here would misrepresent "
                  "most submissions.")
    lines.append("")
    if dev:
        lines.append("For reference, the most recent `outputs/dev_generation_metrics.json` at snapshot time "
                      "(NOT necessarily this submission's own generator -- this file is overwritten by whichever "
                      "`train_generator.py --mode dev` run happened most recently, which may predate this "
                      "submission) showed: "
                      f"BLEU={dev.get('bleu'):.2f}, chrF={dev.get('chrf'):.2f}, "
                      f"StyleAcc={dev.get('sentiment_style_accuracy_pct')}%.")
        lines.append("")

    lines.append("## 5. Official Codabench result")
    lines.append("")
    if official:
        lines.append("| Metric | Value |")
        lines.append("|---|---|")
        for k, label in (("sentiment_style_accuracy", "Sentiment Style Accuracy / Preservation"),
                          ("bleu", "BLEU"), ("chrf", "chrF")):
            if k in official:
                lines.append(f"| {label} | {official[k]} |")
    else:
        lines.append("*Pending — not yet recorded. Run:*")
        lines.append(f"```\nconda run -n mo python3 log_submission.py --record {data['tag']} "
                      f"--sentiment_style_accuracy ... --bleu ... --chrf ...\n```")
    lines.append("")

    lines.append("## 6. Files in this directory")
    lines.append("")
    lines.append("- `predictions.xlsx` / `predictions.zip` — the exact submission uploaded to Codabench")
    lines.append("- `predictions_diagnostic.csv` — per-row source/output/method (`retrieval` vs `generated`), "
                  "not part of the submission, useful for error analysis")
    lines.append("- `system_snapshot.json` — machine-readable version of everything in this report")
    lines.append("- `REPORT.md` — this file")
    lines.append("")

    with open(os.path.join(entry_dir, "REPORT.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def snapshot(tag, note, source_dir=None):
    source_dir = source_dir or cfg.OUTPUT_DIR
    idx = next_index()
    entry_dir_name = f"{idx:03d}_{tag}"
    entry_dir = os.path.join(SUB_DIR, entry_dir_name)
    os.makedirs(entry_dir, exist_ok=True)

    for fname in ("predictions.xlsx", "predictions.zip", "predictions_diagnostic.csv"):
        src = os.path.join(source_dir, fname)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(entry_dir, fname))

    dev_metrics = load_json_if_exists(os.path.join(cfg.OUTPUT_DIR, "dev_generation_metrics.json"))
    dev_summary = "n/a"
    if dev_metrics:
        dev_summary = (f"BLEU={dev_metrics.get('bleu'):.2f} / chrF={dev_metrics.get('chrf'):.2f} / "
                        f"StyleAcc={dev_metrics.get('sentiment_style_accuracy_pct')}")

    pipeline_from_log = parse_predict_log(find_predict_log_for_submission(entry_dir_name, None))

    snapshot_data = {
        "tag": tag,
        "note": note,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "config": config_snapshot(),
        "dev_generation_metrics": dev_metrics,
        "pipeline_from_log": pipeline_from_log,
        "official_score": None,  # filled in later via --record
    }
    with open(os.path.join(entry_dir, "system_snapshot.json"), "w", encoding="utf-8") as f:
        json.dump(snapshot_data, f, ensure_ascii=False, indent=2)
    write_report_md(entry_dir, snapshot_data)

    append_log_row(entry_dir_name, tag, note, dev_summary)
    print(f"Snapshotted submission to {entry_dir} (predictions + system_snapshot.json + REPORT.md)")
    print(f"Appended row to {LOG_PATH} -- fill in the official score with:")
    print(f"  conda run -n mo python3 log_submission.py --record {tag} --sentiment_style_accuracy ... --bleu ... --chrf ...")


def record_result(tag, sentiment_style_accuracy, bleu, chrf):
    matches = [d for d in os.listdir(SUB_DIR) if d.endswith(f"_{tag}")]
    if not matches:
        raise FileNotFoundError(f"No submission snapshot found for tag '{tag}' under {SUB_DIR}")
    entry_dir = os.path.join(SUB_DIR, matches[0])
    snap_path = os.path.join(entry_dir, "system_snapshot.json")
    with open(snap_path, encoding="utf-8") as f:
        data = json.load(f)
    if "pipeline_from_log" not in data:
        data["pipeline_from_log"] = parse_predict_log(find_predict_log_for_submission(matches[0], data.get("timestamp_utc")))
    metrics = {"sentiment_style_accuracy": sentiment_style_accuracy, "bleu": bleu, "chrf": chrf}
    data["official_score"] = {k: v for k, v in metrics.items() if v is not None}
    with open(snap_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    write_report_md(entry_dir, data)
    update_log_row(tag, data["official_score"])
    print(f"Recorded official Codabench result for '{tag}' in {snap_path} (and updated REPORT.md)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", help="short identifier for this submission, e.g. v1_mt5_retrieval")
    parser.add_argument("--note", default="", help="free-text note on what's different about this run")
    parser.add_argument("--record", help="tag of an existing submission to attach an official score to")
    parser.add_argument("--source_dir", default=None, help="directory containing predictions.xlsx/.zip to snapshot (default: outputs/)")
    parser.add_argument("--sentiment_style_accuracy", type=float, default=None)
    parser.add_argument("--bleu", type=float, default=None)
    parser.add_argument("--chrf", type=float, default=None)
    args = parser.parse_args()

    if args.record:
        record_result(args.record, args.sentiment_style_accuracy, args.bleu, args.chrf)
    elif args.tag:
        snapshot(args.tag, args.note, source_dir=args.source_dir)
    else:
        parser.error("Provide either --tag (to snapshot a new submission) or --record (to attach an official score)")
