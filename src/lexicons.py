"""
Emoji-flip lexicon mined from the training data (see ../EDA/eda_subtask2.py section 6),
used as a deterministic post-processing safety net: if a generated candidate still
contains a source-polarity emoji unchanged, force-flip it to its most common training-data
counterpart. This is a cheap, high-precision fix for the most literal failure mode of the
generator (leaving an obviously-wrong-polarity emoji untouched) and directly protects the
Sentiment Style Accuracy metric for emoji-heavy inputs (61%+ of sources contain emoji).

If the EDA has already been run, reuse its mined table (EDA/tables/mined_emoji_flip_lexicon.csv);
otherwise mine it fresh from the training file so this module works standalone.
"""
import os
import re
from collections import Counter, defaultdict

import pandas as pd

import config as cfg

EMOJI_RE = re.compile(
    "[\U0001F300-\U0001FAFF\U00002600-\U000027BF\U0001F1E6-\U0001F1FF]"
)

EDA_LEXICON_PATH = os.path.join(cfg.TASK_DIR, "EDA", "tables", "mined_emoji_flip_lexicon.csv")


def extract_emojis(s):
    return EMOJI_RE.findall(str(s))


def mine_emoji_flip_lexicon(train_df, min_count=5):
    flip_counter = Counter()
    for src, tgt in zip(train_df["source"], train_df["target"]):
        se, te = extract_emojis(src), extract_emojis(tgt)
        if len(se) == 0 or len(se) != len(te):
            continue
        for a, b in zip(se, te):
            if a != b:
                flip_counter[(a, b)] += 1

    # For each source emoji, keep only its single most common flip target.
    best = {}
    counts_per_source = defaultdict(list)
    for (a, b), c in flip_counter.items():
        counts_per_source[a].append((b, c))
    for a, options in counts_per_source.items():
        options.sort(key=lambda x: -x[1])
        b, c = options[0]
        if c >= min_count:
            best[a] = b
    return best


def load_emoji_flip_lexicon(train_df=None):
    if os.path.exists(EDA_LEXICON_PATH):
        df = pd.read_csv(EDA_LEXICON_PATH)
        best = {}
        seen = set()
        for _, row in df.sort_values("count", ascending=False).iterrows():
            src_e = row["source_emoji"]
            if src_e in seen:
                continue
            seen.add(src_e)
            best[src_e] = row["target_emoji"]
        return best
    if train_df is None:
        raise FileNotFoundError(
            f"{EDA_LEXICON_PATH} not found and no train_df given to mine it from scratch."
        )
    return mine_emoji_flip_lexicon(train_df)


def apply_emoji_safety_net(source: str, candidate: str, flip_lexicon: dict) -> str:
    """If `candidate` still contains a source emoji that the mined lexicon says should
    have flipped, and the flipped counterpart is not already present in candidate, replace
    the first unflipped occurrence. Conservative: only fires when there is unambiguous
    lexicon evidence, so it cannot hurt already-correct generations."""
    result = candidate
    for src_emoji in extract_emojis(source):
        if src_emoji not in flip_lexicon:
            continue
        target_emoji = flip_lexicon[src_emoji]
        if src_emoji in result and target_emoji not in result:
            result = result.replace(src_emoji, target_emoji, 1)
    return result


# ----------------------------------------------------------------------------------
# Word-level antonym safety net -- identical mechanism to the emoji one above, but using
# the short-span lexical substitution pairs mined in EDA section 7
# (mined_word_flip_lexicon.csv, e.g. سيء<->جميل, رائع<->سيء): if the generator left an
# obviously source-polarity word untouched, force-substitute it with its mined
# counterpart. Same conservative design -- only fires on unambiguous lexicon hits, and
# only when the counterpart isn't already present (so it can't double-flip a candidate
# that already handled the substitution correctly, e.g. via a synonym).
# ----------------------------------------------------------------------------------
EDA_WORD_LEXICON_PATH = os.path.join(cfg.TASK_DIR, "EDA", "tables", "mined_word_flip_lexicon.csv")
_WORD_RE = re.compile(r"[؀-ۿ]+|\w+")


def load_word_flip_lexicon(min_count=20):
    """Only single-word (not multi-word phrase) entries are used, and only the most
    frequent target per source word, kept conservative with a higher min_count than the
    emoji lexicon since word substitution is riskier (more ways to be wrong / to clash
    with a word that's part of a longer, already-correct phrase)."""
    if not os.path.exists(EDA_WORD_LEXICON_PATH):
        return {}
    df = pd.read_csv(EDA_WORD_LEXICON_PATH)
    df = df[df["count"] >= min_count]
    df = df[df["source_phrase"].astype(str).str.split().str.len() == 1]
    df = df[df["target_phrase"].astype(str).str.split().str.len() == 1]
    best = {}
    seen = set()
    for _, row in df.sort_values("count", ascending=False).iterrows():
        src_w = row["source_phrase"]
        if src_w in seen:
            continue
        seen.add(src_w)
        best[src_w] = row["target_phrase"]
    return best


def apply_word_safety_net(source: str, candidate: str, word_lexicon: dict) -> str:
    src_tokens = set(_WORD_RE.findall(str(source)))
    cand_tokens = _WORD_RE.findall(str(candidate))
    cand_token_set = set(cand_tokens)
    result = candidate
    for src_word in src_tokens:
        if src_word not in word_lexicon:
            continue
        target_word = word_lexicon[src_word]
        if src_word in cand_token_set and target_word not in cand_token_set:
            # whole-word replace (word boundary via regex) of the first occurrence only
            result = re.sub(rf"(?<!\w){re.escape(src_word)}(?!\w)", target_word, result, count=1)
    return result
