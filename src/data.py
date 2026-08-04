"""Data loading utilities for Subtask 2."""
import pandas as pd
import torch
from torch.utils.data import Dataset

import config as cfg


def load_train():
    return pd.read_excel(cfg.TRAIN_PATH).reset_index(drop=True)


def load_val():
    return pd.read_excel(cfg.VAL_PATH).reset_index(drop=True)


def load_test():
    return pd.read_excel(cfg.TEST_PATH).reset_index(drop=True)


def prefix_for(polarity: str) -> str:
    """polarity is the SOURCE polarity ('Positive'/'Negative'); the prefix instructs the
    model which direction to swap."""
    return cfg.POS2NEG_PREFIX if str(polarity).strip().lower().startswith("pos") else cfg.NEG2POS_PREFIX


def target_polarity_for(source_polarity: str) -> str:
    return "negative" if str(source_polarity).strip().lower().startswith("pos") else "positive"


class SwapSeq2SeqDataset(Dataset):
    """(prefix + source) -> target, tokenized for a T5-style encoder-decoder."""

    def __init__(self, sources, polarities, targets, tokenizer,
                 max_src_len=cfg.GENERATOR_MAX_SRC_LEN, max_tgt_len=cfg.GENERATOR_MAX_TGT_LEN):
        self.sources = list(sources)
        self.polarities = list(polarities)
        self.targets = list(targets) if targets is not None else None
        self.tokenizer = tokenizer
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len

    def __len__(self):
        return len(self.sources)

    def __getitem__(self, idx):
        inp = prefix_for(self.polarities[idx]) + str(self.sources[idx])
        enc = self.tokenizer(inp, truncation=True, max_length=self.max_src_len, padding="max_length")
        item = {k: torch.tensor(v) for k, v in enc.items()}
        if self.targets is not None:
            tgt_enc = self.tokenizer(
                text_target=str(self.targets[idx]), truncation=True, max_length=self.max_tgt_len, padding="max_length"
            )
            labels = torch.tensor(tgt_enc["input_ids"])
            labels[labels == self.tokenizer.pad_token_id] = -100
            item["labels"] = labels
        return item
