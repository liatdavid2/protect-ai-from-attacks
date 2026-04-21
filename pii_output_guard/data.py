import os
import numpy as np
import pandas as pd

from dotenv import load_dotenv
from datasets import load_dataset

from config import DATASET_NAME

load_dotenv()

TRAIN_SAMPLE_SIZE = int(os.getenv("PII_TRAIN_SAMPLE_SIZE", "5000"))
VALID_SAMPLE_SIZE = int(os.getenv("PII_VALID_SAMPLE_SIZE", "1000"))

def _sample_df(df, n: int):
    if n <= 0 or n >= len(df):
        return df.reset_index(drop=True)
    return df.sample(n=n, random_state=42).reset_index(drop=True)


def _normalize_mask_value(value) -> bool:
    if value is None:
        return False
    if isinstance(value, float) and pd.isna(value):
        return False
    if isinstance(value, np.ndarray):
        return value.size > 0
    if isinstance(value, (list, tuple, set, dict)):
        return len(value) > 0
    if isinstance(value, str):
        cleaned = value.strip().lower()
        if cleaned in {"", "[]", "{}", "none", "null", "nan"}:
            return False
        return True
    return bool(value)

def to_binary_label(row) -> int:
    return 1 if _normalize_mask_value(row["privacy_mask"]) else 0

def load_splits():
    ds = load_dataset(
        DATASET_NAME,
        token=os.getenv("HF_TOKEN"),
    )

    train_df = ds["train"].to_pandas()
    valid_df = ds["validation"].to_pandas()

    train_df = _sample_df(train_df, TRAIN_SAMPLE_SIZE)
    valid_df = _sample_df(valid_df, VALID_SAMPLE_SIZE)

    for df in (train_df, valid_df):
        df["text"] = df["source_text"].astype(str)
        df["target"] = df.apply(to_binary_label, axis=1)

    return (
        train_df[["text", "target"]],
        valid_df[["text", "target"]],
    )

def build_label_map():
    return {
        0: "safe",
        1: "contains_pii",
    }
