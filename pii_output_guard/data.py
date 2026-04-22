import os
import numpy as np
import pandas as pd

from dotenv import load_dotenv
from datasets import load_dataset

from pii_output_guard.config import DATASET_NAME

load_dotenv()

TRAIN_SAMPLE_SIZE = int(os.getenv("PII_TRAIN_SAMPLE_SIZE", "20000"))
VALID_SAMPLE_SIZE = int(os.getenv("PII_VALID_SAMPLE_SIZE", "5000"))


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


def _balanced_sample_df(df, n: int):
    df = df.reset_index(drop=True)

    if n <= 0 or n >= len(df):
        return df

    class_0 = df[df["target"] == 0]
    class_1 = df[df["target"] == 1]

    if len(class_0) == 0 or len(class_1) == 0:
        return df.sample(n=min(n, len(df)), random_state=42).reset_index(drop=True)

    per_class = n // 2
    remainder = n % 2

    n0 = min(per_class, len(class_0))
    n1 = min(per_class + remainder, len(class_1))

    sampled_0 = class_0.sample(n=n0, random_state=42)
    sampled_1 = class_1.sample(n=n1, random_state=42)

    sampled = pd.concat([sampled_0, sampled_1], axis=0)
    sampled = sampled.sample(frac=1, random_state=42).reset_index(drop=True)

    return sampled


def load_splits():
    ds = load_dataset(
        DATASET_NAME,
        token=os.getenv("HF_TOKEN"),
    )

    train_df = ds["train"].to_pandas()
    valid_df = ds["validation"].to_pandas()

    for df in (train_df, valid_df):
        df["text"] = df["source_text"].astype(str)
        df["target"] = df.apply(to_binary_label, axis=1)

    train_df = _balanced_sample_df(train_df, TRAIN_SAMPLE_SIZE)
    valid_df = _balanced_sample_df(valid_df, VALID_SAMPLE_SIZE)

    print("Train target distribution:")
    print(train_df["target"].value_counts().to_dict())

    print("Validation target distribution:")
    print(valid_df["target"].value_counts().to_dict())

    return (
        train_df[["text", "target"]],
        valid_df[["text", "target"]],
    )


def build_label_map():
    return {
        0: "safe",
        1: "contains_pii",
    }