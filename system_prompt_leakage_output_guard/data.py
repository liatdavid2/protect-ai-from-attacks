import os

import pandas as pd
from dotenv import load_dotenv
from datasets import load_dataset

from config import DATASET_NAME

load_dotenv()

TRAIN_SAMPLE_SIZE = int(os.getenv("SPL_TRAIN_SAMPLE_SIZE", "20000"))
EVAL_SAMPLE_SIZE = int(os.getenv("SPL_EVAL_SAMPLE_SIZE", "5000"))


def _balanced_sample_df(df: pd.DataFrame, n: int) -> pd.DataFrame:
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
    return sampled.sample(frac=1, random_state=42).reset_index(drop=True)


def load_splits():
    ds = load_dataset(DATASET_NAME, token=os.getenv("HF_TOKEN"))

    train_df = ds["train"].to_pandas()
    eval_split_name = "test" if "test" in ds else "validation"
    eval_df = ds[eval_split_name].to_pandas()

    for df in (train_df, eval_df):
        df["text"] = df["content"].astype(str)
        df["target"] = df["leakage"].astype(int)

    train_df = _balanced_sample_df(train_df, TRAIN_SAMPLE_SIZE)
    eval_df = _balanced_sample_df(eval_df, EVAL_SAMPLE_SIZE)

    print("Train target distribution:")
    print(train_df["target"].value_counts().to_dict())
    print(f"Eval split: {eval_split_name}")
    print("Eval target distribution:")
    print(eval_df["target"].value_counts().to_dict())

    return train_df[["text", "target"]], eval_df[["text", "target"]], eval_split_name


def build_label_map():
    return {
        0: "safe",
        1: "system_prompt_leakage",
    }
