import os
from typing import Dict

import pandas as pd
from datasets import load_dataset
from dotenv import load_dotenv

from prompt_injection_input_guard.config import DATASET_NAME, DATASET_CONFIG

load_dotenv()


def load_splits():
    ds = load_dataset(
        DATASET_NAME,
        DATASET_CONFIG,
        token=os.getenv("HF_TOKEN"),
    )
    train_df = ds["train"].to_pandas()
    valid_df = ds["validation"].to_pandas()
    test_df = ds["test"].to_pandas()
    return train_df, valid_df, test_df


def build_label_map(train_df: pd.DataFrame) -> Dict[int, str]:
    labels = sorted(train_df["label"].unique().tolist())
    default_names = {0: "benign", 1: "malicious"}
    return {int(label): default_names.get(int(label), f"class_{int(label)}") for label in labels}
