import os

from datasets import load_dataset
from dotenv import load_dotenv

from harmful_content.config import DATASET_NAME

load_dotenv()


def to_binary_label(value) -> int:
    label = str(value).strip().lower()
    return 0 if label == "safe" else 1


def load_splits():
    ds = load_dataset(
        DATASET_NAME,
        token=os.getenv("HF_TOKEN"),
    )
    train_df = ds["train"].to_pandas()
    valid_df = ds["validation"].to_pandas()
    test_df = ds["test"].to_pandas()

    for df in (train_df, valid_df, test_df):
        df["text"] = df["prompt"].astype(str)
        df["label"] = df["prompt_label"].apply(to_binary_label)

    return train_df[["text", "label"]], valid_df[["text", "label"]], test_df[["text", "label"]]


def build_label_map():
    return {0: "safe", 1: "unsafe"}
