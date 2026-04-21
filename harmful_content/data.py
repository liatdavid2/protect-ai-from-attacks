import os

from dotenv import load_dotenv
from datasets import load_dataset


from config import DATASET_NAME


load_dotenv()


def to_binary_label(row) -> int:
    label = str(row["prompt_label"]).strip().lower()
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
        df["target"] = df.apply(to_binary_label, axis=1)

    return (
        train_df[["text", "target"]],
        valid_df[["text", "target"]],
        test_df[["text", "target"]],
    )


def build_label_map():
    return {
        0: "safe",
        1: "unsafe",
    }