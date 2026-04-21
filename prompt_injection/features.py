import os

import numpy as np
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv()


class EmbeddingEncoder:
    def __init__(self, model_name: str):
        hf_token = os.getenv("HF_TOKEN") or None
        self.model = SentenceTransformer(model_name, token=hf_token)

    def encode(self, texts, batch_size: int = 64) -> np.ndarray:
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return embeddings.astype("float32")
