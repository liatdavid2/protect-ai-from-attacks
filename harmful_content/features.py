import os

from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer


load_dotenv()


class EmbeddingEncoder:
    def __init__(self, model_name: str):
        hf_token = os.getenv("HF_TOKEN") or None
        self.model = SentenceTransformer(model_name, token=hf_token)

    def encode(self, texts):
        return self.model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=True,
        )