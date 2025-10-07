from typing import List, Iterable
import numpy as np
from config import Config

# Local SBERT
from sentence_transformers import SentenceTransformer
# Optional OpenAI
import openai

EMBED_DIM_LOCAL = 384

class Embedder:
    def __init__(self):
        self.mode = Config.EMBEDDER.lower()
        if self.mode == "local":
            # choose a compact embedding model
            self.model = SentenceTransformer("all-MiniLM-L6-v2")
            self.dim = self.model.get_sentence_embedding_dimension()
        elif self.mode == "openai":
            if not Config.OPENAI_API_KEY:
                raise ValueError("OPENAI_API_KEY is required for openai mode")
            openai.api_key = Config.OPENAI_API_KEY
            self.dim = None
        else:
            raise ValueError("Unknown EMBEDDER mode")

    def embed(self, texts: Iterable[str]) -> List[List[float]]:
        texts = list(texts)
        if self.mode == "local":
            arr = self.model.encode(texts, normalize_embeddings=True)
            return arr.tolist()
        else:
            # OpenAI batching
            out = []
            B = 20
            for i in range(0, len(texts), B):
                batch = texts[i:i+B]
                resp = openai.Embedding.create(model="text-embedding-3-large", input=batch)
                out.extend([d["embedding"] for d in resp["data"]])
            # normalize optionally
            return out
