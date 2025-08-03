from sentence_transformers import SentenceTransformer
import numpy as np

class Embedder:
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def embed(self, text: str) -> np.ndarray:
        return self.model.encode([text])[0]