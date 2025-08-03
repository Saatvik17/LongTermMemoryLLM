from sentence_transformers import SentenceTransformer

class Embedder:
    def __init__(self):
        try:
            self.model = SentenceTransformer("all-MiniLM-L6-v2")
        except Exception as e:
            print("Failed to load embeddings model:", e)
            self.model = None

    def embed(self, text: str):
        if not self.model:
            return [0.0] * 384
        return self.model.encode([text])[0]