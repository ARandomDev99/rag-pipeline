from collections.abc import Sequence

import numpy as np
from sentence_transformers import SentenceTransformer


class Embedder:
    def __init__(self, model_name: str) -> None:
        self.model = SentenceTransformer(model_name, local_files_only=True)
        self.dim = int(self.model.get_embedding_dimension())

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        vecs = self.model.encode(
            list(texts),
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return vecs.astype(np.float32)