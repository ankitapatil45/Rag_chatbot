from __future__ import annotations

import numpy as np
from functools import lru_cache
from typing import List, Union
from sentence_transformers import SentenceTransformer

DEFAULT_MODEL = "all-MiniLM-L6-v2"


class EmbeddingEngine:

    def __init__(self, model_name: str = DEFAULT_MODEL):
        self.model_name = model_name
        print(f"[EmbeddingEngine] Loading model: {model_name}")
        self._model = SentenceTransformer(model_name)
        print("[EmbeddingEngine] Model ready ✓")

    def encode(self, text: Union[str, List[str]]) -> np.ndarray:
        single  = isinstance(text, str)
        texts   = [text] if single else text
        vectors = self._model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return vectors[0] if single else vectors

    @property
    def dimension(self) -> int:
        return self._model.get_sentence_embedding_dimension()


@lru_cache(maxsize=1)
def get_engine(model_name: str = DEFAULT_MODEL) -> EmbeddingEngine:
    return EmbeddingEngine(model_name)
