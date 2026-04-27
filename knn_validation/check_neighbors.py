"""
Run this to check neighbor values from genres and tags
"""

import json
import os
import numpy as np

EMBEDDING_DIR = "../data/embeddings/"
GENRE_EMBEDDING = os.path.join(EMBEDDING_DIR, "genre_embeddings.json")
TAG_EMBEDDING = os.path.join(EMBEDDING_DIR, "tag_embeddings.json")


class LoadedEmbedding:
    def __init__(self, raw: dict[str, list[float]]):
        self.vocab: list[str] = list(raw.keys())
        self.token2idx: dict = {g: i for i, g in enumerate(self.vocab)}
        self.Z = np.array(list(raw.values()), dtype=np.float32)
        self.Z_norm = self.Z / (np.linalg.norm(self.Z, axis=1, keepdims=True) + 1e-9)

    def neighbors(self, q: str, k: int = 5):
        if q not in self.token2idx:
            print(f"{q} not in vocab")
            return
        idx = self.token2idx[q]
        sims = self.Z_norm @ self.Z_norm[idx]
        top = np.argsort(sims)[::-1][1:k+1]
        print(f"\n-> {q}")
        for i in top:
            print(f"  {self.vocab[i]:<20} {sims[i]:.3f}")


def load_embedding(path: str) -> LoadedEmbedding:
    with open(path) as f:
        raw = json.load(f)
    return LoadedEmbedding(raw)


if __name__ == "__main__":
    try:
        assert (os.path.exists(GENRE_EMBEDDING) and os.path.exists(TAG_EMBEDDING))

        GENRES = ["pop rap", "death metal", "jazz", "classical", "edm", "folk", "shoegaze", "blues"]
        TAGS = ["melancholic", "aggressive", "chill", "dark", "happy", "romantic", "nostalgic", "sad", "upbeat", "heavy"]
        genre_embedding = load_embedding(GENRE_EMBEDDING)
        tag_embedding = load_embedding(TAG_EMBEDDING)

        for genre in GENRES:
            genre_embedding.neighbors(genre)

        for tag in TAGS:
            tag_embedding.neighbors(tag)
    except AssertionError as e:
        print(f"AssertionError: {e}")
