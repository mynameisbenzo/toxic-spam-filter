from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

MODEL = "sentence-transformers/all-MiniLM-L6-v2"
OUT = Path("data/embeddings")


def main():
    model = SentenceTransformer(MODEL, device="cpu")
    OUT.mkdir(parents=True, exist_ok=True)

    for name in ["train", "val", "test"]:
        path = OUT / f"{name}.npy"
        if path.exists():
            print(f"{name}: already embedded, skipping")
            continue
        df = pd.read_csv(f"data/processed/{name}.csv").fillna({"text": ""})
        print(f"{name}: embedding {len(df):,} comments")
        emb = model.encode(df["text"].tolist(), batch_size=64, show_progress_bar=True,
                           normalize_embeddings=True, convert_to_numpy=True)
        np.save(path, emb.astype(np.float32))
        print(f"{name}: saved {emb.shape}")


if __name__ == "__main__":
    main()