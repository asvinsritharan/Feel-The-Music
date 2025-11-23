import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np


def build_model(df):
    model = SentenceTransformer("sentence-transformers/all-roberta-large-v1")
    df["embedding"] = df["clean_lyrics"].apply(
        lambda x: model.encode(x, normalize_embeddings=True)
    )

def save_model_and_embeddings():
    np.save("../data/out/lyrics_embeddings_roberta.npy", np.vstack(df["embedding"].values))
    df.to_csv("../data/out/songs_roberta.csv", index=False)

if __name__ == "__main__":
    df = pd.read_csv("../data/in/cleaned_spotify_dataset.csv")
    build_model(df)
    save_model_and_embeddings()