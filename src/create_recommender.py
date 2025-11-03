import pandas as pd
import re
from sentence_transformers import SentenceTransformer
import numpy as np

def clean_lyrics(text):
    text = str(text)
    text = re.sub(r'\[.*?\]', '', text)  # remove [Chorus], [Verse]
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text.strip().lower()

def build_model(df):
    model = SentenceTransformer("sentence-transformers/all-roberta-large-v1")
    df["embedding"] = df["clean_lyrics"].apply(
        lambda x: model.encode(x, normalize_embeddings=True)
    )

def save_model_and_embeddings():
    np.save("../data/out/lyrics_embeddings_roberta.npy", np.vstack(df["embedding"].values))
    df.to_csv("../data/out/songs_roberta.csv", index=False)

if __name__ == "__main__":
    df = pd.read_csv("../data/in/spotify_dataset.csv")
    build_model(df)
    save_model_and_embeddings()