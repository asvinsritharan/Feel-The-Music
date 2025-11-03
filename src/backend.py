from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd

df = pd.read_csv("songs_roberta.csv")
embeddings = np.load("lyrics_embeddings_roberta.npy")

def recommend_songs(df, user_query, top_k=10):
    model = SentenceTransformer("sentence-transformers/all-roberta-large-v1")
    query_vec = model.encode([user_query], normalize_embeddings=True)
    sims = cosine_similarity(embeddings, query_vec).flatten()
    top_idx = np.argsort(sims)[::-1][:top_k]
    return df.iloc[top_idx][["Artist(s)", "song", "clean_lyrics"]]


