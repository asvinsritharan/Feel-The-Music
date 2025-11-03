from backend import *

# app.py
import streamlit as st
from sentence_transformers import SentenceTransformer, util
import pandas as pd
import numpy as np

# Load data + model
model = SentenceTransformer('sentence-transformers/all-roberta-large-v1')
df = pd.read_csv("songs_roberta.csv")
embeddings = np.load("lyrics_embeddings_roberta.npy")   # pre-computed SBERT vectors

st.title("🎵 Mood-Based Music Recommender")

user_mood = st.text_input("Describe your mood:", "chill rainy evening")
if st.button("Find songs"):
    query_emb = model.encode(user_mood, normalize_embeddings=True)
    scores = util.cos_sim(query_emb, embeddings)[0]
    top_k = scores.argsort(descending=True)[:5]

    for idx in top_k:
        st.write(f"**{df.iloc[idx]['song']}** – {df.iloc[idx]['Genre']}  \n"
                 f"*Mood:* {df.iloc[idx]['emotion']}")
