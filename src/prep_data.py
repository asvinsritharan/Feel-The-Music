import pandas as pd
import re

def clean_lyrics(text):
    text = str(text)
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text.strip().lower()

def create_clean_lyrics(df):
    df["lyrics_and_data"] = "The title of this song is " + df["song"]+ " - " + df["text"] + " - " + ". The genre of this song is "+ df["Genre"] + " - " + "The emotion of this song is " + df["emotion"]
    df["clean_lyrics"] = df["lyrics_and_data"].apply(clean_lyrics)
    df = df.dropna(subset=["clean_lyrics"])
    df["clean_lyrics"] = df["clean_lyrics"].str.slice(0, 1000)
    return df

if __name__ == "__main__":
    df = pd.read_csv("../data/in/spotify_dataset.csv")
    cleaned_lyrics = create_clean_lyrics(df)
    cleaned_lyrics.to_csv("../data/in/cleaned_spotify_dataset.csv")