from flask import Flask, request, jsonify, render_template_string
from run_recommender import recommend_songs
import pandas as pd
import numpy as np

app = Flask(__name__)

df = pd.read_csv("../data/out/songs_roberta.csv")
embeddings = np.load("../data/out/lyrics_embeddings_roberta.npy")
# Simple homepage with form
@app.route("/", methods=["GET"])
def home():
    return render_template_string("""
        <h2>🎵 Mood-Based Music Recommender</h2>
        <form method="POST" action="/recommend">
            <input name="query" placeholder="Enter your mood or theme" size="50">
            <button type="submit">Find Songs</button>
        </form>
    """)

# Route for form submission (browser)
@app.route("/recommend", methods=["POST"])
def recommend_form():
    user_query = request.form["query"]
    results = recommend_songs(df, embeddings, user_query, top_k=10)
    html = f"<h3>Results for: '{user_query}'</h3><ul>"
    for _, row in results.iterrows():
        html += f"<li><b>{row['song']}</b> by {row['Artist(s)']}<br><small>{row['clean_lyrics'][:150]}...</small></li>"
    html += "</ul><a href='/'>Search again</a>"
    return html

# Route for API access (JSON)
@app.route("/api/recommend", methods=["POST"])
def recommend_api():
    data = request.get_json()
    query = data.get("query", "")
    results = recommend_songs(df, embeddings, query, top_k=10)
    response = results.to_dict(orient="records")
    return jsonify(response)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=9332)