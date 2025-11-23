# Feel-The-Music
A Mood Based Music Recommendation System

Feel-The-Music/
├── Dockerfile
├── README.md
├── data
│   ├── in
│   │   └── spotify_dataset.csv
│   └── out
│       ├── lyrics_embeddings_roberta.npy
│       ├── lyrics_embeddings_roberta_test.npy
│       ├── songs_roberta.csv
│       └── songs_roberta_test.csv
├── requirements.txt
└── src
    │   └── run_recommender.cpython-313.pyc
    ├── app.py
    ├── create_recommender.py
    ├── run_recommender.py
    └── test.ipynb



# How it works?
SBERT is used for understanding the meaning and emotion of the song lyrics as well as the user queries. SBERT is used for sentence embeddings. This allows us to look at the meanings of sentences and figure out which queries closely resemble which songs.

# Dataset
To train the reccommender model you will need to download the spotify dataset from here: https://www.kaggle.com/datasets/devdope/900k-spotify/data

The file `spotify_dataset.csv` will need to be stored in `data/in/`

This dataset contains a variety of information from over 500k+ songs on spotify including emotion, genre, lyrics, song information, etc.

# Data Cleaning
The following columns were merged into one string: `song` `text` `Genre` `emotion`. newlines, markings for chorus and verses, and random characters were removed from the text. Songs which had no lyrics were also removed from the dataset.

# How to Run

First you will need to run `prep_data.py`. This will prepare the data for use with the SBERT. You will need the `spotify_dataset.csv` in the `data/in/` folder as specified above. This will create a new file called `cleaned_spotify_dataset.csv` in `data/in/`.

Next you will need to run `create_recommender.py`. This will save both the cleaned dataset with lyric embeddings and a separate file containing just the lyric embeddings as `.csv` and `.npy` files in the `data/out/` folder.

`run_recommender.py` contains a function `recommend_songs` which you can call and submit a query to, so that you can manually retrieve n number of songs best suited to your songs.

## To run the app

You will need to build the docker container. It is very simple as the Dockerfile is included in the repo. 

1. Navigate to the root folder which contains the Dockerfile and run `docker build -t mood-recommender .`
2. Once it has been built run `docker run -p 9332:9332 mood-recommender`
3. Open your browser and enter `http://localhost:9332`

# SBERT Model Choice

The SBERT uses a pre-trained model: `all-roberta-large-v1` to convert the prepped text into embeddings. This was chosen after attempts with `all-mpnet-base-v2`, `twitter-roberta-base-emotion`. Tests were made using the following queries: `I was cheated on, give me rnb songs to listen to`, `i want happy pop songs`, `I am feeling sad`, `i am very angry, give me rock songs for my anger`. `all-roberta-large-v1` was the only model that was able to produce songs that not only matched the emotion well, but also the suggested genre as well. The models `all-mpnet-base-v2` and `twitter-roberta-base-emotion` produced results where the lyrics seemed to mention the emotion directly, a good number of times. It seemed like instead of producing results that matched the emotion, it was finding songs that had lyrics that contained as much of the exact emotion as possible. My goal was to create an app that can effectively produce song suggestions for the query, best fitting the user's requirements. Not just finding songs that contained the most mentions of the emotion as possible. This made `all-roberta-large-v1` the best choice in this use case.

