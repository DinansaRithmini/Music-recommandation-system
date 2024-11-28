from flask import Flask, render_template, request
import pandas as pd
from sklearn.neighbors import NearestNeighbors

app = Flask(__name__, template_folder="templates")

# Load dataset (ensure the file is in the same directory or provide a full )
data = pd.read_csv("Music.csv")

# Features to use for the recommendation system
audio_features = ["danceability", "energy", "valence", "tempo", "acousticness"]


def recommend_songs(track_genre, popularity, data, audio_features):
    # Filter songs by genre and popularity
    filtered_songs = data[(data["track_genre"] == track_genre) & 
                          (data["popularity"] >= popularity)]
    
    if filtered_songs.empty:
        return {
          "error": f"""No tracks found for the genre '{track_genre}' 
                     with popularity >= {popularity}."""
               }
    
    # Extract features for the filtered songs
    features = data[audio_features]
    knn = NearestNeighbors(n_neighbors=5, metric="cosine")
    knn.fit(features)
    
    recommendations = []
    count = 0
    for _, song in filtered_songs.iterrows():
        track_features = features.iloc[song.name].values  
        distances, indices = knn.kneighbors([track_features])
        
        for _, rec_song in data.iloc[indices[0]].iterrows():
            recommendations.append({
                "track_name": rec_song["track_name"],
                "artists": rec_song["artists"],
                "popularity": rec_song["popularity"]
            })
            count += 1
            if count >= 10:  # Limit to 10 recommendations
                break
        if count >= 10:
            break

    return recommendations


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        track_genre = request.form.get("track_genre", "").strip()
        popularity = request.form.get("popularity")

        # Check if the inputs are valid
        if not track_genre:
            return render_template("index.html", error="Please enter a genre.")
        
        try:
            popularity = int(popularity)
        except (ValueError, TypeError):
            return render_template(
                "index.html", 
                error="Please enter a valid number for popularity.")

        # Get recommendations
        recommendations = recommend_songs(track_genre, popularity, data,
                                          audio_features)
        if "error" in recommendations:
            return render_template("index.html", 
                                   error=recommendations["error"])
        return render_template("index.html", recommendations=recommendations, 
                               track_genre=track_genre)

    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)




