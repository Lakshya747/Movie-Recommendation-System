import pandas as pd
import numpy as np
import os
import zipfile
import urllib.request
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split

if not os.path.exists('ml-latest-small'):
    urllib.request.urlretrieve(
        'https://files.grouplens.org/datasets/movielens/ml-latest-small.zip',
        'ml-latest-small.zip'
    )
    with zipfile.ZipFile('ml-latest-small.zip', 'r') as z:
        z.extractall('.')

ratings = pd.read_csv('ml-latest-small/ratings.csv')
movies = pd.read_csv('ml-latest-small/movies.csv')

matrix = ratings.pivot_table(
    index='userId',
    columns='movieId',
    values='rating'
).fillna(0)

sim = cosine_similarity(matrix.T)
sim_df = pd.DataFrame(sim, index=matrix.columns, columns=matrix.columns)

def find_movie(title):
    match = movies[movies['title'].str.contains(title, case=False, na=False)]
    if match.empty:
        return None, None
    return match.iloc[0]['movieId'], match.iloc[0]['title']

def recommend(title, n=5):
    mid, full = find_movie(title)
    if mid is None:
        print("Movie not found")
        return
    if mid not in sim_df.columns:
        print("Not enough data")
        return

    scores = sim_df[mid].sort_values(ascending=False).drop(mid, errors='ignore')
    top_ids = scores.head(n).index.tolist()
    results = movies[movies['movieId'].isin(top_ids)][['title', 'genres']]

    print(f"\nBecause you watched: {full}")
    print("Recommendations:\n")
    for i, row in enumerate(results.itertuples(), 1):
        print(f"  {i}. {row.title} | {row.genres}")

def evaluate():
    active = ratings.groupby('userId').filter(lambda x: len(x) >= 20)
    train, test = train_test_split(active, test_size=0.2, random_state=42)
    avg = train.groupby('userId')['rating'].mean()
    global_avg = train['rating'].mean()

    preds = []
    actual = []

    for _, row in test.iterrows():
        uid = row['userId']
        preds.append(avg.get(uid, global_avg))
        actual.append(row['rating'])

    rmse = np.sqrt(np.mean((np.array(preds) - np.array(actual)) ** 2))
    print(f"RMSE: {rmse:.4f}")

evaluate()

def run():
    print("\n--- Movie Recommendation System ---\n")
    while True:
        print("1. Get recommendations")
        print("2. Quit")
        choice = input("\nChoice: ").strip()

        if choice == '2':
            break
        elif choice == '1':
            title = input("Movie name: ").strip()
            try:
                n = int(input("How many recommendations: ").strip())
            except ValueError:
                n = 5
            recommend(title, n)
        else:
            print("Enter 1 or 2")

run()
