import pandas as pd
from pymongo import MongoClient
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from datetime import datetime
from collections import Counter

from config import MONGO_URI

client = MongoClient(MONGO_URI)
db = client['anime_reddit']
posts_collection = db['posts']
predictions_collection = db['predictions']


def get_data_from_mongo():
    cursor = posts_collection.find({})
    df = pd.DataFrame(list(cursor))
    return df


def clean_data(df):
    print("Data columns:", df.columns)

    if 'created_utc' not in df.columns:
        raise ValueError("The dataset does not contain 'created_utc' column.")

    df['created_utc'] = pd.to_datetime(df['created_utc'], errors='coerce')

    df = df.dropna(subset=['created_utc'])

    df = df.dropna(subset=['sentiment'])

    df['week'] = df['created_utc'].dt.isocalendar().week
    return df


def train_and_predict(df):

    vectorizer = TfidfVectorizer(stop_words='english', max_features=50)
    X = vectorizer.fit_transform(df['cleaned_title'].fillna(''))

    y = df['sentiment']

    print("Sentiment distribution in training data:")
    print(Counter(y))

    model = RandomForestClassifier(class_weight='balanced')
    model.fit(X, y)

    predicted_sentiment = model.predict_proba(X)
    sentiment_distribution = predicted_sentiment.mean(axis=0)

    sentiment_distribution = np.pad(sentiment_distribution, (0, 3 - len(sentiment_distribution)), mode='constant')

    word_freq = np.asarray(X.sum(axis=0)).flatten()
    word_freq_sorted_idx = word_freq.argsort()[::-1]
    predicted_keywords = [vectorizer.get_feature_names_out()[idx] for idx in word_freq_sorted_idx[:5]]

    prediction_data = {
        'next_week': datetime.now().isocalendar()[1] + 1,
        'predicted_keywords': predicted_keywords,
        'sentiment_distribution': sentiment_distribution.tolist()
    }
    predictions_collection.insert_one(prediction_data)

    return predicted_keywords, sentiment_distribution


def main():
    df = get_data_from_mongo()
    df = clean_data(df)

    predicted_keywords, sentiment_dist = train_and_predict(df)

    print(f"Predicted Top Keywords for next week: {predicted_keywords}")
    print(f"Predicted sentiment distribution for next week: -1: {sentiment_dist[0]:.2f}, 0: {sentiment_dist[1]:.2f}, 1: {sentiment_dist[2]:.2f}")


if __name__ == "__main__":
    main()
