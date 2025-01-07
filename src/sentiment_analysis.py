import os
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
from transformers import pipeline
import pymongo
import pickle
from datetime import datetime
from config import MONGO_URI

mongo_uri = MONGO_URI
client = pymongo.MongoClient(mongo_uri)
db = client["anime_reddit"]
posts_collection = db["posts"]

output_dir = 'models'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

vader_analyzer = SentimentIntensityAnalyzer()

huggingface_classifier = pipeline('sentiment-analysis', model='nlptown/bert-base-multilingual-uncased-sentiment', device=0)

def get_vader_sentiment(text):
    sentiment = vader_analyzer.polarity_scores(text)
    if sentiment['compound'] >= 0.05:
        return 1
    elif sentiment['compound'] <= -0.05:
        return -1
    else:
        return 0

def get_textblob_sentiment(text):
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    if polarity > 0:
        return 1
    elif polarity < 0:
        return -1
    else:
        return 0

def get_huggingface_sentiment(texts):
    results = huggingface_classifier(texts)
    return [1 if result['label'] == 'POSITIVE' else -1 if result['label'] == 'NEGATIVE' else 0 for result in results]

def combined_sentiment_analysis(texts):
    vader_sentiments = [get_vader_sentiment(text) for text in texts]
    textblob_sentiments = [get_textblob_sentiment(text) for text in texts]
    huggingface_sentiments = get_huggingface_sentiment(texts)

    sentiments = []
    for i in range(len(texts)):
        sentiments.append(max([vader_sentiments[i], textblob_sentiments[i], huggingface_sentiments[i]], key=[vader_sentiments[i], textblob_sentiments[i], huggingface_sentiments[i]].count))

    return sentiments

def sentiment_analysis():
    posts = posts_collection.find({"sentiment": {"$exists": False}, "cleaned_title": {"$exists": True}})
    data = []

    for post in posts:
        data.append({
            '_id': post.get('_id'),
            'title': post.get('title', ''),
            'content': post.get('content', ''),
            'cleaned_title': post.get('cleaned_title', ''),
            'cleaned_content': post.get('cleaned_content', '')
        })

    if len(data) == 0:
        print("No new posts to analyze.")
        return

    df = pd.DataFrame(data)

    if 'cleaned_title' not in df.columns:
        print("Error: 'cleaned_title' column not found in the dataset.")
        return

    texts = df['cleaned_title'].tolist()
    sentiments = combined_sentiment_analysis(texts)
    df['sentiment'] = sentiments

    if 'sentiment' in df.columns:
        print(df['sentiment'].value_counts())
    else:
        print("Error: 'sentiment' column was not created.")
        return

    for _, row in df.iterrows():
        post_id = row['_id']
        sentiment = row['sentiment']
        posts_collection.update_one(
            {'_id': post_id},
            {'$set': {'sentiment': sentiment}}
        )

    print(f"Sentiment analysis completed for {len(df)} posts and updated in the database.")

    model_path = os.path.join(output_dir, 'sentiment_model.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(huggingface_classifier, f)

    print(f"Sentiment model saved to {model_path}")

if __name__ == '__main__':
    sentiment_analysis()
