import streamlit as st
import pandas as pd
import pymongo
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from datetime import datetime
from config import MONGO_URI

mongo_uri = MONGO_URI
client = pymongo.MongoClient(mongo_uri)
db = client["anime_reddit"]
posts_collection = db["posts"]
predictions_collection = db["predictions"]

def load_data():
    posts = posts_collection.find({"topic_keywords": {"$exists": True}, "sentiment": {"$exists": True}})
    data = []

    for post in posts:
        data.append({
            'id': post.get('id', ''),
            'title': post.get('title', ''),
            'content': post.get('content', ''),
            'created_utc': post.get('created_utc', ''),
            'score': post.get('score', 0),
            'topic': post.get('topic', 'No Topic'),
            'topic_keywords': post.get('topic_keywords', '').split('+'),
            'sentiment': post.get('sentiment', 0),
            'url': post.get('url', ''),
            'cleaned_title': post.get('cleaned_title', None),
            'cleaned_content': post.get('cleaned_content', None)
        })

    df = pd.DataFrame(data)
    df['created_utc'] = pd.to_datetime(df['created_utc'], errors='coerce')

    return df

def show_data(df):
    st.title("Reddit Anime Posts with Topic Modeling and Sentiment Analysis")

    num_posts = st.slider("Select the number of posts to display:", min_value=1, max_value=20, value=10)

    for index, row in df.head(num_posts).iterrows():
        st.subheader(row['title'])
        st.write(f"Score: {row['score']} | Sentiment: {'Positive' if row['sentiment'] == 1 else 'Negative' if row['sentiment'] == -1 else 'Neutral'}")
        st.write(f"Topic: {row['topic']}")
        st.write(f"Keywords: {row['topic_keywords']}")

        post_url = row.get('url', None)
        if post_url:
            st.write(f"[Read more on Reddit]({post_url})")
        else:
            st.write("No URL available")

        st.write(f"Content: {row['content'][:200]}...")
        st.write("---")

def get_predictions():
    predictions = predictions_collection.find().sort("week", pymongo.ASCENDING)

    data = []
    for prediction in predictions:
        data.append({
            'week': prediction.get('week', ''),
            'predicted_sentiment': prediction.get('predicted_sentiment', 0),
            'predicted_topic': prediction.get('predicted_topic', 0)
        })

    return pd.DataFrame(data)

def sentiment_distribution(df):
    sentiment_counts = df['sentiment'].value_counts()
    sentiment_labels = ['Positive', 'Neutral', 'Negative']

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(x=sentiment_labels, y=sentiment_counts, ax=ax, palette="Set2")
    ax.set_title('Sentiment Distribution')
    st.pyplot(fig)

def word_cloud(df):
    if 'cleaned_title' in df.columns and df['cleaned_title'].notna().any():
        text = ' '.join(df['cleaned_title'].dropna())
    elif 'title' in df.columns and df['title'].notna().any():
        text = ' '.join(df['title'].dropna())
    elif 'topic_keywords' in df.columns and df['topic_keywords'].notna().any():
        text = ' '.join(df['topic_keywords'].dropna())
    else:
        st.warning("No valid text columns found for word cloud.")
        return

    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)

    st.subheader("Word Cloud of Post Titles/Keywords")
    st.image(wordcloud.to_array(), use_container_width=True)

def topic_distribution(df):
    topic_counts = df['topic'].value_counts()

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.barplot(x=topic_counts.index, y=topic_counts.values, ax=ax, palette="viridis")
    ax.set_title('Topic Distribution')
    ax.set_ylabel('Number of Posts')
    ax.set_xlabel('Topic')
    st.pyplot(fig)

def topic_distribution_over_time(df):
    df['date'] = df['created_utc'].dt.date

    topic_time_distribution = df.groupby(['date', 'topic']).size().unstack(fill_value=0)

    st.subheader("Topic Distribution Over Time")
    st.line_chart(topic_time_distribution)

def sentiment_over_time(df):
    df['date'] = df['created_utc'].dt.date

    sentiment_count = df.groupby(['date', 'sentiment']).size().unstack(fill_value=0)

    st.subheader("Sentiment Distribution Over Time")
    st.line_chart(sentiment_count)

def show_predictions():
    predictions = get_predictions()

    if predictions.empty:
        st.warning("No predictions available.")
        return

    st.subheader("Predicted Sentiment and Topic Trends for Next Week")
    st.write(predictions)

    sentiment_fig, sentiment_ax = plt.subplots(figsize=(6, 4))
    sns.lineplot(data=predictions, x='week', y='predicted_sentiment', ax=sentiment_ax, marker='o', color='b')
    sentiment_ax.set_title('Predicted Sentiment Trend')
    sentiment_ax.set_xlabel('Week')
    sentiment_ax.set_ylabel('Sentiment')
    st.pyplot(sentiment_fig)

    topic_fig, topic_ax = plt.subplots(figsize=(6, 4))
    sns.lineplot(data=predictions, x='week', y='predicted_topic', ax=topic_ax, marker='o', color='g')
    topic_ax.set_title('Predicted Topic Trend')
    topic_ax.set_xlabel('Week')
    topic_ax.set_ylabel('Topic Count')
    st.pyplot(topic_fig)

def main():
    df = load_data()

    show_data(df)

    sentiment_distribution(df)
    word_cloud(df)
    topic_distribution(df)
    topic_distribution_over_time(df)
    sentiment_over_time(df)

    show_predictions()

if __name__ == '__main__':
    main()
