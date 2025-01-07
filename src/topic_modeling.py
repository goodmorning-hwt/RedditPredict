import pymongo
import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from config import MONGO_URI

client = pymongo.MongoClient(MONGO_URI)

db = client['anime_reddit']
posts_collection = db['posts']

def get_data_from_mongo():
    posts = posts_collection.find({'topic': {'$exists': False}}, {'_id': 1, 'cleaned_title': 1, 'cleaned_content': 1})
    data = list(posts)
    return pd.DataFrame(data)

def topic_modeling(df):
    vectorizer = CountVectorizer(stop_words='english')
    X = vectorizer.fit_transform(df['cleaned_title'])

    lda = LatentDirichletAllocation(n_components=5, random_state=42)
    lda.fit(X)

    topic_keywords = {}
    for topic_idx, topic in enumerate(lda.components_):
        topic_keywords[topic_idx] = [vectorizer.get_feature_names_out()[i] for i in topic.argsort()[:-10 - 1:-1]]

    topic_assignments = lda.transform(X).argmax(axis=1)
    df['topic'] = topic_assignments
    df['topic_keywords'] = df['topic'].map(lambda x: ' + '.join(topic_keywords[x]))
    return df

def save_topic_modeling_result(df):
    for _, row in df.iterrows():
        post_id = row['_id']
        topic = row['topic']
        topic_keywords = row['topic_keywords']
        posts_collection.update_one(
            {'_id': post_id},
            {'$set': {'topic': topic, 'topic_keywords': topic_keywords}}
        )

def perform_topic_modeling():
    df = get_data_from_mongo()
    if not df.empty:
        df = topic_modeling(df)
        save_topic_modeling_result(df)
        print(f"Topic modeling completed for {len(df)} posts and updated in the database.")
    else:
        print("No posts need topic modeling.")

if __name__ == "__main__":
    perform_topic_modeling()
