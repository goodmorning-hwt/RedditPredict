import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from pymongo import MongoClient
from config import MONGO_URI

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

def clean_text(text):
    if not isinstance(text, str):
        return ""

    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))

    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = text.lower()

    words = nltk.word_tokenize(text)
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]

    return ' '.join(words)

def clean_data():
    mongo_client = MongoClient(MONGO_URI)
    db = mongo_client['anime_reddit']
    posts_collection = db['posts']

    posts = list(posts_collection.find({}))
    df = pd.DataFrame(posts)

    if 'title' not in df.columns or 'content' not in df.columns:
        print("Error: 'title' or 'content' column not found in the dataset.")
        return

    df['title'] = df['title'].fillna('')
    df['content'] = df['content'].fillna('')

    df['cleaned_title'] = df['title'].apply(clean_text)
    df['cleaned_content'] = df['content'].apply(clean_text)

    for _, row in df.iterrows():
        posts_collection.update_one(
            {"_id": row["_id"]},
            {"$set": {
                "cleaned_title": row["cleaned_title"],
                "cleaned_content": row["cleaned_content"]
            }}
        )

    print(f"Cleaned data saved to MongoDB Atlas.")

if __name__ == '__main__':
    clean_data()
