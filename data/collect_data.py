# collect data
import praw
import pandas as pd
import time
from datetime import datetime
from pymongo import MongoClient

from config import MONGO_URI, USER_AGENT, CLIENT_SECRET, CLIENT_ID


def collect_data(subreddit_name='anime', limit=500, rate_limit=8):
    reddit = praw.Reddit(
        client_id=CLIENT_ID,
        client_secret=CLIENT_SECRET,
        user_agent=USER_AGENT
    )


    mongo_client = MongoClient(MONGO_URI)
    db = mongo_client['anime_reddit']
    posts_collection = db['posts']

    subreddit = reddit.subreddit(subreddit_name)

    posts = []

    for submission in subreddit.new(limit=limit):
        post_data = {
            'title': submission.title,
            'score': submission.score,
            'id': submission.id,
            'url': submission.url,
            'created_utc': datetime.utcfromtimestamp(submission.created_utc),
            'comments': submission.num_comments,
            'author': submission.author.name if submission.author else 'N/A',
            'content': submission.selftext
        }

        posts_collection.update_one(
            {"id": submission.id},
            {"$set": post_data},
            upsert=True
        )

        posts.append(post_data)

        print(f"Post {submission.id} saved to MongoDB.")

        time.sleep(1 / rate_limit)

    print(f"Data collection completed. {len(posts)} posts saved to MongoDB.")

if __name__ == '__main__':
    collect_data(subreddit_name='anime', limit=500, rate_limit=8)
