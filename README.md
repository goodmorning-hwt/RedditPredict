# README

## Basic Starategy explaination

### Data Collecting

Used reddit API to fetch the content of the newly-updated recent posts from `r/anime`. It can be easily changed to other subreddit, and can even be easily changed to fetching twitter's data. I chose reddit as a testground because `r/anime` is active enough, thus worth data mining; and reddit api is free.

The data is stored in MongoDB Atlas, a easy-to-use free cloud database.

### Data cleaning

To find valuable data, I do data cleaning for these natural language material. Including: removing url/numbers/punctuations/stopwords, convert to lowercase, do tokenization and lemmatization, handle missing values, and finally save cleaned data to MongoDB.

### Topic Modeling

First, we do Laten Dirichlet Allocation (LDA) on the topics, thus created topic_keywords dictionary. Then assign topics to posts through lda.transform.

### Sentiment analysis

The program performs sentiment analysis using a combination of VADER sentiment analysis, TextBlob sentiment analysis, and Hugging Face's pretrained BERT model. It classifies the sentiment of each post using a majority voting method and assigns a sentiment label. Finally, the results are updated in the MongoDB database.

### Predicting

First, extracts keywords from the database using TF-IDF. Then trains a Random Forest model to predict sentiment based on these keywords.
Base on the data from topic modeling and sentiment analysis, it predicts the sentiment distribution for the next week and the top keywords that are likely to be popular in the next week based on keyword frequency.
Finally stores the predictions (keywords and sentiment distribution) in MongoDB.

### Visualization

To do the visualization we use `streamlit`. This is a simple yet fancy library allowing me to quickly create and share data applications.


