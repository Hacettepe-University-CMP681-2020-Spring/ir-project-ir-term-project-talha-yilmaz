from rank_bm25 import BM25Okapi
from pickle import load, dump

corpus = load(open('data/cleantweets.pkl', 'rb'))

queries = load(open('data/cleanqueries.pkl', 'rb'))
tokenized_queries = [doc.split(" ") for doc in queries]

bm25 = BM25Okapi(tokenized_queries)

query_tweets = list()

for tweet in corpus:
    tokenized_tweet = tweet.split(" ")
    doc_scores = bm25.get_scores(tokenized_tweet)

    best_matching_doc = bm25.get_top_n(tokenized_tweet, queries, n=1)

    query_tweets.append({'Tweet': tweet, 'Query': best_matching_doc[0]})

dump(query_tweets, open('data/querytweets.pkl', 'wb'))