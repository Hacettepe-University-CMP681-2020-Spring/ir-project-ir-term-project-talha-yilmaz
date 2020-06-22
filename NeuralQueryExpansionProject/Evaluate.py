from pickle import load
from rank_bm25 import BM25Okapi

expandedtweets = load(open('data/expandedTweets.pkl', 'rb'))
queries = load(open('data/cleanqueriesevaluation.pkl', 'rb'))

tweets = list()
K_value=1

original_and_expanded_queries = list()

for expandedtweet in expandedtweets:
    tweet = expandedtweet['Tweet'].strip()
    tweets.append(tweet)

    original_query = expandedtweet['Original_Query'].strip()
    expanded_query = expandedtweet['Expanded_Query'].strip()

    print("Original query: ",original_query)
    print("Expanded query: ",expanded_query)

    original_and_expanded_queries.append({'Original_Query': original_query, 'Expanded_Query':expanded_query})

tokenized_tweets = [tweet.split(" ") for tweet in tweets]
bm25 = BM25Okapi(tokenized_tweets)

doc_scores_original_query = []
doc_scores_expanded_query = []

reciprocal_rank = 0

for original_and_expanded_query in original_and_expanded_queries:
    original_query = original_and_expanded_query["Original_Query"]
    expanded_query = original_and_expanded_query["Expanded_Query"]

    tokenized_original_query = original_query.split(" ")
    top_relevant_tweet = bm25.get_top_n(tokenized_original_query, tweets, n=K_value)

    tokenized_expanded_query = expanded_query.split(" ")
    all_tweets_by_relevance_expanded = bm25.get_top_n(tokenized_expanded_query, tweets, n=len(tweets))

    expanded_rank_of_top_relevant_tweet = all_tweets_by_relevance_expanded.index(top_relevant_tweet[0].strip())+1

    #if original_query != expanded_query:
    reciprocal_rank += 1/expanded_rank_of_top_relevant_tweet

print("Mean Reciprocal Rank = ", reciprocal_rank/len(tweets))
