from pickle import load, dump
from gensim.models import KeyedVectors

predictedquerytweets = load(open('data/predictedquerytweets.pkl', 'rb'))

tweeetList=list()
predictedQueryList=list()
expandedTweets=list()

expansion_word_count=2

model = KeyedVectors.load_word2vec_format('data/GoogleNews-vectors-negative300.bin', binary=True)

for predictedquerytweet in predictedquerytweets:
    tweet = predictedquerytweet['Tweet'].strip()
    original_query = predictedquerytweet['Original_Query'].strip()
    predicted_query = predictedquerytweet['Predicted_Query'].strip()

    predictedWords = predicted_query.split(" ")

    similarityDict = {}

    expanded_query = list()

    for or_q in original_query.split(" "):
        expanded_query.append(or_q)

    for predictedqueryword in predictedWords:
        wordscore = 0
        for tweetword in tweet.split(" "):
            if len(tweetword) > 0:
                try:
                    score = model.similarity(predictedqueryword, tweetword)
                    wordscore += score
                except:
                    print("Error")

        similarityDict[predictedqueryword] = wordscore/len(tweet.split(" "))

    sorted_dict = sorted(similarityDict.items(), reverse=True, key=lambda x: x[1])

    for i in range(0,expansion_word_count):
        word = sorted_dict[i][0]

        if word not in expanded_query:
            expanded_query.append(word)

    expandedString=""
    for word in expanded_query:
        expandedString = expandedString + " " + word

    expandedTweets.append({"Tweet":tweet,"Original_Query":original_query,"Expanded_Query":expandedString})

dump(expandedTweets, open('data/expandedTweets.pkl', 'wb'))