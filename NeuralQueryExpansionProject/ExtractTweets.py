import json
from langdetect import detect
from pickle import dump

tweets=[]

for line in open('json/all.json', 'r'):
    line = json.loads(line)
    if len(line['text']) > 0:
        try:
            if detect(line['text']) == 'en':
                tweets.append(line['text'])
                print(line['text'])
        except:
            language = "error"
            print("This row throws and error")

print(len(tweets))
dump(tweets, open('data/extractedtweets.pkl', 'wb'))




