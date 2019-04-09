from gensim.models.doc2vec import (
	Doc2Vec,
	TaggedDocument,
)
import multiprocessing
from gensim.models.callbacks import CallbackAny2Vec
from datetime import datetime
import os
import gensim
import pandas as pd
import re
import json

dataset = "dataset/nepal/"

train = dataset + "2015_Nepal_Earthquake_train.tsv"
train_tweets = pd.read_csv(train, sep="\t", encoding="latin")

unlabelled = dataset + "2015_Nepal_Earthquake_unlabelled.txt"
with open(unlabelled, 'r') as f:
	unlabelled_tweets = json.load(f)

tweets = dict()
tag_to_tweet = dict()
tag_to_id = dict()

def parsing(tweet):
	try:
		tweet = tweet.lower() #lowercase
		tweet = re.sub(r'http\S+', "", tweet) #Remove url
		tweet = re.sub("\d", "", tweet)
		tweet = re.sub("[^a-zA-Z\.\@\#\s]", "", tweet)
		words = re.split(" ", tweet)
		new_words = []
		for word in words:
			if len(word) < 2:
				pass
			elif word[0] == '@':
				pass
			elif word[0] == '#':
				pass
			else:
				new_words.append(word)
		tweet = " ".join(new_words)
		tweet = re.sub("\s\s+"," ",tweet)
		return tweet
	except:
		return ""

for index, row in train_tweets.iterrows():
	tweets[row[0]] = parsing(row[1])

for id_ in unlabelled_tweets.keys():
	tweets[id_] = parsing(unlabelled_tweets[id_])

class EpochLogger(CallbackAny2Vec):
	'''Callback to log information about training'''  
	def __init__(self, path):
		self.epoch = 0
		self.path = path
		self.start_time = datetime.now()
	def on_epoch_begin(self,  model):
		pass                                 
	def on_epoch_end(self,  model):
		self.epoch += 1
		print("Epoch #{} end, time elapsed: {}".format(self.epoch, datetime.now()-self.start_time))
		model.save("tweet_doc2vec.model")

class corpus(object):
	"""docstring for corpus"""
	def __init__(self, file):
		super(corpus, self).__init__()
		self.dict = file
	def __iter__(self):
		for i, ID in enumerate(self.dict.keys()):
			tag_to_tweet[i] = tweets[ID]
			tag_to_id[i] = ID
			yield TaggedDocument(gensim.utils.simple_preprocess(tweets[ID]), tags=[i])

epoch_logger = EpochLogger("tweet_doc2vec")
print("model initialised")

model = Doc2Vec(seed=0,workers = multiprocessing.cpu_count(), window=5,  callbacks=[epoch_logger],min_count=5, vector_size=300, epochs=20)

print("Corpus instance initialised")
cor = corpus(tweets)

print("build vocab")
model.build_vocab(cor)

print("train", model.epochs)
model.train(cor, total_examples=model.corpus_count, epochs=model.epochs)
model.save("tweet_doc2vec.model")