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
from datetime import datetime
import concurrent.futures

dataset = "dataset/nepal/"

train = dataset + "2015_Nepal_Earthquake_train.tsv"
train_tweets = pd.read_csv(train, sep="\t", encoding="latin")

unlabelled = dataset + "small_unlabelled_tweets.txt"
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



for index, row in train_tweets.iterrows():
	tweets[row[0]] = parsing(row[1])

for id_ in list(unlabelled_tweets.keys()):
	tweets[id_] = parsing(unlabelled_tweets[id_])

model= Doc2Vec.load("tweet_doc2vec.model")
tweet_vectors = dict()

for id_ in tweets.keys():
	sent = tweets[id_]
	vec = model.infer_vector(gensim.utils.simple_preprocess(sent))
	tweet_vectors[id_] = vec

tweet_ids = list(tweets.keys())

from scipy import spatial
print("Building KD tree")
tree = spatial.KDTree(list(tweet_vectors.values()))

graph_context = dict()
print("Building graph")
start_time = datetime.now()

def build_graph(id_):
	y_vec, y_ind = tree.query(tweet_vectors[id_], k=10)
	p = list()
	for j in y_ind:
		p.append(tweet_ids[j])
	return id_, p
	# graph_context[id_] = p

count = 0
start_time = datetime.now()
with concurrent.futures.ProcessPoolExecutor() as executor:
	for id_, p in executor.map(build_graph, tweets.keys()):
		graph_context[id_] = p
		count += 1
		if (count+1)%200==0:
			print("Building graph ,Time: {}, {}/{}".format(datetime.now() - start_time, count + 1, len(tweets.keys())))


with open("dataset/nepal/graph.txt","w") as f:
	json.dump(graph_context, f)

t_vec = dict()

def create_dict(key):
	return key, tweet_vectors[key].tolist()

start_time = datetime.now()
count = 0
with concurrent.futures.ProcessPoolExecutor() as executor:
	for k, l in executor.map(create_dict, tweet_vectors.keys()):
		t_vec[k] = l
		count += 1
		if (count+1)%2000 == 0:
			print("Manipulating graph, time elpased: {}, {}/{}".format(datetime.now() - start_time, count + 1, len(tweet_vectors.keys())))

with open("dataset/nepal/tweet_vectors.txt","w") as f:
	json.dump(t_vec, f)
