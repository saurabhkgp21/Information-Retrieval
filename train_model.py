
# coding: utf-8

# In[2]:


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from datetime import datetime
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR

import gensim
from gensim.models.doc2vec import Doc2Vec
from gensim.models.callbacks import CallbackAny2Vec

import json
import pandas as pd


# In[3]:


class model(nn.Module):
	def __init__(self, in_dim, z1_dim, z2_dim, z3_dim, z4_dim):
		super(model, self).__init__()
		self.z1 = nn.Linear(in_dim, z1_dim)
		self.z2 = nn.Linear(z1_dim, z2_dim)
		self.z3 = nn.Linear(z1_dim, z3_dim)
		self.z4 = nn.Linear(z3_dim, z4_dim)
		self.out = nn.Linear(z2_dim + z4_dim, 1)

	def forward(self, x):
		z1 = self.z1(x)
		z2 = self.z2(z1)
		z3 = self.z3(z1)
		z4 = self.z4(z3)
		out = self.out(torch.cat((z2,z4), dim=1))
		return torch.sigmoid(out)

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


# In[4]:


dataset = "dataset/nepal/"
train = dataset + "train.csv"
train_tweets = pd.read_csv(train, sep="\t", encoding="latin")
print(train_tweets.shape)
doc_model = Doc2Vec.load("tweet_doc2vec.model")
tweets = dict()
with open("dataset/nepal/tag_to_id.txt", 'r') as f:
	tag_to_id = json.load(f)

for index, row in train_tweets.iterrows():
	if row[3] == "relevant":
		tweets[row[0]] = 1
	else:
		tweets[row[0]] = 0
	# tweets[row[0]] = row[3]

num_epocs = 20
total_tags = len(doc_model.docvecs)
batch_size = 100

m = model(300, 120, 30, 60, 30)

labelled_vectors = []
class_labels = []
unlabelled_vectors = []
label_unlabel = dict()


for tag in tag_to_id.keys():
	id_ = tag_to_id[tag]
	if id_ in tweets.keys():
		label_unlabel[int(tag)] = 1
		labelled_vectors.append(doc_model.docvecs[int(tag)])
		class_labels.append(tweets[id_])
	else:
		label_unlabel[int(tag)] = 0
		unlabelled_vectors.append(doc_model.docvecs[int(tag)])

total_label = len(labelled_vectors)
total_unlabelled = len(unlabelled_vectors)


# In[16]:


m = model(300, 120, 30, 60, 30)
m_optimizer = torch.optim.Adam(m.parameters(), lr=0.001)
# scheduler = ReduceLROnPlateau(m_optimizer, 'min')
scheduler = StepLR(m_optimizer, step_size=200, gamma=0.5)
l_history = []

for step in range(1000):
    scheduler.step()
    start = (step*batch_size)%total_label
    end = start + batch_size
    vectors = labelled_vectors[start:end]
    labels = class_labels[start:end]
    targets = torch.tensor(labels, dtype=torch.float).view(len(labels),1)
    m.zero_grad()
    out = m(torch.tensor(vectors))

    loss_function = nn.BCELoss()
    loss = loss_function(out, targets)
    l_history.append(loss)
    loss.backward()
    m_optimizer.step()
    


# In[17]:


import matplotlib.pyplot as plt
plt.plot(l_history)
plt.show()
print(l_history[-1])

