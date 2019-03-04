import gensim
import nltk
import numpy as np
import re
import sklearn.manifold
import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pylab import figure
from sklearn.decomposition import PCA
import multiprocessing
from datetime import datetime
from gensim.models.callbacks import CallbackAny2Vec

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
        if self.epoch%5 == 0:
            print("######################################")
            model.save('{}_epoch{}.model'.format(self.path, self.epoch))

print("Reading corpus")
file = np.loadtxt('corpus.txt', dtype=str, delimiter = '\t', comments=chr(0))
corp = []

start = datetime.now()
print("Tokenize corpus")
for sen in file[:, 1]:
	clean = re.sub("[^a-zA-Z]"," ",sen.lower())
	corp.append(nltk.word_tokenize(clean))

print("Tokens done: {}".format(datetime.now()-start))
np.savetxt('tokens.txt', corp, fmt='%s', comments=chr(0), delimiter='\t')

print("Skip gram model initialised")

epoch_logger = EpochLogger("./model/nepal")
model = gensim.models.Word2Vec(min_count=1,size=300, seed=0,sg=1,workers = multiprocessing.cpu_count(), window=5,  callbacks=[epoch_logger])
model.build_vocab(corp)

print("total_sentences: {}, total_words: {}".format(model.corpus_count, len(model.wv.vocab.keys())))
print("Start training...")
model.train(corp, total_examples=model.corpus_count, epochs=30)

print("Final model saved as {}".format('./model/nepal.model'))
model.save('./model/nepal.model')

tsne2d = sklearn.manifold.TSNE(n_components=2, random_state=0)
tsne3d = sklearn.manifold.TSNE(n_components=3, random_state=0)

vector_matrix = model.wv.vectors

pca2d = PCA(n_components=80)
pca3d = PCA(n_components=80)

print("Use PCA to reduce size for 2d graph")
transformed_vector2d = pca2d.fit_transform(vector_matrix)
print("Use PCA to reduce size for 3d graph")
transformed_vector3d = pca3d.fit_transform(vector_matrix)

print("tSNE - 2d graph")
start = datetime.now()
vector_matrix_2d = tsne2d.fit_transform(transformed_vector2d)
print("Time elapsed for tSNE 2d: {}".format(datetime.now() - start))
print("tSNE - 3d graph")
start = datetime.now()
vector_matrix_3d = tsne3d.fit_transform(transformed_vector3d)
print("Time elapsed for tSNE 3d: {}".format(datetime.now() - start))

print("Save t-SNE 2d values")
np.save("vector_2d.npy", vector_matrix_2d)
print("Save t-SNE 3d values")
np.save("vector_3d.npy", vector_matrix_3d)

#plt.scatter(vector_matrix_2d[:,0], vector_matrix_2d[:,1])


#for i, word_value in enumerate(model.wv.vocab.values()):
#    index = word_value.index
#    plt.text(vector_matrix_2d[index][0], vector_matrix_2d[index][1], list(model.wv.vocab.keys())[i])
#plt.show()



#plt.savefig("plot.png")
# fig = figure()
# ax = Axes3D(fig)


# ax.scatter(vector_matrix_3d[:,0], vector_matrix_3d[:,1], vector_matrix_3d[:,2])

# for i, word_value in enumerate(model.wv.vocab.values()):
# 	index = word_value.index
# 	ax.text(vector_matrix_3d[index][0], vector_matrix_3d[index][1], vector_matrix_3d[index][2], model.wv.vocab.keys()[i])

