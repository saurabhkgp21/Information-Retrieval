import numpy as np
def save_to_npy(filename):
    print("Reading dataset,  filename:{}".format(filename))
    file = np.loadtxt(filename, dtype=str, delimiter='\t',encoding=None,comments=chr(0))
    print("Saving to npy file: {}".format("nepal.npy"))
    np.save("nepal.npy", file)

def extract_raw_corpus(filename):
    print("Reading dataset, filename:{}".format(filename))
    file = np.loadtxt(filename, dtype=str, delimiter='\t',encoding=None,comments=chr(0))
    corpus = file[:,[1,3]]
    print("Saving corpus, filename:{}".format("raw_corpus.txt"))
    np.savetxt("raw_corpus.txt",corpus, fmt="%s", comments=chr(0), delimiter='\t')

def process_corpus(filename):
    print("Loading Raw corpus")
    try:
        file = np.loadtxt(filename, dtype=str, delimiter='\t', encoding=None, comments=chr(0))
    except Exception as e:
        print(e)
        return
    temp = np.empty((file.shape[0], 3), dtype=file.dtype)
    temp[:, 0] = file[:, 0]
    for i, sentence in enumerate(file[:, 1]):
        word_list = sentence.split()
        normal_text = []
        hashtag_and_mentions = []
        for word in word_list:
            if word[0] == '#' or word[0] == '@':
                hashtag_and_mentions.append(word)
            elif word[0:4] == 'http':
                pass
            else:
                normal_text.append(word)
        temp[i, 1] = " ".join(normal_text)
        temp[i, 2] = " ".join(hashtag_and_mentions)
    np.savetxt("corpus.txt", temp,  fmt="%s", comments=chr(0), delimiter='\t')

if __name__ == "__main__":
    process_corpus("raw_corpus.txt")
    #extract_corpus("nepal_TMCS_JOURNAL_THREE_10032017.txt")    
    #save_to_npy("nepal_TMCS_JOURNAL_THREE_10032017.txt")
