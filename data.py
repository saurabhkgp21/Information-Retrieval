import numpy as np
def save_to_npy(filename):
    print("Reading dataset,  filename:{}".format(filename))
    file = np.loadtxt(filename, dtype=str, delimiter='\t',encoding=None,comments=chr(0))
    print("Saving to npy file: {}".format("nepal.npy"))
    np.save("nepal.npy", file)

def extract_corpus(filename):
    print("Reading dataset, filename:{}".format(filename))
    file = np.loadtxt(filename, dtype=str, delimiter='\t',encoding=None,comments=chr(0))
    corpus = file[:,[1,3]]
    print("Saving corpus, filename:{}".format("raw_corpus.txt"))
    np.savetxt("raw_corpus.txt",corpus, fmt="%s", comments=chr(0), delimiter='\t')

if __name__ == "__main__":
    extract_corpus("nepal_TMCS_JOURNAL_THREE_10032017.txt")    
    #save_to_npy("nepal_TMCS_JOURNAL_THREE_10032017.txt")
