import numpy as np
def save_to_npy(filename):
    print("Reading dataset,  filename:{}".format(filename))
    file = np.loadtxt(filename, dtype=str, delimiter='\t',encoding=None,comments=chr(0))
    print("Saving to npy file: {}".format("nepal.npy"))
    np.save("nepal.npy", file)

if __name__ == "__main__":
    save_to_npy("nepal_TMCS_JOURNAL_THREE_10032017.txt")
