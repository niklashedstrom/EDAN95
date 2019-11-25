import numpy as np

def dictionize(file_path):
    dic = {}
    with open(file_path, 'r') as f:
        for line in f:
            splitted_line = line.split()
            word = splitted_line.pop(0)
            dic[word] = [i for i in splitted_line]
    return dic

def tf(t, d, sum):
    """
    t - the word
    d - the dictionary for the document,
    sum - sum of all the words in the given dictionary
    term frequency
    """
    try:
        raw_count = float(len(d[t]))
    except: 
        raw_count = 0.0
    return raw_count/sum

def idf(N, t, D):
    """
    N - number of text files
    t - the word
    D - master dictionary 
    """
    nt = float(len(D[t]))
    if nt == 0.0:
        nt = 1.0
    return np.log10(float(N)/nt)

def tf_idf(t, d, N, D, sum):
    return tf(t, d, sum) * idf(N, t, D)

def cosine_similarity(A, B):
    """
    A, B - arrays containing tfidf values
    """
    return np.dot(A, B) / (np.linalg.norm(A) * np.linalg.norm(B))

if __name__ == "__main__":
    dic = dictionize('glove.6B.100d.txt')
    print(dic['the'])