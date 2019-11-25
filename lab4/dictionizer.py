import numpy as np
import operator

def dictionize(file_path):
    dic = {}
    with open(file_path, 'r') as f:
        for line in f:
            splitted_line = line.split()
            word = splitted_line.pop(0)
            dic[word] = [float(i) for i in splitted_line]
    return dic

def cosine_similarity(A, B):
    """
    A, B - arrays containing tfidf values
    """
    return np.dot(A, B) / (np.linalg.norm(A) * np.linalg.norm(B))

def top_five_CS(master):
    words = ["table", "france", "sweden"]
    dics = {"table": {}, "france": {}, "sweden" : {}}
    for m in master:
        for w in words:
            dics[w][m] = cosine_similarity(np.array(master[m]), np.array(master[w]))
    
    sorted_dics = {}
    for d in dics:
        sorted_dics[d] = sorted(dics[d].items(), key=lambda x : x[1], reverse=True)[1:6]
    
    return sorted_dics



if __name__ == "__main__":
    dic = dictionize('glove.6B.100d.txt')
    #print(dic['the'])
    dics = top_five_CS(dic)
    for d in dics:
        print("{}: {}".format(d, dics[d]))