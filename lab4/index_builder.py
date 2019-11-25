from datasets import load_conll2003_en
from conll_dictorizer import CoNLLDictorizer
from dictionizer import dictionize
from sklearn.feature_extraction import DictVectorizer

import numpy as np

def build_sequences(dic):
    X, Y = [], []
    for sentence in dic:
        x, y = [], []
        for word in sentence:
            x += [word['form']]
            y += [word['ner']]
        X += [x]
        Y += [y]
    return X,Y

def vocabulary(train_dict, dictionizer):
    vocabulary_words = []
    for dic in train_dict:
        for d in dic:
            vocabulary_words.append(d['form'].lower())
    for d in dictionizer:
        vocabulary_words.append(d.lower())
    return set(vocabulary_words)

if __name__ == "__main__":
    train_sentences, dev_sentences, test_sentences, column_names = load_conll2003_en()

    conll_dict = CoNLLDictorizer(column_names, col_sep=' +')
    train_dict = conll_dict.transform(train_sentences)

    X, Y = build_sequences(train_dict)

    dic = dictionize('glove.6B.100d.txt')

    vocabulary = vocabulary(train_dict, dic)

    tmp_y = []
    for y in Y: 
        for y_i in y:
            tmp_y.append(y_i)

    rev_word_idx = dict(enumerate(vocabulary, start=2))
    word_idx = {v: k for k, v in rev_word_idx.items()}

    rev_ner_id = dict(enumerate(set(tmp_y), start=2))
    ner_idx = {v: k for k, v in rev_ner_id.items()}

    M = len(vocabulary) + 2
    N = 100
    matrix = np.random.rand(M, N)

    for word in vocabulary:
        if word in dic.keys():
            matrix[word_idx[word]] = dic[word]

    dict_vect = DictVectorizer(sparse=False)

    X_test = dict_vect.fit_transform(X)

    print(X_test)