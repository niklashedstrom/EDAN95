from datasets import load_conll2003_en
from conll_dictorizer import CoNLLDictorizer
from dictionizer import dictionize
from sklearn.feature_extraction import DictVectorizer
from keras.preprocessing.sequence import pad_sequences

from keras import models, layers
from keras.utils import to_categorical
from keras.layers import SimpleRNN, Dense

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

def to_index(seq, idx):
    tmp_seq = []
    for s in seq:
        s_idx = []
        for l in s:
            #Get the value, if not in word_idx => 0 else value
            if l in idx:
                s_idx.append(word_idx[l])
            else:
                s_idx.append(0)
        tmp_seq.append(s_idx)
    print(tmp_seq)

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

    nb_classes = len(set(tmp_y))
    print(nb_classes)

    M = len(vocabulary) + 2
    print(M)
    N = 100
    matrix = np.random.rand(M, N)

    for word in vocabulary:
        if word in dic.keys():
            matrix[word_idx[word]] = dic[word]

    dict_vect = DictVectorizer(sparse=False)

    #Ta X och gör om till index-värden
    X_idx = to_index(X, word_idx)
    # for x in X:
        # x_idx = []
        # for l in x:
            # #Get the value, if not in word_idx => 0 else value
            # if l in word_idx:
                # x_idx.append(word_idx[l])
            # else:
                # x_idx.append(0)
        # X_idx.append(x_idx)

    Y_idx = to_index(Y, ner_idx)
    # for y in Y:
        # y_idx = []
        # for l in y:
            # if l in ner_idx:
                # y_idx.append(ner_idx[l])
            # else:
                # y_idx.append(0)
        # Y_idx.append(y_idx)

    padded_x = pad_sequences(X_idx, maxlen=150)
    padded_y = pad_sequences(Y_idx, maxlen=150)

    y_train = to_categorical(padded_y, num_classes=nb_classes + 2)

    model = models.Sequential()
    model.add(layers.Embedding(
        M,
        N,
        mask_zero=True,
        input_length=None
    ))
    # model.layers[0].set_weights((matrix))

    model.add(SimpleRNN(100, return_sequences=True))
    model.add(Dense(nb_classes + 2, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                optimizer='rmsprop',
                metrics=['acc'])
    
    model.summary()

    model.fit(padded_x, y_train, epochs=2, batch_size=128)