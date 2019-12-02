from datasets import load_conll2003_en
from conll_dictorizer import CoNLLDictorizer
from dictionizer import dictionize
from sklearn.feature_extraction import DictVectorizer
from keras.preprocessing.sequence import pad_sequences

from keras import models, layers
from keras.utils import to_categorical
from keras.layers import LSTM, Bidirectional, SimpleRNN, Dense

import numpy as np

def build_sequences(dic):
    X, Y = [], []
    for sentence in dic:
        x, y = [], []
        for word in sentence:
            x.append(word['form'].lower())
            y.append(word['ner'])
        X.append(x)
        Y.append(y)
    return X,Y

def vocabulary(train_dict, dictionizer):
    vocabulary_words = []
    for dic in train_dict:
        for d in dic:
            vocabulary_words.append(d['form'].lower())
    for d in dictionizer:
        vocabulary_words.append(d)
    return sorted(list(set(vocabulary_words)))

def to_index(seq, idx):
    tmp_seq = []
    for s in seq:
        s_idx = []
        for l in s:
            #Get the value, if not in word_idx => 0 else value
            if l in idx:
                s_idx.append(idx[l])
            else:
                s_idx.append(0)
        tmp_seq.append(s_idx)
    return tmp_seq

if __name__ == "__main__":
    train_sentences, dev_sentences, test_sentences, column_names = load_conll2003_en()

    conll_dict = CoNLLDictorizer(column_names, col_sep=' +')
    train_dict = conll_dict.transform(train_sentences)
    test_dict = conll_dict.transform(test_sentences)


    X, Y = build_sequences(train_dict)

    dic = dictionize('glove.6B.100d.txt')

    vocabulary = vocabulary(train_dict, dic)

    #tmp_y = []
    #for y in Y: 
    #    for y_i in y:
    #        tmp_y.append(y_i)

    tmp_y = sorted(list(set([ner for sentence in Y for ner in sentence])))

    rev_word_idx = dict(enumerate(vocabulary, start=2))
    word_idx = {v: k for k, v in rev_word_idx.items()}

    rev_ner_id = dict(enumerate(tmp_y, start=2))
    ner_idx = {v: k for k, v in rev_ner_id.items()}

    nb_classes = len(tmp_y)
    print(nb_classes)

    M = len(vocabulary) + 2
    #print(M)
    N = 100
    matrix = np.random.rand(M, N)

    for word in vocabulary:
        if word in dic.keys():
            matrix[word_idx[word]] = dic[word]

    #dict_vect = DictVectorizer(sparse=False)

    dev_dict = conll_dict.transform(dev_sentences)
    X_dev, Y_dev = build_sequences(dev_dict)
    X_dev_i = to_index(X_dev, word_idx)
    Y_dev_i = to_index(Y_dev, ner_idx)
    X_dev_pad = pad_sequences(X_dev_i)
    Y_dev_pad = pad_sequences(Y_dev_i)
    Y_dev_cat = to_categorical(Y_dev_pad, num_classes=nb_classes + 2)

    #Ta X och gör om till index-värden
    X_idx = to_index(X, word_idx)

    Y_idx = to_index(Y, ner_idx)


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
    model.layers[0].set_weights([matrix])
    model.layers[0].trainable = False

    #model.add(SimpleRNN(100, return_sequences=True))
    model.add(layers.Dropout(0.25))
    model.add(Bidirectional(LSTM(100, recurrent_dropout=0.25, return_sequences=True)))
    model.add(layers.Dropout(0.25))
    model.add(Bidirectional(LSTM(100, recurrent_dropout=0.25, return_sequences=True)))
    model.add(layers.Dropout(0.25))
    #model.add(Dense(512, activation='relu'))
    #model.add(layers.Dropout(0.25))
    model.add(Dense(nb_classes + 2, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                optimizer='rmsprop',
                metrics=['acc'])
    
    model.summary()

    model.fit(padded_x, y_train, epochs=10, batch_size=128, validation_data=(X_dev_pad, Y_dev_cat))


    X_test, Y_test = build_sequences(test_dict)

    X_test_idx = to_index(X_test, word_idx)
    Y_test_idx = to_index(Y_test, ner_idx)

    #print('X[0] test idx', X_test_idx[0])
    #print('Y[0] test idx', Y_test_idx[0])

    X_test_padded = pad_sequences(X_test_idx)
    Y_test_padded = pad_sequences(Y_test_idx)

    #print('X[0] test idx passed', X_test_padded[0])
    #print('Y[0] test idx padded', Y_test_padded[0])

    Y_test_padded_vectorized = to_categorical(Y_test_padded, num_classes=nb_classes + 2)

    #print('Y[0] test idx padded vectorized', Y_test_padded_vectorized[0])
    print(X_test_padded.shape)
    print(Y_test_padded_vectorized.shape)

    test_loss, test_acc = model.evaluate(X_test_padded, Y_test_padded_vectorized)
    #print('Loss:', test_loss)
    #print('Accuracy:', test_acc)

    print('X_test', X_test[0])
    print('X_test_padded', X_test_padded[0])
    corpus_ner_predictions = model.predict(X_test_padded)
    print('Y_test', Y_test[0])
    print('Y_test_padded', Y_test_padded[0])
    print('predictions', corpus_ner_predictions[0])

    ner_pred_num = []
    for sent_nbr, sent_ner_predictions in enumerate(corpus_ner_predictions):
        ner_pred_num += [sent_ner_predictions[-len(X_test[sent_nbr]):]]
    print(ner_pred_num[:2])

    ner_pred = []
    for sentence in ner_pred_num:
        ner_pred_idx = list(map(np.argmax, sentence))
        ner_pred_cat = list(map(rev_ner_id.get, ner_pred_idx))
        ner_pred += [ner_pred_cat]
    
    result = open("result_ltsm_no_rnn.txt", "w+")
    for id_s, sentence in enumerate(X_test):
        for id_w, word in enumerate(sentence):
            result.write(f"{word} {Y_test[id_s][id_w]} {ner_pred[id_s][id_w]}\n")
    result.close()