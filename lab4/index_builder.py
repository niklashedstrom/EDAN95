from datasets import load_conll2003_en
from conll_dictorizer import CoNLLDictorizer
from dictionizer import dictionize

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

    vocabulary = vocabulary(train_dict, dictionize('glove.6B.100d.txt'))
    print(vocabulary)
    print(len(vocabulary))
