import ToyData as td
import ID3
import new_ID3
import id3_working

import numpy as np
from sklearn import tree, metrics, datasets


def main():

    attributes, classes, data, target, data2, target2 = td.ToyData().get_data()

    digits = datasets.load_digits()

    split = int(0.7 * digits.data.shape[0])
    train_feature = digits.data[:split].tolist()
    train_label = digits.target[:split].tolist()
    test_feature = digits.data[split:]
    test_label = digits.target[split:]

    attributes = {}

    for i in range(64):
        attributes[i] = list(range(17))

    # id3 = new_ID3.ID3DecisionTreeClassifier()
    id3 = id3_working.ID3DecisionTreeClassifier()
    
    # for i in range(len(test_feature)):
    #     print(test_feature[i])
    #     print(test_label[i])
        

    # myTree = id3.fit(data, target, attributes, classes)
    myTree = id3.fit(train_feature, train_label, attributes, classes)
    print(myTree)
    plot = id3.make_dot_data()
    plot.render("testTree")
    predicted = id3.predict(test_feature, myTree, attributes)
    print(predicted)


if __name__ == "__main__": main()