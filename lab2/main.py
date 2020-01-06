import ToyData as td
import ID3
import new_ID3
import ID3PO

import numpy as np
from sklearn import tree, metrics, datasets

def modify_data(data):
    modified = []
    for d in data:
        tmp = []
        for value in d:
            if value < 5:
                tmp.append(0)
            elif value < 10:
                tmp.append(1)
            else:
                tmp.append(2)
        modified.append(np.array(tmp))
    return np.array(modified)


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
        attributes[i] = list(range(3))

    # id3 = new_ID3.ID3DecisionTreeClassifier()
    id3 = ID3PO.ID3DecisionTreeClassifier()
    
    # for i in range(len(test_feature)):
    #     print(test_feature[i])
    #     print(test_label[i])
    modified_digits_data = modify_data(digits.data)
    modified_digits_labels = digits.target

    train_feature_mod = modified_digits_data[:split].tolist()
    train_label_mod = modified_digits_labels[:split].tolist()
    test_feature_mod = modified_digits_data[split:]
    test_label_mod = modified_digits_labels[split:]

    print(attributes)    

    # myTree = id3.fit(data, target, attributes, classes)
    # myTree = id3.fit(train_feature, train_label, attributes, classes)
    myTree = id3.fit(train_feature_mod, train_label_mod, attributes, classes)
    # print(myTree)
    plot = id3.make_dot_data()
    plot.render("testTree")
    predicted = id3.predict(test_feature_mod, myTree, attributes)
    print(predicted)

    print("Classification report:\n%s\n"
      % (metrics.classification_report(test_label_mod, predicted)))
    print("Confusion matrix:\n%s" % metrics.confusion_matrix(test_label_mod, predicted))



if __name__ == "__main__": main()