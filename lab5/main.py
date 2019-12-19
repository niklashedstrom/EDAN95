from classifiers.NCC import NCC
from classifiers.NBC import NBC
from classifiers.GNB import GNB

from sklearn.naive_bayes import GaussianNB

from sklearn import metrics
from sklearn import datasets

import MNIST

import numpy as np

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

def modify_target(target):
    modified = []
    for t in target:
        if t < 3:
            modified.append(0)
        elif t < 6:
            modified.append(1)
        else:
            modified.append(2)
    return np.array(modified)

def main():
    # SciKitLearn digits (2.1)
    digits = datasets.load_digits()

    split = int(0.7 * digits.data.shape[0])
    train_feature = digits.data[:split]
    train_label = digits.target[:split]
    test_feature = digits.data[split:]
    test_labels = digits.target[split:]

    # SciKitLearn digits summarised (2.2)
    modified_digits_data = modify_data(digits.data)
    modified_digits_labels = digits.target

    train_feature_mod = modified_digits_data[:split]
    train_label_mod = modified_digits_labels[:split]
    test_feature_mod = modified_digits_data[split:]
    test_label_mod = modified_digits_labels[split:]


    # MNIST_Light (2.3)
    mnist = MNIST.MNISTData('MNIST_Light/*/*.png')
    train_features_mnist, test_features_mnist, train_labels_mnist, test_labels_mnist = mnist.get_data()

    # gnb = GNB()
    # gnb.fit(train_feature, train_label)
    # y_pred = gnb.predict(test_feature)
    # gnb = GaussianNB()
    # gnb.fit(train_feature, train_label)
    # y_pred = gnb.predict(test_feature)
# 
    # print("Classification report SKLearn GNB:\n%s\n"
    #   % (metrics.classification_report(test_labels, y_pred)))
    # print("Confusion matrix SKLearn GNB:\n%s" % metrics.confusion_matrix(test_labels, y_pred))

    ncc = NCC()
    ncc.fit(train_feature, train_label)
    y_pred = ncc.predict(test_feature)

    print("Classification report NCC (dataset 2.1):\n%s\n"
      % (metrics.classification_report(test_labels, y_pred)))
    print("Confusion matrix NCC:\n%s" % metrics.confusion_matrix(test_labels, y_pred))

    ncc2 = NCC()
    ncc2.fit(train_feature_mod, train_label_mod)
    y_pred_mod = ncc2.predict(test_feature_mod)

    print("Classification report NCC (dataset 2.2):\n%s\n"
      % (metrics.classification_report(test_label_mod, y_pred_mod)))
    print("Confusion matrix NCC:\n%s" % metrics.confusion_matrix(test_label_mod, y_pred_mod))

    ncc3 = NCC()
    ncc3.fit(train_features_mnist, train_labels_mnist)
    y_pred_mnist = ncc3.predict(test_features_mnist)

    print("Classification report NCC (dataset 2.3):\n%s\n"
      % (metrics.classification_report(test_labels_mnist, y_pred_mnist)))
    print("Confusion matrix NCC:\n%s" % metrics.confusion_matrix(test_labels_mnist, y_pred_mnist))

if __name__ == "__main__":
    main()