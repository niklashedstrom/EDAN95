from classifiers.NCC import NCC
from classifiers.NBC import NBC
from classifiers.GNB import GNB

from sklearn import metrics
from sklearn import datasets

def main():
    digits = datasets.load_digits()

    split = int(0.7 * digits.data.shape[0])
    train_feature = digits.data[:split].tolist()
    train_label = digits.target[:split].tolist()
    test_feature = digits.data[split:]
    test_labels = digits.target[split:]

    ncc = NCC()
    ncc.fit(train_feature, train_label)
    y_pred = ncc.predict(test_feature)

    print("Classification report NCC:\n%s\n"
      % (metrics.classification_report(test_labels, y_pred)))
    print("Confusion matrix NCC:\n%s" % metrics.confusion_matrix(test_labels, y_pred))

if __name__ == "__main__":
    main()