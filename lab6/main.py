from EM import EM

from sklearn import metrics
from sklearn import datasets


if __name__ == "__main__":
    
    digits = datasets.load_digits()

    split = int(0.7 * digits.data.shape[0])
    train_feature = digits.data[:split]
    train_label = digits.target[:split]
    test_feature = digits.data[split:]
    test_labels = digits.target[split:]

    em = EM()
    em.fit(train_feature, train_label)
    

    # print("Classification report:\n%s\n"
    #   % (metrics.classification_report(test_labels, y_pred)))
    # print("Confusion matrix:\n%s" % metrics.confusion_matrix(test_labels, y_pred))
