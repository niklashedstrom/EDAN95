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

    classes = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}

    normalised_train_feat = []
    for f in train_feature:
        normalised_train_feat.append([i/16 for i in f])

    em.em(normalised_train_feat, classes)
    

    # print("Classification report:\n%s\n"
    #   % (metrics.classification_report(test_labels, y_pred)))
    # print("Confusion matrix:\n%s" % metrics.confusion_matrix(test_labels, y_pred))
