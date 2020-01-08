from EM import EM
from EM2 import EM2

from sklearn import metrics
from sklearn import datasets
from sklearn.metrics import completeness_score, homogeneity_score

if __name__ == "__main__":
    
    digits = datasets.load_digits()

    split = int(0.7 * digits.data.shape[0])
    train_feature = digits.data[:split]
    train_label = digits.target[:split]
    test_feature = digits.data[split:]
    test_labels = digits.target[split:]

    em = EM2()

    classes = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}

    normalised_train_feat = []
    for f in train_feature:
        normalised_train_feat.append([i/16 for i in f])

    em.EM(normalised_train_feat, train_label, classes)

    norm_test_feat = []
    for f in test_feature:
        norm_test_feat.append([i/16 for i in f])

    y_pred = em.predict(norm_test_feat, classes)

    print(y_pred)

    print("Classification report:\n%s\n"
      % (metrics.classification_report(test_labels, y_pred)))
    print("Confusion matrix:\n%s" % metrics.confusion_matrix(test_labels, y_pred))

    print('Completeness: {}'.format( completeness_score(test_labels, y_pred) ))

    print('Homogeneity: {}'.format( homogeneity_score(test_labels, y_pred)))