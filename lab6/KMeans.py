from sklearn.cluster import KMeans
import numpy as np
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

    km = KMeans(n_clusters=10, init='random',n_init=45, max_iter=300, tol=1e-1,random_state=0).fit(train_feature)
    y_km = km.predict(test_feature)

    print("Classification report:\n%s\n" % (metrics.classification_report(test_labels, y_km)))

    print("Confusion matrix:\n%s" % metrics.confusion_matrix(test_labels, y_km))

    print('Completeness: {}'.format( completeness_score(test_labels, y_km) ))

    print('Homogeneity: {}'.format( homogeneity_score(test_labels, y_km)))