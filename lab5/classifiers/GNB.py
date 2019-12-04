"""
Gaussian Naive Bayesian Classifier
"""

from collections import Counter
import numpy as np

class GNB:
    def __init__(self):
        self.prior = {}
        self.index_posterior = {}   # From index feature[0] -> feature[63]
        self.mean = 0
        self.variance = 0

    def fit(self, features, labels):
        classes = list(set(labels))
        counter = Counter(labels)
        total = len(labels)
        self.prior = {c: counter[c]/total for c in classes}

        self.mean, self.variance = self.mean_var(np.array(features), np.array(labels), len(classes))

        index_length = len(features[0])
        entire_list = [x for y in features for x in y]
        uniq_vals = list(set(entire_list))
        self.index_posterior = {i : {c : {v : 0 for v in uniq_vals} for c in classes} for i in range(index_length)}

        for i in range(len(labels)):
            in_class = labels[i] # Column

            for j in range(len(features[i])): # 0 - 63 (What matrix)
                val = features[i][j] # Row
                self.index_posterior[j][in_class][val] += 1

        for matrix in self.index_posterior:
            for c in self.index_posterior[matrix]:
                for v in self.index_posterior[matrix][c]:
                    self.index_posterior[matrix][c][v] = self.index_posterior[matrix][c][v] / counter[c]

    def predict(self, features):
        predict = []
        for f in features:
            prob_per_class = []
            for c in self.prior:
                val_vector = []
                for i in range(len(f)): # i vilken matris
                    x_i = self.index_posterior[i][c][f[i]]
                    v = self.variance[c][i] + 0.1
                    m = self.mean[c][i]
                    gauss = (1 / np.sqrt(2*np.pi * v**2)) ** (-(((x_i - m)**2) / (2*(v**2))))
                    val_vector.append(gauss)
                val_vector.append(self.prior[c])
                mul_sum = np.prod(val_vector)
                prob_per_class.append(mul_sum)
            max_prob = max(prob_per_class)
            max_class = prob_per_class.index(max_prob)
            predict.append(max_class)
        return predict
    
    def mean_var(self, X, y, nbr_classes):
        n_features = X.shape[1]
        m = np.ones((nbr_classes, n_features))
        v = np.ones((nbr_classes, n_features))
        n_0 = np.bincount(y)[np.nonzero(np.bincount(y))[0]][0]
        x0 = np.ones((n_0, n_features))
        x1 = np.ones((X.shape[0] - n_0, n_features))
        
        k = 0
        for i in range(0, X.shape[0]):
            if y[i] == 0:
                x0[k] = X[i]
                k = k + 1
        k = 0
        for i in range(0, X.shape[0]):
            if y[i] == 1:
                x1[k] = X[i]
                k = k + 1
            
        for j in range(0, n_features):
            m[0][j] = np.mean(x0.T[j])
            v[0][j] = np.var(x0.T[j])*(n_0/(n_0 - 1))
            m[1][j] = np.mean(x1.T[j])
            v[1][j] = np.var(x1.T[j])*((X.shape[0]-n_0)/((X.shape[0]
                                                        - n_0) - 1))
        return m, v # mean and variance 