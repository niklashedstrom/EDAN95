"""
Gaussian Naive Bayesian Classifier
"""

from collections import Counter
import numpy as np
import math

class GNB:
    def __init__(self):
        self.prior = {}
        self.index_posterior = {}
        self.means = []
        self.variances = []

    def fit(self, features, labels):
        classes = list(set(labels))
        freq_per_class = Counter(labels)
        self.prior = {c : freq_per_class[c]/len(labels) for c in classes}

        index_length = len(features[0])
        entire_list = [x for y in features for x in y]
        attributes = list(set(entire_list))

        self.index_posterior = {i : {c : {v : 0 for v in attributes} for c in classes} for i in range(index_length)}

        for i in range(len(labels)):
            in_class = labels[i] # Column

            for j in range(len(features[i])): # 0 - 63 (What matrix)
                val = features[i][j] # Row
                self.index_posterior[j][in_class][val] += 1
        
        epsilon = 0.16
        for c in self.prior:
            X = np.array([np.array(features[i]) for i in range(len(features)) if labels[i] == c])
            X = np.transpose(X)
            tmp_means = []
            tmp_vars = []
            for x in X:
                tmp_means.append(np.mean(x))
                tmp_vars.append(np.var(x) + epsilon)
            self.means.append(tmp_means)
            self.variances.append(tmp_vars)

        for matrix in self.index_posterior:
            for c in self.index_posterior[matrix]:
                for v in self.index_posterior[matrix][c]:
                    self.index_posterior[matrix][c][v] = self.index_posterior[matrix][c][v] / freq_per_class[c]

    def predict(self, features):
        predict = []
        for f in features:
            prob_per_class = []
            for c in self.prior:
                val_vector = []
                for i in range(len(f)): # i vilken matris
                    x_i = self.index_posterior[i][c][f[i]]
                    v = self.variances[c][i] 
                    m = self.means[c][i]
                    gauss = 1.0 / np.sqrt(2*np.pi*v) * np.exp(-1.0 * math.pow(m - f[i], 2) / v)
                    val_vector.append(gauss)
                val_vector.append(self.prior[c])
                mul_sum = np.prod(val_vector)
                prob_per_class.append(mul_sum)
            summ = sum(prob_per_class)
            if summ > 0.0:
                prob_per_class = prob_per_class / summ
                max_class = np.argmax(prob_per_class)
                predict.append(max_class)
            else:
                predict.append(10)
        return predict