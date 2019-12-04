"""
Naive Bayesian Classifier
"""

from collections import Counter
import numpy as np

class NBC:
    def __init__(self):
        self.prior = {}
        self.index_posterior = {}   # From index feature[0] -> feature[63]

    def fit(self, features, labels):
        classes = list(set(labels))
        counter = Counter(labels)
        total = len(labels)
        self.prior = {c: counter[c]/total for c in classes}

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
                    val_vector.append(self.index_posterior[i][c][f[i]])
                val_vector.append(self.prior[c])
                mul_sum = np.prod(val_vector)
                prob_per_class.append(mul_sum)
            max_prob = max(prob_per_class)
            max_class = prob_per_class.index(max_prob)
            predict.append(max_class)
        return predict