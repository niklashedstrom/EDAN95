"""
Nearest Centroid Classifier
"""

import numpy as np

class NCC:
    def __init__(self):
        self.centroids = {}

    def fit(self, features, labels):
        classes = list(set(labels))
        dic = {key: [] for key in classes}
        self.centroids = {key: [] for key in classes}

        for i in range(len(features)):
            dic[labels[i]].append(features[i])
        
        for c in classes:
            for i in range(len(dic[c][0])):
                sum = 0
                for feature in dic[c]:
                    sum += feature[i]
                self.centroids[c].append(sum/len(dic[c]))

    def predict(self, features):
        predict = []
        for f in features:
            dist_list = [self.distance_to_centroid(f, self.centroids[i]) for i in range(len(self.centroids))]
            predict.append(np.argmin(dist_list))
        return predict

    def distance_to_centroid(self, feature, centroid):
        return np.linalg.norm(feature - centroid)