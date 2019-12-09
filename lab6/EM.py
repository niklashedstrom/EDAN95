from collections import Counter

import numpy as np
import math

class EM:
    def __init__(self):
        self.prior = {}
        self.means = []
        self.variances = []
        # self.index_posterior = {}



    def em(self, features, classes):
        pass

    # def fit(self, features, labels):
    #     classes = list(set(labels))
    #     counter = Counter(labels)
    #     self.prior = {c : counter[c]/len(labels) for c in counter}

    #     attributes = list(set([x for y in features for x in y]))

    #     for c in self.prior:
    #         X = np.array([np.array(features[i]) for i in range(len(features)) if labels[i] == c])
    #         X = np.transpose(X)
    #         tmp_means = []
    #         tmp_vars = []
    #         for x in X:
    #             tmp_means.append(np.mean(x))
    #             tmp_vars.append(np.var(x) + 0.01)
    #         self.means.append(tmp_means)
    #         self.variances.append(tmp_vars)

    #     self.index_posterior = {i : {c : {v : 0 for v in attributes} for c in classes} for i in range(len(features[0]))}

    #     for i in range(len(labels)):
    #         in_class = labels[i] # Column

    #         for j in range(len(features[i])): # 0 - 63 (What matrix)
    #             val = features[i][j] # Row
    #             self.index_posterior[j][in_class][val] += 1

    #     for matrix in self.index_posterior:
    #         for c in self.index_posterior[matrix]:
    #             for v in self.index_posterior[matrix][c]:
    #                 self.index_posterior[matrix][c][v] = self.index_posterior[matrix][c][v] / counter[c]

    #     # E-step
    #     t = 1
    #     r_ik = []
    #     P_xi_omegak_list = []
    #     for f in features:
    #         for k in classes:
    #             P_xi_omegak = []
    #             for j in range(len(features[i])):
    #                 x = self.index_posterior[j][k][f[j]]
    #                 v = self.variances[k][j]
    #                 m = self.means[k][j]
    #                 gauss = 1.0 / np.sqrt(2*np.pi*v) * np.exp(-1.0 * math.pow(m - x, 2) / v)
    #                 # P_xi_omegak.append(1 / np.sqrt(2 * np.pi * v**2) * np.exp( -((x - m)**2 / 2*(v**2) )))
    #                 P_xi_omegak.append(gauss)
    #             P_xi_omegak_list.append(np.prod(P_xi_omegak))
    #     print(P_xi_omegak_list[0])
    #     for k1 in classes:
    #         tmp_class = []
    #         for k2 in classes:
    #             tmp_class.append( (self.prior[k1] * P_xi_omegak_list[k1]) / (self.prior[k2] * sum(self.variances[k2]) * P_xi_omegak_list[k2]) )
    #         r_ik.append(tmp_class)

    #     print(r_ik)
            
            

    # def predict(self, features):
    #     pass


    # def P(self, x_i, omega_k, attributes):
        # p = 1
        # for j in attributes:
            # 
# 