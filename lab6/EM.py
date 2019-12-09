from collections import Counter
from scipy.stats import multivariate_normal

import numpy as np
import math 

class EM:
    def __init__(self):
        self.prior = {}
        self.means = []
        self.variances = []

    def em(self, features, classes):
        theta = []
        # (pi, mu, std) for pi, mu, std in (1/len(classes), np.random.randn(), 1)

        for c in classes:
            mu = np.random.rand(1, 64) * 16
            std = np.random.rand(1, 64) * 16
            theta.append((1/len(classes), list(map(int, mu[0])), list(std[0])))

        """
        E-step
        """
        
        r_ik = []
        for i in range(len(features)):
            tmp_ri = []
            denominatior = sum(theta[k][0] * self.P(features[i], theta[k]) for k in classes)
            for k in classes:
                tmp_ri.append( (theta[k][0] * self.P(features[i], theta[k])) / denominatior )
            r_ik.append(tmp_ri)
        print(r_ik)

    def P(self, xi, theta_k):
        prod = 1.0
        for j in range(64):
            print('for values: mu: {}, std: {}, xi: {}'.format(theta_k[1][j], theta_k[2][j], xi[j]))
            print(1 / np.sqrt(2*np.pi*(theta_k[2][j]**2)))
            print(np.exp( - (((xi[j] - theta_k[1][j])**2) / (2*(theta_k[2][j]**2))) ))
            print(prod)
            print(20 * '-')
            prod *= (1 / np.sqrt(2*np.pi*(theta_k[2][j]**2)) * np.exp( - (((xi[j] - theta_k[1][j])**2) / (2*(theta_k[2][j]**2))) ))