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
            theta.append( [1/len(classes), list(map(int, mu[0])), list(std[0])] )
        
        # for 50 iterations: 

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

        """
        M-step
        """

        r_k = []
        new_mu_k = []
        new_sigma_k = []
        for k in classes:
            r_k.append(sum(r_ik[i][k] for i in range(len(features))))
            new_mu_k.append(sum(r_ik[i][k] * features[i] for i in range(len(features))))
            new_sigma_k.append(sum(r_ik[i][k] * (np.array(features[i]) * np.transpose(np.array(features[i]))) for i in range(len(features))))

        new_pi = [i/len(features) for i in r_k]
        for k in classes:
            new_mu_k[k] = [mu/r_k[k] for mu in new_mu_k[k]]
            new_sigma_k[k] = [(sigma/r_k[k]) - new_mu_k[k]*np.transpose(new_mu_k[k]) for sigma in new_sigma_k[k]]

        # Count the diff 
        # diff_mu = [np.abs(sum(new_mu_k[k]) - sum(theta[k][1])) for k in classes]

    def P(self, xi, theta_k):
        prod = 1.0
        for j in range(64):
            prod *= (1 / np.sqrt(2*np.pi*(theta_k[2][j]**2)) * np.exp( - (((xi[j] - theta_k[1][j])**2) / (2*(theta_k[2][j]**2))) )) 
        return prod