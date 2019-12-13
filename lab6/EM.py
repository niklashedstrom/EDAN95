from collections import Counter
from scipy.stats import multivariate_normal

import numpy as np
import math

class EM:
    def __init__(self):
        self.prior = {}
        self.pi = []
        self.means = []
        self.std = []

    def em(self, features, classes):
        theta = []
        # (pi, mu, std) for pi, mu, std in (1/len(classes), np.random.randn(), 1)

        for c in classes:
            mu = np.random.rand(1, 64)
            std = np.random.rand(1, 64) + 0.1
            # theta.append( [1/len(classes), list(map(int, mu[0])), list(std[0])] )
            self.pi.append(1/len(classes))
            self.means.append(list(mu[0]))
            self.std.append(list(std[0]))
        
        # for 50 iterations: 
        for iteration in range(50):
            print(iteration)
            # for k in classes:
                # print(k)
                # theta[k] = [self.pi[k], self.means[k], self.std[k]]

            """
            E-step
            """

            print(self.pi[0])
            print(self.means[0])
            print(self.std[0])

            r_ik = []
            for i in range(len(features)):
                tmp_ri = []
                denominatior = sum(self.pi[k] * self.P(features[i], self.means[k], self.std[k]) for k in classes)
                # denom = self.logSumExp(features[i])
                for k in classes:
                    tmp_ri.append( self.pi[k] * self.P(features[i], self.means[k], self.std[k]) / denominatior )
                r_ik.append(tmp_ri)

            """
            M-step
            """

            r_k = []
            new_mu_k = []
            new_sigma_k = []
            for k in classes:
                r_k.append(sum(r_ik[i][k] for i in range(len(features))))

                big_mu = []
                for i in range(len(features)):
                    sm = []
                    for n in range(len(features[i])):
                        sm.append(r_ik[i][k] * features[i][n])
                    big_mu.append(sm)
                
                tmp_attr = []
                for attr in range(64):
                    tmp_sum = 0
                    for i in big_mu:
                        tmp_sum += i[attr]
                    tmp_attr.append(tmp_sum)

                new_mu_k.append(tmp_attr)
                
                big_std = []
                for i in range(len(features)):
                    smol = []
                    for n in range(len(features[i])):
                        smol.append(r_ik[i][k] * np.multiply(features[i][n], features[i][n])) 
                    big_std.append(smol)

                tmp_attr = []
                for attr in range(64):
                    tmp_sum = 0
                    for i in big_std:
                        tmp_sum += i[attr]
                    tmp_attr.append(tmp_sum)
                
                new_sigma_k.append(tmp_attr)

            new_pi = [i/len(features) for i in r_k]
            # print(new_sigma_k[0][0])
            # print(r_k)
            for k in classes:
                new_mu_k[k] = [mu/r_k[k] for mu in new_mu_k[k]]
                new_sigma_k[k] = [(sigma/r_k[k]) - np.multiply(new_mu_k[k],new_mu_k[k]) + 0.01 for sigma in new_sigma_k[k]]
            # Count the diff 
            # diff_mu = [np.abs(sum(new_mu_k[k]) - sum(theta[k][1])) for k in classes]

            """
            Set new values for mu and sigma
            """
            self.pi = new_pi
            self.means = new_mu_k
            # self.std = new_sigma_k
            self.std = []
            for k in classes:
                diag = []
                for j in range(64):
                    diag.append(new_sigma_k[k][j][j])
                self.std.append(diag)

    def P(self, xi, mu_k, std_k):
        prod = 1.0
        for j in range(64):
            # sqrt = np.sqrt(2*np.pi*(std_k[j]**2))
            # if sqrt == 0.0:
            #     print('sqrt: {}'.format( np.sqrt(2*np.pi*(std_k[j]**2))) )
            # two = (2*(std_k[j]**2))
            # if two == 0.0:
            #     print('2*theta: {}'.format( (2*(std_k[j]**2))) )
            prod *= (1 / np.sqrt(2*np.pi*(std_k[j]**2)) * np.exp( - (((xi[j] - mu_k[j])**2) / (2*(std_k[j]**2))) )) 
        # print(prod)
        return prod

    def logSumExp(self, ns):
        max = np.max(ns)
        ds = ns - max
        sumOfExp = np.exp(ds).sum()
        return max + np.log(sumOfExp)