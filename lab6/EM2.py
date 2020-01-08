import numpy as np
import math
from collections import Counter

class EM2:
    def __init__(self):
        self.prob = []
        self.mean = []
        self.var = []
        self.labels = {}
    
    def EM(self, features, targets, classes):
        attr_len = len(features[0])
        N = len(features)

        self.prob = [1/len(classes) for k in classes]
        self.mean = [np.random.rand(1, attr_len)[0] for k in classes]
        self.var = [np.random.rand(1, attr_len)[0] + 0.1 for k in classes]

        means = [[] for k in classes]
        actual_means = []
        for i in range(N):
            means[targets[i]].append(features[i])
        for means_per_class in means:
            tot_val = np.zeros(attr_len)
            for feat in means_per_class:
                for f in range(len(feat)):
                    tot_val[f] += feat[f]
            actual_means.append([tot_val[i] / len(means_per_class) for i in range(len(tot_val))])

        self.mean = actual_means

        variances = [[] for k in classes]
        actual_variances = []

        for i in range(N):
            variances[targets[i]].append(features[i])
        c = 0
        for var_per_class in variances:
            attr_vec = np.zeros(attr_len)
            for v in var_per_class:
                for iv in range(len(v)):
                    attr_vec[iv] += math.pow(v[iv] - actual_means[c][iv], 2)
            c += 1
            actual_variances.append([attr_vec[i] / len(var_per_class) for i in range(len(attr_vec))])
        self.var = actual_variances

        for iterations in range(30):
            print(iterations)
            # print(self.prob[0])
            # print(self.mean[0])
            # print(self.var[0])
            print('-' * 20)
            """
            E-step
            """
            # denom = sum([self.prob[c] * np.prod([self.P(features[i][j], self.mean[c][j], self.var[c][j]) 
                                # for j in range(attr_len)]) for c in classes for i in range(N)])
            # print(denom)

            r_ik = []
            for i in range(N):
                prod_class = []
                for c in classes:
                    prod = []
                    for j in range(attr_len):
                        prod.append(self.P(features[i][j], self.mean[c][j], self.var[c][j]))
                    prod_class.append(np.prod(prod))
                norm_prod_class = [n / sum(prod_class) for n in prod_class]
                # print(norm_prod_class)
                denominator = sum([self.prob[k] * norm_prod_class[k] for k in classes])
                nominators = [(self.prob[k] * norm_prod_class[k]) / denominator for k in range(len(norm_prod_class))]
                # self.prob[c] * norm_prod_class[c]
                # print(denominator)
                # print(nominators)
                r_ik.append(nominators)
            # print(prod_class)
            # print(r_ik)

            # r_ik = [[(self.prob[k] * np.prod([self.P(features[i][j], self.mean[k][j], self.var[k][j]) 
                                # for j in range(attr_len)]) / denom) for k in classes] for i in range(N)]
            # sum([self.prob[c] * np.prod([self.P(features[i][j], self.mean[c][j], self.var[c][j]) for j in range(attr_len)]) for c in classes]))
            """
            M-step
            """
            old_mean = self.mean
            old_var = self.var

            r_k = [sum(r_ik[i][k] for i in range(N)) for k in classes]
            self.prob = [r_k[k] / N for k in classes]
            prob_sum = sum(self.prob)
            self.prob = [p / prob_sum for p in self.prob]
            # print(self.prob)
            self.mean = [[sum([r_ik[i][k] * features[i][j] for i in range(N)])/r_k[k] for j in range(attr_len)] for k in classes]
            self.var = [[sum([r_ik[i][k] * features[i][j] * features[i][j] 
                        for i in range(N)])/r_k[k] - np.multiply(self.mean[k], self.mean[k])[0] + 0.01
                        for j in range(attr_len)] for k in classes]
            
            # print("----OLD MEAN: {}".format(old_mean))
            # print("----NEW MEAN: {}".format(self.mean))
            diff_mean = [np.subtract(old_mean[i], self.mean[i]) for i in range(len(self.mean))]
            tot = 0
            for d in diff_mean:
                tot += sum(d)
            # print(tot)
            # print("----DIFF MEAN: {}".format(diff_mean[0]))
        
        # Label clusters
        cluster_idx = {n : [] for n in range(len(classes))}

        for i, x in enumerate(r_ik):
            print(i)
            print(x)
            cluster = np.argmax(x)
            print(cluster)
            cluster_idx[cluster].append(targets[i])
        print(cluster_idx)
        print(len(targets))
        for c in cluster_idx.keys():
            cnt = Counter(cluster_idx[c])
            print(cnt)
            mc = cnt.most_common(1)

            if not mc:
                self.labels[c] = 10
            else:
                self.labels[c] = mc[0][0]

    def P(self, f, m, v):
        v += 0.1
        gauss = 1.0 / np.sqrt(2*np.pi*v) * np.exp(-1.0 * math.pow(m - f, 2) / (2 * v))
        # if gauss < 0.001:
        #     return 0.001
        # else:
        #     return gauss
        return gauss
    
    def predict(self, features, classes):
        classes = list(classes)
        predicted = []
        for f in features:
            # probs = [np.prod([self.P(f[j], self.mean[k][j], self.var[k][j]) for j in range(len(f))]) for k in classes]
            prob = []
            for c in classes:
                gauss = 1
                for i in range(len(f)):
                    # print(gauss)
                    gauss *= self.P(f[i], self.mean[c][i], self.var[c][i])
                prob.append(gauss)
            prob_norm = [p / sum(prob) for p in prob]
            # print(prob_norm)
            predicted.append(self.labels[np.argmax(prob_norm)])
            # print(probs)
            # predicted.append(np.argmax(probs))
        return predicted
