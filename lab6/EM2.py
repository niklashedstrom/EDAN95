import numpy as np
import math

class EM2:
    def __init__(self):
        self.prob = []
        self.mean = []
        self.var = []
    
    def EM(self, features, classes):
        attr_len = len(features[0])
        N = len(features)

        self.prob = [1/len(classes) for k in classes]
        self.mean = [np.random.rand(1, attr_len)[0] for k in classes]
        self.var = [np.random.rand(1, attr_len)[0] + 0.1 for k in classes]
        
        for iterations in range(1):
            print(iterations)
            print(self.prob[1])
            print(self.mean[1])
            print(self.var[1])
            print('-' * 20)
            """
            E-step
            """
            r_ik = [[(self.prob[k] * np.prod([self.P(features[i][j], self.mean[k][j], self.var[k][j]) 
                                for j in range(attr_len)]) / sum([self.prob[c] * np.prod([self.P(features[i][j], self.mean[c][j], self.var[c][j]) 
                                for j in range(attr_len)]) for c in classes])) for k in classes] for i in range(N)]
            """
            M-step
            """
            r_k = [sum(r_ik[i][k] for i in range(N)) for k in classes]
            self.prob = [r_k[k] / N for k in classes]
            self.mean = [[sum([r_ik[i][k] * features[i][j] for i in range(N)])/r_k[k] for j in range(attr_len)] for k in classes]
            self.var = [[sum([r_ik[i][k] * features[i][j] * features[i][j] 
                        for i in range(N)])/r_k[k] - np.multiply(self.mean[k], self.mean[k])[0] + 0.1
                        for j in range(attr_len)] for k in classes]

    def P(self, f, m, v):
        gauss = 1.0 / np.sqrt(2*np.pi*v) * np.exp(-1.0 * math.pow(m - f, 2) / v)
        if gauss == 0.0:
            return 0.001
        else:
            return gauss
    
    def predict(self, features, classes):
        predicted = []
        for f in features:
            probs = [np.prod([self.P(f[j], self.mean[k][j], self.var[k][j]) for j in range(len(f))]) for k in classes]
            print(probs)
            predicted.append(np.argmax(probs))
        return predicted
