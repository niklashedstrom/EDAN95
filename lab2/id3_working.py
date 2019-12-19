from collections import Counter
from graphviz import Digraph
import numpy as np



class ID3DecisionTreeClassifier :

    def __init__(self, minSamplesLeaf = 1, minSamplesSplit = 2) :

        self.__nodeCounter = 0

        # the graph to visualise the tree
        self.__dot = Digraph(comment='The Decision Tree')

        # suggested attributes of the classifier to handle training parameters
        self.__minSamplesLeaf = minSamplesLeaf
        self.__minSamplesSplit = minSamplesSplit


        # Create a new node in the tree with the suggested attributes for the visualisation.
        # It can later be added to the graph with the respective function
    def new_ID3_node(self):
        node = {'id': self.__nodeCounter, 'label': None, 'attribute': None, 'entropy': None, 'samples': None,
                        'classCounts': None, 'nodes': None}

        self.__nodeCounter += 1
        return node

        # adds the node into the graph for visualisation (creates a dot-node)
    def add_node_to_graph(self, node, parentid=-1):
        nodeString = ''
        for k in node:
            if ((node[k] != None) and (k != 'nodes')):
                nodeString += "\n" + str(k) + ": " + str(node[k])

        self.__dot.node(str(node['id']), label=nodeString)
        if (parentid != -1):
            self.__dot.edge(str(parentid), str(node['id']))


    # make the visualisation available
    def make_dot_data(self) :
        return self.__dot


    # For you to fill in; Suggested function to find the best attribute to split with, given the set of
    # remaining attributes, the currently evaluated data and target.
    def find_split_attr(self, data, target, attributes, classes):

        # Infogain
        best_gain = 0
        best_attribute = ''
        best_index = 0
        current_entropy = self.entropy(target)

        for index, a in enumerate(attributes):
            gain = self.infogain(data,target,a,index,current_entropy, attributes)
            if gain > best_gain:
                best_gain = gain
                best_attribute = a
                best_index = index
        return best_attribute, best_index


    # the entry point for the recursive ID3-algorithm, you need to fill in the calls to your recursive implementation
    def fit(self, data, target, attributes, classes):

        root = self.new_ID3_node()
        root['entropy'] = self.entropy(target)
        root['classCounts'] = Counter(target)
        root['samples'] = len(target)
        root['nodes'] = {}

        if len(set(target)) == 1:
            root['label'] = target[0]
            self.add_node_to_graph(root)
            return root
            
        if not attributes:
            f = Counter(target)
            root['label'] = max(f, key=f.get)
            self.add_node_to_graph(root)
            return root

        a, index = self.find_split_attr(data, target, attributes, classes)
        remaining_attributes = dict(attributes)
        remaining_attributes.pop(a, None)
        root['attribute'] = a


        for value in attributes.get(a):
            data_subset, target_subset = self.partition(data,target,value,attributes,a)
            if len(data_subset) == 0:
                leaf = self.new_ID3_node()
                leaf['label'] = max(set(target),key=target.count)
                leaf['samples'] = 0
                leaf['classCounts'] = Counter(target_subset)
                root['nodes'][value] = leaf
                self.add_node_to_graph(leaf, root['id'])
            else:
                node = self.fit(data_subset, target_subset, remaining_attributes, classes)
                self.add_node_to_graph(node, root['id'])
                root['nodes'][value] = node
        self.add_node_to_graph(root)
        return root


    # Calculates entropy for a set of targets
    def entropy(self,target):

        nbr_samples = len(target)
        Y = set(target)
        entropy = 0

        for y in Y:
            py = target.count(y) / nbr_samples
            entropy += -py * np.log2(py)
        return entropy

    # Calculates infogain of splitting data on attribute a
    def infogain(self, data, target, a, index, current_entropy, attributes):

        N = len(target)
        gain = 0

        for value in attributes.get(a):
            _, target_subset = self.partition(data,target,value, attributes, a)
            n = len(target_subset)
            gain += n/N * self.entropy(target_subset)

        return current_entropy - gain


    # Splits data and targets to subsets where attribute a=value,
    def partition(self,data,target,value, attributes, a):
        data_subset = []
        target_subset = []
        index = list(attributes.keys()).index(a)

        for i in range(len(data)):
            if data[i][index] == value:
                data_subset.append(data[i])
                target_subset.append(target[i])

        data_shrink = []
        for j in range(len(data_subset)):
            row = data_subset[j][:index] + data_subset[j][index+1:]
            data_shrink.append(row)

        return data_shrink, target_subset



    def predict(self, data, root, attributes) :
        predicted = []
        for i in range(len(data)):
            result_node = self.predict_rek(root, data[i],attributes)
            predicted.append(result_node['label'])
        
        return predicted
        
    def predict_rek(self, node, x, attributes):

        if( not node['nodes']):
            return node
            
        attribute = node['attribute']
        index = list(attributes.keys()).index(attribute)
        value = x[index]
        child = node['nodes'][value]

        return self.predict_rek(child, x, attributes)