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
            nodeString += "\n" + str(parentid) + " -> " + str(node['id'])

        print(nodeString)

        return


    # make the visualisation available
    def make_dot_data(self) :
        return self.__dot


    # For you to fill in; Suggested function to find the best attribute to split with, given the set of
    # remaining attributes, the currently evaluated data and target.
    def find_split_attr(self, data, target, attributes):
        best_gain = 0
        best_attribute = ''
        current_entropy = self.entropy(target)

        for _, A in enumerate(attributes):
            gain = self.infogain(data, target, A, current_entropy, attributes)
            if gain > best_gain:
                best_gain = gain
                best_attribute = A
        return best_attribute
    
    def entropy(self, target):
        node_counter = Counter(target)
        values = [int(i[1]) for i in node_counter.items()]
        total = sum([int(i[1]) for i in node_counter.items()])
        ent = 0
        for v in values:
            ent += (v/total)*np.log2(v/total)
        return -ent

    def infogain(self, data, target, A, current_entropy, attributes):

        N = len(target)
        gain = 0

        for value in attributes[A]:
            _, target_sub = self.partition(data,target,value, attributes, A)
            n = len(target_sub)
            gain += n/N * self.entropy(target_sub)

        return current_entropy - gain

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
            cnts = Counter(target)
            root['label'] = max(cnts, key=cnts.get)
            self.add_node_to_graph(root)
            return root
        
        A = self.find_split_attr(data, target, attributes)
        remaining_attr = dict(attributes)
        remaining_attr.pop(A, None)
        root['attribute'] = A

        for v in attributes[A]:
            data_sub, target_sub = self.partition(data, target, v, attributes, A)

            if len(data_sub) == 0:
                leaf = self.new_ID3_node()
                leaf['label'] = max(set(target), key=target.count)
                leaf['samples'] = 0
                leaf['classCounts'] = Counter(target_sub)
                root['nodes'][v] = leaf
                self.add_node_to_graph(leaf, root['id'])
            else:
                node = self.fit(data_sub, target_sub, remaining_attr, classes)
                self.add_node_to_graph(node, root['id'])
                root['nodes'][v] = node
        self.add_node_to_graph(root)
        return root


    def partition(self, data, target, value, attributes, A):
        data_subset = []
        target_subset = []
        index = list(attributes.keys()).index(A)

        for i in range(len(data)):
            if data[i][index] == value:
                data_subset.append(data[i])
                target_subset.append(target[i])

        data_shrink = []
        for j in range(len(data_subset)):
            row = data_subset[j][:index] + data_subset[j][index+1:]
            data_shrink.append(row)

        return data_shrink, target_subset


    def predict(self, data, tree, attributes) :
        predicted = list()

        for d in data:
            node = self.predict_rek(tree, d, attributes)
            predicted.append(node['label'])

        return predicted

    def predict_rek(self, node, x, attributes):
        if not node['nodes']:
            return node
        
        attribute = node['attribute']
        index = list(attributes.keys()).index(attribute)
        value = x[index]
        child = node['nodes'][value]

        return self.predict_rek(child, x, attributes)