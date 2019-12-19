from collections import Counter
from graphviz import Digraph
import operator
import numpy as np

class ID3DecisionTreeClassifier :


    def __init__(self, minSamplesLeaf = 1, minSamplesSplit = 2) :

        self.__nodeCounter = 0

        # the graph to visualise the tree
        self.__dot = Digraph(comment='The Decision Tree', strict=True)

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
        # print('Added node: {}, parentid: {}'.format(node,parentid))
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
    def find_split_attr(self):

        # Change this to make some more sense
        # INFORMATION GAIN 
        return None


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
            counter = Counter(target)
            root['label'] = max(counter, key=counter.get)
            self.add_node_to_graph(root)
            return root
        
        info_gain = self.info_gain(self.entropy(target), data, target, attributes)
        A = max(info_gain.items(), key=operator.itemgetter(1))[0]

        remaining_attr = dict(attributes)
        remaining_attr.pop(A, None)
        root['attribute'] = A

        for v in attributes[A]:
            samp, targ = self.partition(data, target, v, attributes, A)
            if len(samp) == 0:
                leaf = self.new_ID3_node()
                leaf['label'] = max(set(target), key=target.count)
                leaf['samples'] = 0
                leaf['classCounts'] = Counter(targ)
                root['nodes'][v] = leaf
                self.add_node_to_graph(leaf, root['id'])
            else:
                node = self.fit(samp, targ, remaining_attr, classes)
                self.add_node_to_graph(node, root['id'])
                root['nodes'][v] = node
        self.add_node_to_graph(root)
        return root

    def predict(self, data, tree) :
        predicted = list()

        # Not needed, just need to go through the tree to search for the highest info gain going down
        # unitl we hit a label that's not None. 
        # edges = set([(i.split()[0], i.split()[2]) for i in self.__dot.body if '->' in i])
        # print(edges)

        for d in data:
            predicted.append(self.find_label(d, tree))

        # fill in something more sensible here... root should become the output of the recursive tree creation
        return predicted

    def find_label(self, data, tree):
        # data = (d,d,d)
        # tree, bunch of nodes
        if not tree['nodes']:
            return tree['label']
        else:
            for t in tree['nodes']:
                if t in data:
                    return self.find_label(data, tree['nodes'][t])
    
    def entropy(self, target_attributes):
        node_counter = Counter(target_attributes)
        values = [int(i[1]) for i in node_counter.items()]
        total = sum([int(i[1]) for i in node_counter.items()])
        ent = 0
        for v in values:
            ent += (v/total)*np.log2(v/total)
        return -ent

    def info_gain(self, entropy, samples, target_attributes, attributes):
        info_dict = {}
        tmp_samples = []
        classes = set(target_attributes)
        for i in range(len(samples)):
            tmp_samples.append((*samples[i], target_attributes[i]))
        for a in attributes:
            tmp_dict = {i:{j:0 for j in classes} for i in attributes[a]}
            lst = attributes[a]

            for s in tmp_samples:
                for l in lst:
                    if l in s:
                        tmp_dict[l][s[-1]] += 1

            tot_ent = 0
            for key in tmp_dict:
                ent = 0
                tot_sum = sum([int(i[1]) for i in tmp_dict[key].items()])
                if tot_sum > 0:
                    for val in [int(i[1]) for i in tmp_dict[key].items()]:
                        # print('val: {}, sum: {}, div: {}'.format(val, tot_sum, val/tot_sum))
                        if (val/tot_sum) != 0:
                            ent += (val/tot_sum) * np.log2(val/tot_sum)
                    tot_ent += -ent * tot_sum/len(samples)
            info_dict[a] = entropy-tot_ent
        return info_dict

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
