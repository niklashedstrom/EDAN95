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

        root = self.id3(data, target, attributes, -1, None, '')

        return root

    def predict(self, data, tree) :
        predicted = list()
        for d in data:
            predicted.append(self.find_label(d, tree, tree['label']))

        return predicted

    def find_label(self, data, tree, label):
        # data = (d,d,d)
        # tree, bunch of nodes
        if label != None:
            return label
        else:
            for t in tree['nodes']:
                if t in data:
                    return self.find_label(data, tree['nodes'][t], tree['nodes'][t]['label'])
    
    def entropy(self, samples, target_attributes, attribute):
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
                        if (val/tot_sum) != 0:
                            ent += (val/tot_sum) * np.log2(val/tot_sum)
                    tot_ent += -ent * tot_sum/len(samples)
            info_dict[a] = entropy-tot_ent
        return info_dict
    
    def add_node(self, node, parentnode, v):
        if parentnode != None:
            if parentnode['nodes'] == None:
                parentnode['nodes'] = {v: node}
            else:
                parentnode['nodes'][v] = node

    def id3(self, samples, target_attributes, attributes, parentid, parentnode, attr):
        node = self.new_ID3_node()
        # print(len(samples))
        nodeCounter = Counter(target_attributes)
        entro = self.entropy(samples, target_attributes, attributes)
        
        # print(attr)

        # If all samples belong to one class <class_name>
        # Return the single-node tree Root, with label = <class_name>.
        cnt = 0
        label = ''
        for c in nodeCounter:
            if nodeCounter[c] > 0:
                cnt += 1
                label = c
        if cnt == 1:
            node['label'] = label
            node['samples'] = len(samples)
            node['classCounts'] = nodeCounter
            node['entropy'] = entro
            # self.add_node(node, parentnode, attr)
            self.add_node_to_graph(node, parentid)
            return node

        # If Attributes is empty, then
        # Return the single node tree Root, with label = most common class value in Samples.
        elif len(attributes) == 0:
            max_val = -1
            label = ''
            for c in nodeCounter:
                if nodeCounter[c] > max_val:
                    max_val = nodeCounter[c]
                    label = c
            node['label'] = label
            node['samples'] = len(samples)
            node['classCounts'] = nodeCounter
            node['entropy'] = entro
            # self.add_node(node, parentnode, attr)
            self.add_node_to_graph(node, parentid)
            return node
        else:
            info_gain = self.info_gain(entro, samples, target_attributes, attributes)
            A = max(info_gain.items(), key=operator.itemgetter(1))[0]

            node['attribute'] = A
            for v in attributes[A]:
                # samp = [val for val in samples if val[A]==v]
                # targ = []
                # for i in range(len(samples)):
                #     if v == samples[i][A]: 
                #         targ.append(target_attributes[i])
                samp, targ = self.partition(samples, target_attributes, v, attributes, A)
                if len(samp) == 0:
                    new_node = self.new_ID3_node()
                    new_node['label'] = max(target_attributes)
                    new_node['samples'] = len(samp)
                    new_node['classCounts'] = nodeCounter
                    # self.add_node(new_node, node, v)
                    self.add_node_to_graph(new_node, node['id'])
                    return new_node
                else:
                    node['attribute'] = A
                    node['entropy'] = entro
                    node['samples'] = len(samples)
                    node['classCounts'] = nodeCounter
                    if len(attributes) != 0:
                        tmp_attr = dict([i for i in attributes.items() if i[0] != A])
                    # self.add_node(node, parentnode, attr)
                    self.add_node_to_graph(node, parentid)
                    node['nodes'] = self.id3(samp, targ, tmp_attr, node['id'], node, v)
        return node

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