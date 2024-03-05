"""
util.py holds utility functions for difference use cases.
These include:
    - outputting data/results
    - calculating differences
    - resetting datasets
    - handling datasets
    - adding edges
    - checking for edges
    - conversions between data types, etc.
    - loading models
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv
from torch_geometric.utils import to_networkx, from_networkx
import networkx as nx
import numpy as np
import random, math
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Dataset:
    def __init__(self, root='/tmp/Cora', name='Cora', device='cuda'):
        self.root = root
        self.name = name
        self.device = device
        self.data = None
        self.in_feats = None
        self.h_feats = None
        self.num_classes = None
        self.reset_dataset()

    def reset_dataset(self):
        cora_dataset = Planetoid(root=self.root, name=self.name)
        self.data = cora_dataset[0]
        self.data = self.data.to(self.device)

        self.in_feats = self.data.x.shape[1]
        self.h_feats = 64
        self.num_classes = cora_dataset.num_classes

    def get_data(self):
        self.reset_dataset()
        return self.data, self.in_feats, self.h_feats, self.num_classes

class GCN(nn.Module):
    def __init__(self, in_feats, h_feats, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_feats, h_feats)
        self.conv2 = GCNConv(h_feats, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        h = self.conv1(x, edge_index)
        h = F.relu(h)
        h = self.conv2(h, edge_index)
        return h

def get_model(in_feats, h_feats, num_classes, name):
    model = GCN(in_feats, h_feats, num_classes)
    model = model.to(device)
    model.load_state_dict(torch.load(f'../../model/{name}_gt.pt'))
    model.eval()

    return model

def test_model(model, d):
    model.eval()
    out = model(d)
    pred = out.argmax(dim=1)
    
    acc = (pred[d.test_mask] == d.y[d.test_mask]).sum().item() / d.test_mask.sum().item()
    return acc

def output_accuracy_change(gt, cv):
    print("\n----")
    if gt != cv:
        print(f'The accuracy has changed by {cv - gt:.4f}')
    else:
        print("The accuracy has not changed.")

def convert_to_networkx(g):
    G = to_networkx(g)
    x = g.x
    y = g.y
    train_mask = g.train_mask
    test_mask = g.test_mask

    return G, x, y, train_mask, test_mask
    
def convert_to_pyg(G, x, y, train_mask, test_mask):
    d = from_networkx(G).to(device)

    d.x = x
    d.y = y
    d.train_mask = train_mask
    d.test_mask = test_mask
    d.num_nodes = None

    return d

def add_edge(g, i, j, undirected):
    if g.has_edge(i, j):
        return;
    
    if undirected:
        g.add_edge(i, j)
        g.add_edge(j, i)
    else:
        g.add_edge(i, j)

def get_ground_truth(model, data):
    return test_model(model, data)

def number_added_edges(init, final, is_undirected):
    change = final - init

    if is_undirected:
        percentage_change = (change / init) * 100 if init != 0 else 0
        print("Change in edges: ", change / 2, " | Percentage change: {:.2f}%".format(percentage_change))
    else:
        percentage_change = (change / init) * 100 if init != 0 else 0
        print("Change in edges: ", change, " | Percentage change: {:.2f}%".format(percentage_change))


def print_graph(G):
    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(G)
    
    nx.draw_networkx_nodes(G, pos, node_size=10)
    
    nx.draw_networkx_edges(G, pos, alpha=0.5)
    
    plt.title("Large Graph Visualization")
    plt.xticks([])
    plt.yticks([])
    plt.show()


def make_clique(G, s):
    s = list(s)
    for i in range(0, len(s)):
        for j in range(i + 1, len(s)):
            add_edge(G, s[i], s[j], undirected=True)

def get_total_homophily(data):
    G, x, y, train_mask, test_mask = convert_to_networkx(data)
    same = 0
    num_edges = 0
    
    for i in range(0, G.number_of_nodes()):
        edges = G.out_edges(i)
        for e in edges:
            if y[e[0]] == y[e[1]]:
                same += 1
            num_edges += 1

    return same / num_edges
    
def get_node_homophily_rates(G, n, y, num_classes):
    edges = G.out_edges(n)
    vals = {key: 0 for key in range(num_classes)}
    for edge in edges:
        vals[y[edge[1]].item()] += 1
        
    for v in vals.keys():
        vals[v] = vals[v]/len(edges)
    
    return vals