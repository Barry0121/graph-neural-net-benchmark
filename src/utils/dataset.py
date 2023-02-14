"""
dataset.py
- Functions to retrieve datasets.
- Instantiate and Train Graph2Vec embedding.
- Generate embeddings for each graph dataset.
"""

import torch
from torch_geometric.datasets import TUDataset
from torch_geometric.utils import to_networkx
from karateclub import Graph2Vec
import numpy as np
import random
from random import shuffle
import networkx as nx
import time

def get_dataset(name, train_val_ratio=0.8, filepath="../data/TUDataset", seed=42):
    """Return list of graphs"""
    # create graphs dataset
    dataset = TUDataset(root=filepath, name=name)
    graphs = [to_networkx(data)  for data in dataset]
    # split dataset
    random.seed(seed)
    shuffle(graphs)
    graphs_len = len(graphs)
    graphs_test = graphs[int(train_val_ratio * graphs_len):]
    graphs_train = graphs[0:int(train_val_ratio*graphs_len)]
    graphs_validate = graphs[0:int((train_val_ratio/4)*graphs_len)]
    return graphs_train, graphs_validate, graphs_test

def get_graph2vec(name, dataset_source="../data/TUDataset"):
    """Get the graph embedding"""
    dataset = TUDataset(root=dataset_source, name=name)
    graphs = [to_networkx(data)  for data in dataset]
    print("======Generating Embedding======")
    start = time.time()
    graph2vec = Graph2Vec(dimensions=128, workers=-1)
    graph2vec.fit(graphs)
    print(f"======Embedding Created (used {(time.time()-start) % 60} sec)======")
    return graph2vec

# =========From GraphRNN: load data for training==================

def encode_adj(adj, max_prev_node=10, is_full = False):
    '''
    From GraphRNN codebase
    :param adj: n*n, rows means time step, while columns are input dimension
    :param max_degree: we want to keep row number, but truncate column numbers
    :return:
    '''
    if is_full:
        max_prev_node = adj.shape[0]-1

    # pick up lower tri
    adj = np.tril(adj, k=-1)
    n = adj.shape[0]
    adj = adj[1:n, 0:n-1]

    # use max_prev_node to truncate
    # note: now adj is a (n-1)*(n-1) matrix
    adj_output = np.zeros((adj.shape[0], max_prev_node))
    for i in range(adj.shape[0]):
        input_start = max(0, i - max_prev_node + 1)
        input_end = i + 1
        output_start = max_prev_node + input_start - input_end
        output_end = max_prev_node
        adj_output[i, output_start:output_end] = adj[i, input_start:input_end]
        adj_output[i,:] = adj_output[i,:][::-1] # reverse order

    return adj_output

def decode_adj(adj_output):
    '''
    From GraphRNN codebase
        recover to adj from adj_output
        note: here adj_output have shape (n-1)*m
    '''
    max_prev_node = adj_output.shape[1]
    adj = np.zeros((adj_output.shape[0], adj_output.shape[0]))
    for i in range(adj_output.shape[0]):
        input_start = max(0, i - max_prev_node + 1)
        input_end = i + 1
        output_start = max_prev_node + max(0, i - max_prev_node + 1) - (i + 1)
        output_end = max_prev_node
        adj[i, input_start:input_end] = adj_output[i,::-1][output_start:output_end] # reverse order
    adj_full = np.zeros((adj_output.shape[0]+1, adj_output.shape[0]+1))
    n = adj_full.shape[0]
    adj_full[1:n, 0:n-1] = np.tril(adj, 0)
    adj_full = adj_full + adj_full.T

    return adj_full

class get_sequence_sampler_dataset(torch.utils.data.Dataset):
    def __init__(self, G_list, max_num_node=None):
        self.adj_all = []
        self.len_all = []
        for G in G_list:
            self.adj_all.append(np.asarray(nx.to_numpy_matrix(G)))
            self.len_all.append(G.number_of_nodes())
        if max_num_node is None:
            self.n = max(self.len_all)
        else:
            self.n = max_num_node

    def __len__(self):
        return len(self.adj_all)

    def __getitem__(self, idx):
        adj_copy = self.adj_all[idx].copy()
        x_batch = np.zeros((self.n, self.n-1))  # here zeros are padded for small graph
        x_batch[0,:] = 1 # the first input token is all ones
        y_batch = np.zeros((self.n, self.n-1))  # here zeros are padded for small graph
        # generate input x, y pairs
        len_batch = adj_copy.shape[0]
        x_idx = np.random.permutation(adj_copy.shape[0])
        adj_copy = adj_copy[np.ix_(x_idx, x_idx)]
        adj_encoded = encode_adj(adj_copy.copy(), max_prev_node=self.n-1)
        # get x and y and adj
        # for small graph the rest are zero padded
        y_batch[0:adj_encoded.shape[0], :] = adj_encoded
        x_batch[1:adj_encoded.shape[0] + 1, :] = adj_encoded
        return {'x':x_batch,'y':y_batch, 'len':len_batch}

def get_dataloader(dataset, batch_size=32, num_workers=0):
    """Return dataloader for training"""
    sample_strategy = torch.utils.data.sampler.WeightedRandomSampler([1.0 / len(dataset) for i in range(len(dataset))], num_samples=batch_size**2, replacement=True)
    return torch.utils.data.DataLoader(get_sequence_sampler_dataset(dataset), batch_size=batch_size, num_workers=num_workers, sampler=sample_strategy)
