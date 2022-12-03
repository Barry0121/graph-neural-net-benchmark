# Standard pacakges
import torch
from torch import nn, utils
import torch.nn.functional as F
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import datetime, time
import argparse

# Pytroch Geometric
import torch_geometric.transforms as T
from torch_geometric import utils as gutils
from torch_geometric import nn as gnn # import layers
from torch_geometric.data import NeighborSampler as RawNeighborSampler

# Pytroch Cluster
from torch_cluster import random_walk

# Custom Import
from .layers import *

class GCN(nn.Module):
        def __init__(self,
                    input_size,
                    hidden_size_1,
                    encoding_size,
                    random_init = True,
                    with_bias = True,
                    device= 'cuda:0' if torch.cuda.is_available() else 'cpu'):
            super().__init__()

            # meta information
            self.device = device
            self.input_size = input_size
            self.hidden_size_1 = hidden_size_1
            self.encoding_size = encoding_size
            self.random_init = random_init
            self.with_bias = with_bias


            self.conv1 = GCNConvCustom(input_dim=self.input_size, output_dim=self.hidden_size_1, random_init=self.random_init, with_bias=self.with_bias, device=self.device)
            self.conv2 = GCNConvCustom(input_dim=self.hidden_size_1, output_dim=self.encoding_size, random_init=self.random_init, with_bias=self.with_bias, device=self.device)
            # activations
            self.relu = nn.ReLU()
            self.softmax = nn.Softmax()
            self.dropout = nn.Dropout()

        def forward(self, x, edge_index):
            x = self.conv1(x, edge_index)
            x = self.relu(x)
            x = self.dropout(x)
            x = self.conv2(x, edge_index)

            return self.softmax(x)


class GCN_AE(nn.Module):
    """
    Graph Convolutional Auto-Encoder

    GCN layer implementation from:
        https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.GCNConv
    """
    def __init__(self,
                input_size,
                hidden_size_1,
                hidden_size_2,
                encoding_size,
                random_init = True,
                with_bias = True,
                device= 'cuda:0' if torch.cuda.is_available() else 'mps'):
        super().__init__()
        # meta information
        self.device = device
        self.input_size = input_size
        self.hidden_size_1 = hidden_size_1
        self.hidden_size_2 = hidden_size_2
        self.encoding_size = encoding_size
        self.random_init = random_init
        self.with_bias = with_bias

        # layers
        self.GCN_1 = GCNConvCustom(input_dim=self.input_size, output_dim=self.hidden_size_1, random_init=self.random_init, with_bias=self.with_bias, device=self.device)
        self.GCN_2 = GCNConvCustom(input_dim=self.hidden_size_1, output_dim=self.hidden_size_2, random_init=self.random_init, with_bias=self.with_bias, device=self.device)
        self.FC = nn.Linear(in_features=self.hidden_size_2, out_features=self.encoding_size, device=self.device)
        # activations
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def encoder(self, X, edge_list):
        X_hat = self.GCN_1(X, edge_list) # first layer: lower dimension feature matrix
        X_hat = self.relu(X_hat)
        H = self.GCN_2(X_hat, edge_list) # second layer: mean matrix
        Z = self.relu(H)
        Z = self.FC(H)
        Z = self.relu(Z) # this activation may or may not be here, doesn't make a difference
        return Z

    def decoder(self, Z):
        Y_inner = torch.mm(Z, Z.T) # calculate inner product of matrix
        Y_inner = Y_inner.reshape((-1)) # flatten the tensor
        Y = self.sigmoid(Y_inner) # apply activation
        return Y

    def forward(self, X, edge_list):
        Z = self.encoder(X, edge_list)
        output = self.decoder(Z)
        return output


"""
# This needs to be run before model training
# For each batch and the adjacency matrix
pos_batch = random_walk(row, col, batch,
                          walk_length=1,
                          coalesced=False)[:, 1]
# row are source nodes, col are target nodes from Adjacency matrix
# index 1 is taken as positive nodes
# Random targets from whole adjacency matrix
neg_batch = torch.randint(0, self.adj_t.size(1), (batch.numel(), ),
                                  dtype=torch.long)
"""

class SAGE(nn.Module):
    """
    Implementation is from: https://towardsdatascience.com/pytorch-geometric-graph-embedding-da71d614c3a
    """
    def __init__(self, in_channels, hidden_channels, num_layers):
        super(SAGE, self).__init__()
        self.num_layers = num_layers
        self.convs = gnn.ModuleList()

        for i in range(num_layers):
            in_channels = in_channels if i == 0 else hidden_channels
            self.convs.append(gnn.SAGEConv(in_channels,
                                   hidden_channels))
    def forward(self, x, adjs):
        for i, (edge_index, _, size) in enumerate(adjs):
            x_target = x[:size[1]]
            x = self.convs[i]((x, x_target), edge_index)
            if i != self.num_layers - 1:
                x = x.relu()
                x = F.dropout(x, p=0.5, training=self.training)
        return x
    def full_forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i != self.num_layers - 1:
                x = x.relu()
                x = F.dropout(x, p=0.5, training=self.training)
        return x