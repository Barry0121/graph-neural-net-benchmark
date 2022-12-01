# Standard pacakges
import torch
from torch import nn, utils
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import datetime, time
import argparse

# Pytroch Geometric
from torch_geometric import utils as gutils
from torch_geometric import nn as gnn # import layers
from torch_geometric.datasets import Planetoid # import dataset CORA

# Custom Import
from .layers import *

class GCN(nn.Module):
        def __init__(self,
                    edge_index,
                    input_size,
                    hidden_size_1,
                    encoding_size,
                    random_init = True,
                    with_bias = True,
                    device= 'cuda:0' if torch.cuda.is_available() else 'cpu'):
            super().__init__()

            # meta information
            self.device = device
            self.edge_index = edge_index
            self.input_size = input_size
            self.hidden_size_1 = hidden_size_1
            self.encoding_size = encoding_size
            self.random_init = random_init
            self.with_bias = with_bias


            self.conv1 = GCNConvCustom(edge_index=self.edge_index, input_dim=self.input_size, output_dim=self.hidden_size_1, random_init=self.random_init, with_bias=self.with_bias, device=self.device)
            self.conv2 = GCNConvCustom(edge_index=self.edge_index, input_dim=self.hidden_size_1, output_dim=self.encoding_size, random_init=self.random_init, with_bias=self.with_bias, device=self.device)
            # activations
            self.relu = nn.ReLU()
            self.softmax = nn.Softmax()
            self.dropout = nn.Dropout()

        def forward(self, x):
            x = self.conv1(x)
            x = self.relu(x)
            x = self.dropout(x)
            x = self.conv2(x)

            return self.softmax(x)


class GCN_AE(nn.Module):
    """
    Graph Convolutional Auto-Encoder

    GCN layer implementation from:
        https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.GCNConv
    """
    def __init__(self,
                edge_index,
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
        self.edge_index = edge_index # same as edge list, replace adjacency matrix
        self.input_size = input_size
        self.hidden_size_1 = hidden_size_1
        self.hidden_size_2 = hidden_size_2
        self.encoding_size = encoding_size
        self.random_init = random_init
        self.with_bias = with_bias

        # training utilities
        # self.criterion = None
        # self.optimizer = None

        # layers
        self.GCN_1 = GCNConvCustom(edge_index=self.edge_index, input_dim=self.input_size, output_dim=self.hidden_size_1, random_init=self.random_init, with_bias=self.with_bias, device=self.device)
        self.GCN_2 = GCNConvCustom(edge_index=self.edge_index, input_dim=self.hidden_size_1, output_dim=self.encoding_size, random_init=self.random_init, with_bias=self.with_bias, device=self.device)
        # self.FC = nn.Linear(in_features=self.hidden_size_2, out_features=self.encoding_size, device=self.device)

        # activations
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def encoder(self, X):
        X_hat = self.GCN_1(X) # first layer: lower dimension feature matrix
        # X_hat = self.relu(X_hat)
        H = self.GCN_2(X_hat) # second layer: mean matrix
        # Z = self.relu(H)
        # Z = self.FC(H)
        # Z = self.relu(Z) # this activation may or may not be here, doesn't make a difference
        return H

    def decoder(self, Z):
        Y_inner = torch.mm(Z, Z.T) # calculate inner product of matrix
        # Y_inner = Y_inner.reshape((-1)) # flatten the tensor
        Y = self.sigmoid(Y_inner) # apply activation
        return Y

    def forward(self, X):
        Z = self.encoder(X)
        output = self.decoder(Z)
        return output


class GCN_VAE(nn.Module):
    """
    Graph Variational Convolutional Auto-Encoder

    GCN layer implementation from:
        https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.GCNConv
    """
    def __init__(self,
                edge_index,
                input_size,
                hidden_size,
                encoding_size,
                output_size,
                device= 'cuda:0' if torch.cuda.is_available() else 'mps'):
        super().__init__()
        # meta information
        self.device = device
        self.edge_index = edge_index # same as edge list, replace adjacency matrix
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.encoding_size = encoding_size
        self.output_size = output_size

        # layers
        self.GCN_1 = gnn.GCNConv(in_channels=self.input_size, out_channels=self.hidden_size, normalize=False, device=self.device)
        self.GCN_2 = gnn.GCNConv(in_channels=self.hidden_size, out_channels=self.encoding_size, normalize=False, device=self.device)

        # activations
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def encoder(self, X):
        X_hat = self.GCN_1(X, self.edge_index) # first layer: lower dimension feature matrix
        X_hat = self.relu(X_hat)
        Z_mean = self.GCN_2(X_hat, self.edge_index) # second layer: mean matrix
        Z_logstd = self.GCN_2(X_hat, self.edge_index) # second layer: natural log of squared standard deviation
        Z_std = torch.sqrt(torch.exp(Z_logstd)) # calculate the standard deviation
        Z = torch.normal(Z_mean, Z_std) # random sample from normal distribution with the calculated mean and std
        return Z

    def decoder(self, Z):
        Y_inner = torch.mm(Z.T, Z) # calculate inner product of matrix
        # Y_inner = Y_inner.reshape((-1)) # flatten the tensor
        Y = self.sigmoid(Y_inner) # apply activation
        return Y

    def forward(self, X):
        Z = self.encoder(X)
        output = self.decoder(Z)
        return output