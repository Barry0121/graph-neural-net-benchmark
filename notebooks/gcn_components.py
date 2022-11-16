#####################################################################################
# Graph Neural Network Components Module                                            #
#                                                                                   #
# Description:                                                                      #
# This module contains the different graph neural network components, including     #
# models, custom layers, and different optimizer and loss functions.                #
#                                                                                   #
# Code References:                                                                  #
# 1.                                                                                #
#####################################################################################

# Standard pacakges
import torch
from torch import nn, utils
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

# Pytroch Geometric
from torch_geometric import utils as gutils
from torch_geometric import nn as gnn # import layers
from torch_geometric.datasets import Planetoid # import dataset CORA


class GCNConvCustom(nn.Module):
    def __init__(self,
                edge_index,
                # node_batch,
                input_dim,
                output_dim,
                random_init=True,
                with_bias=True,
                device='cuda:0' if torch.cuda.is_available() else 'mps'):
        super(GCNConvCustom, self).__init__()
        # print("layer initialized")

        """Metadata"""
        self.device = device # initialize the hosting device
        self.with_bias = with_bias

        """Calculate Matrices"""
        # the adjacency matrix with self-loop

        self.A = gutils.to_dense_adj(edge_index).to(self.device)[0]
        self.A_self = self.A + torch.diag(torch.ones(self.A.shape[0], device=self.device))
        # print("Adj Matrix with self loop: ", self.A)

        # calculate the degree matrix with A after added self loop
        self.D = torch.sum(self.A_self, dim=0).to(self.device)  # Note: these are the elements along the diagonal of D
        # print("Degree Matrix: ", self.D)

        # for diagonal matrix, raising it to any power is the same as raising its diagonal elements to that power
        # we can just apply the -1/2 power to all element of this degree matrix
        # self.D_half_norm = torch.reciprocal(torch.sqrt(self.D))
        # self.D_half_norm = torch.from_numpy(fractional_matrix_power(self.D, -0.5)).to(self.device)
        self.D_half_norm = torch.diag(torch.pow(self.D, -0.5))
        # print("Normalization Matrix: ", self.D_half_norm)

        # normalized adjacency matrix
        # self.A_s = torch.mm(torch.mm(self.D_half_norm, self.A), self.D_half_norm)
        self.A_s = self.D_half_norm @ self.A_self @ self.D_half_norm
        self.A_s = self.A_s.to(self.device)
        # print(self.A_s.shape)

        # initialize learnable weights
        # the weight should have shape of (N , F) where N is the size of the input, and F is the output dimension
        self.W, self.b = None, None
        if random_init:
            self.W = torch.nn.Parameter(
                data=(torch.rand(input_dim, output_dim, device=self.device)*0.01),  # times it by 0.001 to make the weight smaller
                requires_grad=True
            )
            # create trainable a bias term for the layer
            self.b = torch.nn.Parameter(
                data=(torch.rand(output_dim, 1, device=self.device)*0.01),
                requires_grad=True
            )
        else:
            self.W = torch.nn.Parameter(
                data=torch.ones(input_dim, output_dim, device=self.device),
                requires_grad=True
            )
            self.b = torch.nn.Parameter(
                data=torch.ones(output_dim, 1, device=self.device),
                requires_grad=True
            )

    def forward(self, H):
        if self.with_bias:
            return self.A_s @ H @ self.W + self.b.T
        else:
            return self.A_s @ H @ self.W

    def get_adj_matrix(self, with_self=False):
        if with_self:
            return self.A_self
        return self.A

    def get_normalized_adj_matrix(self):
        return self.A_s

    def get_degree_matrix(self, normalization=False):
        if normalization:
            return self.D_half_norm
        return self.D


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
                device= 'cuda:0' if torch.cuda.is_available() else 'mps'):
        super().__init__()
        # meta information
        self.device = device
        self.edge_index = edge_index # same as edge list, replace adjacency matrix
        self.input_size = input_size
        self.hidden_size_1 = hidden_size_1
        self.hidden_size_2 = hidden_size_2
        self.encoding_size = encoding_size

        # training utilities
        # self.criterion = None
        # self.optimizer = None

        # layers
        self.GCN_1 = GCNConvCustom(edge_index=self.edge_index, input_dim=self.input_size, output_dim=self.hidden_size_1, random_init=True, device=self.device)
        self.GCN_2 = GCNConvCustom(edge_index=self.edge_index, input_dim=self.hidden_size_1, output_dim=self.hidden_size_2, random_init=True, device=self.device)
        self.FC = nn.Linear(in_features=self.hidden_size_2, out_features=self.encoding_size, device=self.device)

        # activations
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def encoder(self, X):
        X_hat = self.GCN_1(X) # first layer: lower dimension feature matrix
        X_hat = self.relu(X_hat)
        H = self.GCN_2(X_hat) # second layer: mean matrix
        H = self.relu(H)
        Z = self.FC(H)
        # Z = self.relu(Z) # this activation may or may not be here, doesn't make a difference
        return Z

    def decoder(self, Z):
        # TODO: we don't want matrix product, but inner product of each encoded vector
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
        Y_inner = Y_inner.reshape((-1)) # flatten the tensor
        Y = self.sigmoid(Y_inner) # apply activation
        return Y

    def forward(self, X):
        Z = self.encoder(X)
        output = self.decoder(Z)
        return output
