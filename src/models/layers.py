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
                scale=1,
                random_init=True,
                with_bias=True,
                device='cuda:0' if torch.cuda.is_available() else 'mps'):
        super(GCNConvCustom, self).__init__()
        # print("layer initialized")

        """Metadata"""
        self.device = device # initialize the hosting device
        self.with_bias = with_bias
        self.scale = scale

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
                data=(2 * torch.rand(input_dim, output_dim, device=self.device)-1)*self.scale,
                requires_grad=True
            )
            # create trainable a bias term for the layer
            self.b = torch.nn.Parameter(
                data=(2 * torch.rand(output_dim, 1, device=self.device)-1)*self.scale,
                requires_grad=True
            )
        else:
            self.W = torch.nn.Parameter(
                data=torch.zeros(input_dim, output_dim, device=self.device),
                requires_grad=True
            )
            self.b = torch.nn.Parameter(
                data=torch.zeros(output_dim, 1, device=self.device),
                requires_grad=True
            )

    def forward(self, H):
        if self.with_bias:
            return self.A_s @ (H @ self.W) + self.b.T
        else:
            return self.A_s @ (H @ self.W)

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