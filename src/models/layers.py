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
                input_dim,
                output_dim,
                scale=1,
                random_init=True,
                with_bias=True,
                device='cuda:0' if torch.cuda.is_available() else 'mps'):
        super(GCNConvCustom, self).__init__()

        """Metadata"""
        self.device = device # initialize the hosting device
        self.with_bias = with_bias
        self.scale = scale

        self.As = None

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

    def forward(self, H, edge_list):
        self.A_s = self.get_normalized_adj(edge_list)

        if self.with_bias:
            return self.A_s @ (H @ self.W) + self.b.T
        else:
            return self.A_s @ (H @ self.W)

    def get_normalized_adj(self, edge_list):
        """Calculate Matrices"""
        # the adjacency matrix with self-loop

        self.A = gutils.to_dense_adj(edge_list).to(self.device)[0]
        self.A_self = self.A + torch.diag(torch.ones(self.A.shape[0], device=self.device))

        # calculate the degree matrix with A after added self loop
        self.D = torch.sum(self.A_self, dim=0).to(self.device)  # Note: these are the elements along the diagonal of D

        # for diagonal matrix, raising it to any power is the same as raising its diagonal elements to that power
        # we can just apply the -1/2 power to all element of this degree matrix
        self.D_half_norm = torch.diag(torch.pow(self.D, -0.5))

        # normalized adjacency matrix
        self.A_s = self.D_half_norm @ self.A_self @ self.D_half_norm
        self.A_s = self.A_s.to(self.device)
        return self.A_s