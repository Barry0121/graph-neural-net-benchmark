#####################################################################################
# Model Utility Module                                                              #
#                                                                                   #
# Description:                                                                      #
# This module generalize dataloading, training, validation, and testing process,    #
# model saving and loading procedure, and table and graph generation for all        #
# types of pytorch models.                                                          #
#                                                                                   #
# Code References:                                                                  #
# 1. https://pytorch-geometric.readthedocs.io/en/latest/notes/introduction.html     #
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

# GCN components (My own module)
from gcn_components import *


class model_utils:
    def __init__(self, dataset, epochs):
        # store the data
        # TODO: Change the raw dataset to a dataloader object from PyTorch
        self.dataset = dataset
        if 'x' in self.dataset:
            self.node_features = self.dataset.x
        else:
            print("Input Dataset has no node features.")
        self.edge_index = self.dataset.edge_index
        self.node_labels = self.dataset.y

        # print some dataset statistics
        print(f'Number of nodes: {dataset.num_nodes}')
        print(f'Number of edges: {dataset.num_edges}')
        print(f'Average node degree: {dataset.num_edges / dataset.num_nodes:.2f}')
        if 'train_mask' in dataset:
            print(f'Number of training nodes: {dataset.train_mask.sum()}')
            print(f'Training node label rate: {int(dataset.train_mask.sum()) / dataset.num_nodes:.2f}')
        print(f'Has isolated nodes: {dataset.has_isolated_nodes()}')
        print(f'Has self-loops: {dataset.has_self_loops()}')
        print(f'Is undirected: {dataset.is_undirected()}')


        # training/validation split

        # Hyperparameters
        self.epochs = epochs
        self.train_loss = []
        self.validation_loss = []
        self.test_loss = 0
        self.validation_acc = []
        self.test_acc = 0


    """
    Utility functions:
    - load dataset
    - loss function
    - optimizer
    - train/validation
    - test
    """

    def initialize_training(self):
        """ Initialize Training Utilities """
        pass

    def train_step(self):
        """ One Training Step """
        pass

    def test(self):
        pass


def loader_cora_torch(filepath="../data/raw/Planetoid", transform=None, batch_size=1, shuffle=True, device='cuda:0' if torch.cuda.is_available() else 'mps'):
    """Return the CORA dataset"""
    dataset = Planetoid(root=filepath, name='Cora', split='public', num_train_per_class=20, num_val=500, num_test=1000, transform=transform) # return a class of datasets
    data = dataset[0].to(device)
    # print some dataset statistics
    print(f'Loads Cora dataset, at root location: {filepath}')
    print(f'Number of nodes: {data.num_nodes}')
    print(f'Number of edges: {data.num_edges}')
    print(f'Average node degree: {data.num_edges / data.num_nodes:.2f}')
    print(f'Number of training nodes: {data.train_mask.sum()}')
    print(f'Training node label rate: {int(data.train_mask.sum()) / data.num_nodes:.2f}')
    print(f'Has isolated nodes: {data.has_isolated_nodes()}')
    print(f'Has self-loops: {data.has_self_loops()}')
    print(f'Is undirected: {data.is_undirected()}')

    return data
