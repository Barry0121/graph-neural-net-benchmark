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
import datetime

# Pytroch Geometric
from torch_geometric import utils as gutils
from torch_geometric import nn as gnn # import layers
from torch_geometric.datasets import Planetoid # import dataset CORA
import torch_geometric.transforms as T

def loader_cora_torch(filepath="../data/raw/Planetoid",
                        num_train_per_class=20, num_val=500, num_test=1000, transform=None,
                        device='cuda:0' if torch.cuda.is_available() else 'cpu'):
    """Return the CORA dataset"""
    dataset = Planetoid(root=filepath, name='Cora', split='public',
                        num_train_per_class=num_train_per_class, num_val=num_val, num_test=num_test,
                        transform=transform) # return a class of datasets
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

def loader_pubmed_torch(filepath="../data/raw/Planetoid",
                        num_train_per_class=20, num_val=500, num_test=1000, transform=T.ToSparseTensor(),
                        device='cuda:0' if torch.cuda.is_available() else 'cpu'):
    """Return the PubMed dataset"""
    dataset = Planetoid(root=filepath, name='PubMed', split='public',
                        num_train_per_class=num_train_per_class, num_val=num_val, num_test=num_test,
                        transform=transform) # return a class of datasets
    data = dataset[0].to(device)
    # print some dataset statistics
    print(f'Loads PubMed dataset, at root location: {filepath}')
    print(f'Number of nodes: {data.num_nodes}')
    print(f'Number of edges: {data.num_edges}')
    print(f'Average node degree: {data.num_edges / data.num_nodes:.2f}')
    print(f'Number of training nodes: {data.train_mask.sum()}')
    print(f'Training node label rate: {int(data.train_mask.sum()) / data.num_nodes:.2f}')
    print(f'Has isolated nodes: {data.has_isolated_nodes()}')
    print(f'Has self-loops: {data.has_self_loops()}')
    print(f'Is undirected: {data.is_undirected()}')
    return data

def loader_citeseer_torch(filepath="../data/raw/Planetoid",
                        num_train_per_class=20, num_val=500, num_test=1000, transform=None,
                        device='cuda:0' if torch.cuda.is_available() else 'cpu'):
    """Return the CiteSeer dataset"""
    dataset = Planetoid(root=filepath, name='CiteSeer', split='public',
                        num_train_per_class=num_train_per_class, num_val=num_val, num_test=num_test,
                        transform=transform) # return a class of datasets
    data = dataset[0].to(device)
    # print some dataset statistics
    print(f'Loads CiteSeer dataset, at root location: {filepath}')
    print(f'Number of nodes: {data.num_nodes}')
    print(f'Number of edges: {data.num_edges}')
    print(f'Average node degree: {data.num_edges / data.num_nodes:.2f}')
    print(f'Number of training nodes: {data.train_mask.sum()}')
    print(f'Training node label rate: {int(data.train_mask.sum()) / data.num_nodes:.2f}')
    print(f'Has isolated nodes: {data.has_isolated_nodes()}')
    print(f'Has self-loops: {data.has_self_loops()}')
    print(f'Is undirected: {data.is_undirected()}')
    return data
