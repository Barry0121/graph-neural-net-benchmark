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

def save_result(train_loss, val_loss, val_acc, name):
    time_ran = str(datetime.datetime.now()).replace(" ", '-')
    plt.plot(train_loss, color='blue')
    plt.plot([i.cpu().data for i in val_loss], color='orange')
    plt.savefig(f"./src/visualizations/{time_ran}_{name}_loss.png")

    plt.plot(val_acc, color='green')
    plt.savefig(f"./src/visualizations/{time_ran}_{name}_accuracy.png")

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
