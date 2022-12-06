#####################################################################################
# Model Module (PyTorch Geometric)                                                  #
#                                                                                   #
# Description:                                                                      #
# This module contains GCN, GCN-AE, GCN-VAE, GraphSage and more models implemented  #
# using pytorch-geometric 2.1.0                                                     #
#                                                                                   #
# Code References:                                                                  #
# 1. https://pytorch-geometric.readthedocs.io/en/latest/notes/introduction.html     #
#####################################################################################

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
import torch_geometric as gtorch
import torch_geometric.transforms as T
from torch_geometric import utils as gutils
from torch_geometric import nn as gnn # import layers
from torch_geometric.data import NeighborSampler as RawNeighborSampler

# Pytroch Cluster
from torch_cluster import random_walk

# Custom Import
from .components_torch_geometric import *