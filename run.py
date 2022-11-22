"""
Write up the Graphsage for node prediction task on <New> Testing Dataset

TODO: Figure out how to structure a py file to run all the functions when called

1. Load Testing Dataset and store them in /test/testdata
2. Load Pre-trained Model Parameter
    * Might require new training
    * Accept argument parsing
3. Run Test and Report Test Statistics and Graph
"""

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

# Import custom scripts
from src.models.model_utility import *
from src.models.gcn_components import *

##########################
# Load CORA testing Data #
##########################
filepath = "./test/testdata/cora"
data = loader_cora_torch(filepath=filepath,
                    transform=None,
                    batch_size=1,
                    shuffle=True,
                    device='cuda:0' if torch.cuda.is_available() else 'cpu')

# Note: we use train/val/test masks to create sub-datasets of edge indices for evaluation
edge_index, node_features, labels, train_mask, val_mask, test_mask = \
    data.edge_index, data.node_features, data.labels, data.train_mask, data.val_mask, data.test_mask


#####################################
# Train and Validation on GCN Model #
#####################################



##################
# Test GCN Model #
##################

### TODO: write into function and enable argument passing 