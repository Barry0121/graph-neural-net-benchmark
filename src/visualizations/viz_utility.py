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

def plot_result(train_loss, val_loss, val_acc):
    plt.plot(train_loss, color='blue')
    plt.plot([i.cpu().data for i in val_loss], color='orange')
    plt.show()

    plt.plot(val_acc, color='green')
    plt.show()