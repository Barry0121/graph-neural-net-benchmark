# Standard pacakges
import torch
from torch import nn, utils
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import datetime, time
import argparse
import os

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

def save_result(root_path, expt_name, train_loss, val_loss, val_acc):
    # create result folder
    os.makedirs(os.path.join(root_path, expt_name))

    path1 = os.path.join(root_path, expt_name, 'train_val_loss')
    fig = plt.figure(figsize=(15, 15))
    ax = plt.axes()
    ax.set_title("Loss vs Epochs")
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Loss")
    ax.plot(train_loss, color='blue', label='Training Loss')
    ax.plot([i.cpu().data for i in val_loss], color='orange', label='Validation Loss')
    ax.legend()
    plt.savefig(path1)

    path2 = os.path.join(root_path, expt_name, 'val_acc')
    fig = plt.figure(figsize=(15, 15))
    ax = plt.axes()
    ax.set_title("Validation Accuracy vs Epochs")
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Accuracy")
    ax.plot(val_acc, color='green')
    plt.savefig(path2)

    print(f"Saved visualizations at {root_path+'/'+expt_name}")