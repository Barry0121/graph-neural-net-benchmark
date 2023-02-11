"""
dataset.py
- Functions to retrieve datasets.
- Instantiate and Train Graph2Vec embedding.
"""

import torch
from torch_geometric.datasets import TUDataset
from torch_geometric.utils import to_networkx

def get_dataset(name, filepath="../data/TUDataset"):
    """Return list of graphs"""
    dataset = TUDataset(root=filepath, name=name)
    graphs = [to_networkx(data)  for data in dataset]
    return graphs

def get_embeddings(name, filepath="../cache/Embedding"):
    """Get the graph embedding"""
    graphs = get_dataset(name)
