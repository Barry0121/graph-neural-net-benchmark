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
import datetime, time
import argparse

# Pytroch Geometric
from torch_geometric import utils as gutils
from torch_geometric import nn as gnn # import layers
from torch_geometric.datasets import Planetoid # import dataset CORA

# Import custom scripts
from src.models.model_utility import *
from src.models.gcn_components import *
print("Package Import Successful!")
time.sleep(2)

def plot_result(train_loss, val_loss, val_acc):
    plt.plot(train_loss, color='blue')
    plt.plot([i.cpu().data for i in val_loss], color='orange')
    plt.show()

    plt.plot(val_acc, color='green')
    plt.show()

def save_result(train_loss, val_loss, val_acc, name):
    time_ran = str(datetime.datetime.now()).replace(" ", '-')
    plt.plot(train_loss, color='blue')
    plt.plot([i.cpu().data for i in val_loss], color='orange')
    plt.savefig(f"./src/visualizations/{time_ran}_{name}_loss.png")

    plt.plot(val_acc, color='green')
    plt.savefig(f"./src/visualizations/{time_ran}_{name}_accuracy.png")

def main(name="node-classification", epoches=200, hidden_size=1000, encode_size=100):
    ##########################
    # Define Meta-variables  #
    ##########################

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    hidden_size = encode_size
    encode_size = hidden_size
    epoches = epoches
    name = name


    ##########################
    # Load CORA testing Data #
    ##########################
    filepath = "./test/testdata/cora"
    data = loader_cora_torch(filepath=filepath,
                        transform=None,
                        batch_size=1,
                        shuffle=True,
                        device=device)

    # Note: we use train/val/test masks to create sub-datasets of edge indices for evaluation
    edge_index, node_features, labels, train_mask, val_mask, test_mask = \
        data.edge_index, data.x, data.y, data.train_mask, data.val_mask, data.test_mask
    print("Data loaded in 'test/testdata' directory! ")
    time.sleep(2)

    ################################
    # Setup Model, Loss, Optimizer #
    ################################

    class GCN(nn.Module):
        def __init__(self,
                    edge_index,
                    input_size,
                    hidden_size_1,
                    encoding_size,
                    random_init = True,
                    with_bias = True,
                    device= 'cuda:0' if torch.cuda.is_available() else 'cpu'):
            super().__init__()

            # meta information
            self.device = device
            self.edge_index = edge_index
            self.input_size = input_size
            self.hidden_size_1 = hidden_size_1
            self.encoding_size = encoding_size
            self.random_init = random_init
            self.with_bias = with_bias


            self.conv1 = GCNConvCustom(edge_index=self.edge_index, input_dim=self.input_size, output_dim=self.hidden_size_1, random_init=self.random_init, with_bias=self.with_bias, device=self.device)
            self.conv2 = GCNConvCustom(edge_index=self.edge_index, input_dim=self.hidden_size_1, output_dim=self.encoding_size, random_init=self.random_init, with_bias=self.with_bias, device=self.device)
            # activations
            self.relu = nn.ReLU()
            self.softmax = nn.Softmax()
            self.dropout = nn.Dropout()

        def forward(self, x):
            x = self.conv1(x)
            x = self.relu(x)
            x = self.dropout(x)
            x = self.conv2(x)

            return self.softmax(x)

    model = GCN(edge_index=edge_index, input_size=node_features.shape[1], hidden_size_1=hidden_size, encoding_size=encode_size, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()

    print("Model Initialized!")
    time.sleep(3)

    #####################################
    # Train and Validation on GCN Model #
    #####################################
    print("Start Training!")
    train_loss, val_loss, val_acc = [], [], []

    for epoch in range(epoches):
        model.train()
        optimizer.zero_grad()
        out = model(node_features)
        loss = criterion(out[train_mask], labels[train_mask])
        loss.backward()
        optimizer.step()

        model.eval()
        output = model(node_features)
        vloss = criterion(output[val_mask], labels[val_mask])

        train_loss.append(loss.cpu().detach().numpy())
        val_loss.append(vloss.cpu().detach())
        val_acc.append(np.mean((torch.argmax(output[val_mask], dim=1) == labels[val_mask]).cpu().detach().numpy()))

    # save_result(train_loss, val_loss, val_acc, name) # there are some issue with plt.savefig()
    print("Finish Training!")
    time.sleep(3)

    ##################
    # Test GCN Model #
    ##################
    model.eval()
    pred = model(node_features).argmax(dim=1)
    correct = (pred[test_mask] == labels[test_mask]).sum()
    acc = int(correct) / int(test_mask.sum())
    print(f'Test Accuracy: {acc:.4f}')


#############################
# Run Script with Arguments #
#############################
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train and Test a Node Classification GCN')
    parser.add_argument('-n', '--name', type=str, required=True, help='The Name for this Experiment')
    parser.add_argument('-e', '--epoches', default=100, type=int)
    parser.add_argument('-hs', '--hidden_size', default=1000, type=int)
    parser.add_argument('-es', '--encode_size', default=50, type=int)
    args = parser.parse_args()

    name=args.name.replace(' ', '_')
    epoches=args.epoches
    hidden_size=args.hidden_size
    encode_size=args.encode_size

    print("Name of the Experiment: ", name, '\n')
    print(f"Train with {epoches} # of epoches \n")
    print(f"...with {hidden_size} as hidden layer's size \n")
    print(f"...with {encode_size} as encoding size. \n")

    main(name, epoches, hidden_size, encode_size)