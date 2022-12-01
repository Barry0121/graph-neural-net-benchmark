# Standard pacakges
import torch
from torch import nn, utils
import numpy as np
import time, re
import argparse

# Pytroch Geometric
# from torch_geometric import utils as gutils
# from torch_geometric import nn as gnn # import layers
# from torch_geometric.datasets import Planetoid # import dataset CORA
import torch_geometric.transforms as T

# Import custom scripts
from src.models.utility import *
from src.models.layers import *
from src.models.models import *
from src.visualizations.viz_utility import *

print("Package Import Successful!")
time.sleep(2)

def main(name="node-classification", dataset='cora', task='nodeclassification',
        epochs=200, train_per_class=20, validation=500, testing=1000,
        hidden_size=1000, encode_size=100):

    ##########################
    # Define Meta-variables  #
    ##########################

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    train_per_class = train_per_class
    validation = validation
    testing = testing
    hidden_size = encode_size
    encode_size = hidden_size
    epochs = epochs
    name = name


    ##########################
    # Load CORA testing Data #
    ##########################
    if dataset == 'cora':
        filepath = "./test/testdata/cora"
        data = loader_cora_torch(filepath=filepath,
                            transform=None,
                            num_train_per_class=train_per_class,
                            num_val=validation,
                            num_test=testing,
                            device=device)

        # Note: we use train/val/test masks to create sub-datasets of edge indices for evaluation
        edge_index, node_features, labels, train_mask, val_mask, test_mask = \
            data.edge_index, data.x, data.y, data.train_mask, data.val_mask, data.test_mask
        print("Data loaded in 'test/testdata' directory! ")

    # TODO: Issue with pubmed - the graph is too big and need to be processed as a sparse matrix
    # elif dataset == 'pubmed':
    #     filepath = "./test/testdata/pubmed"
    #     data = loader_pubmed_torch(filepath=filepath,
    #                         transform=T.ToSparseTensor(),
    #                         num_train_per_class=train_per_class,
    #                         num_val=validation,
    #                         num_test=testing,
    #                         device=device)
    #     # Note: we use train/val/test masks to create sub-datasets of edge indices for evaluation
    #     edge_index, node_features, labels, train_mask, val_mask, test_mask = \
    #         data.edge_index, data.x, data.y, data.train_mask, data.val_mask, data.test_mask
    #     print("Data loaded in 'test/testdata' directory! ")

    elif dataset == 'citeseer':
        filepath = "./test/testdata/citeseer"
        data = loader_citeseer_torch(filepath=filepath,
                            transform=None,
                            num_train_per_class=train_per_class,
                            num_val=validation,
                            num_test=testing,
                            device=device)
        # Note: we use train/val/test masks to create sub-datasets of edge indices for evaluation
        edge_index, node_features, labels, train_mask, val_mask, test_mask = \
            data.edge_index, data.x, data.y, data.train_mask, data.val_mask, data.test_mask
        print("Data loaded in 'test/testdata' directory! ")
    else:
        print("Dataset is not available. Pick one of the two: (1) Cora (2) CiteSeer")
        time.sleep(2)



    ################################
    # Setup Model, Loss, Optimizer #
    ################################
    if task == 'nodeclassification':
        print('Start Node Classification Task (Model: GCN)')
        model = GCN(edge_index=edge_index, input_size=node_features.shape[1], hidden_size_1=hidden_size, encoding_size=encode_size, device=device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
        criterion = nn.CrossEntropyLoss()

        print("Model Initialized!")
    elif task == 'edgeprediction':
        print('Start Edge Prediction Task (Model: GCN-AE)')
        return
    else:
        print("Specified Task is not available. Pick one of the two (1) Node Classification (2) Edge Prediction. ")
    time.sleep(3)

    #####################################
    # Train and Validation on GCN Model #
    #####################################
    print("Start Training!")
    train_loss, val_loss, val_acc = [], [], []

    for epoch in range(epochs):
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
    # training
    parser.add_argument('-n', '--name', type=str, required=True, help='The Name for this Experiment')
    parser.add_argument('-e', '--epochs', default=100, type=int)
    parser.add_argument('-d', '--dataset', default='Cora', type=str, help='Choose between CORA and CITESEER')
    parser.add_argument('-t', '--task', default='nodeclassification', type=str, help='Choose between Node_Classification and Edge_Prediction')
    # model specification
    parser.add_argument('-hs', '--hidden_size', default=1000, type=int)
    parser.add_argument('-es', '--encode_size', default=50, type=int)
    # train, validation, test split
    parser.add_argument('-tr', '--train', type=int, default=20, help="Number of training samples per class.")
    parser.add_argument('-v', '--validation', type=int, default=500, help="Number of validation samples.")
    parser.add_argument('-te', '--test', type=int, default=1000, help="Number of testing samples.")
    args = parser.parse_args()

    name=args.name.replace(' ', '_')
    dataset=args.dataset.lower()
    epochs=args.epochs
    train=args.train
    validation=args.validation
    test = args.test
    hidden_size=args.hidden_size
    encode_size=args.encode_size
    task = re.sub('[^a-zA-Z]', '', args.task)

    print("Name of the Experiment: ", name, '\n')
    print(f"Train with {epochs} # of epoches \n")
    print(f"...with {hidden_size} as hidden layer's size \n")
    print(f"...with {encode_size} as encoding size. \n")

    main(name, dataset, task, epochs, train, validation, test, hidden_size, encode_size)