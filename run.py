# Standard pacakges
import torch
from torch import nn, utils
import numpy as np
import time, re, os, tqdm
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

    elif dataset == 'citeseer':
        filepath = "./test/testdata/citeseer"
        data = loader_citeseer_torch(filepath=filepath,
                            transform=None,
                            num_train_per_class=train_per_class,
                            num_val=validation,
                            num_test=testing,
                            device=device)
    else:
        print("Dataset is not available. Pick one of the two: (1) Cora (2) CiteSeer")


    if task == 'edgeprediction':
        print('Split data for edge  prediction task')
        transform = T.RandomLinkSplit(num_val=0.05, num_test=0.1, is_undirected=True,
                                    split_labels=True, add_negative_train_samples=False)
        train_data, val_data, test_data = transform(data)

    else:
        # Note: we use train/val/test masks to create sub-datasets of edge indices for evaluation
        edge_index, node_features, labels, train_mask, val_mask, test_mask = \
            data.edge_index, data.x, data.y, data.train_mask, data.val_mask, data.test_mask

    print("Data loaded in 'test/testdata' directory! ")
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
        model = GCN_AE(edge_index=train_data.edge_index, input_size=train_data.x.shape[1], hidden_size_1=hidden_size, hidden_size_2=(hidden_size+encode_size)//2, encoding_size=encode_size, device=device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
        criterion = nn.CrossEntropyLoss()
        print("Model Initialized!")
    else:
        print("Specified Task is not available. Pick one of the two (1) Node Classification (2) Edge Prediction. ")
    time.sleep(3)

    #####################################
    # Train and Validation on GCN Model #
    #####################################
    print("Start Training!")
    if task == 'nodeclassification':
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

    elif task == 'edgeprediction':
        train_loss, val_loss, val_acc = [], [], []
        with tqdm.tqdm(range(epochs), unit="batch") as tepoch:
            for epoch in tepoch:
                # train
                model.train()
                optimizer.zero_grad()
                model.edge_index = train_data.edge_index
                output = model.encoder(train_data.x)
                output = gutils.dense_to_sparse(output)[0]
                print(output.shape)
                loss = criterion(output.float(), train_data.pos_edge_label_index)
                loss.backward()
                optimizer.step()

                # calculate accuracy
                model.eval()
                model.edge_index = val_data.edge_index
                output = model(val_data.x)
                output = gutils.dense_to_sparse(output)[0]
                vloss = criterion(output, val_data.pos_edge_label_index)
                accuracy = torch.mean(output == val_data.pos_edge_label_index, dtype=torch.float).item()

                train_loss.append(loss)
                val_loss.append(vloss)
                val_acc.append(accuracy)

    print("Finish Training!")
    time.sleep(1)

    ##################
    # Test GCN Model #
    ##################
    if task == 'nodeclassification':
        model.eval()
        pred = model(node_features).argmax(dim=1)
        correct = (pred[test_mask] == labels[test_mask]).sum()
        acc = int(correct) / int(test_mask.sum())
        print(f'Test Accuracy: {acc:.4f}')
    elif task == 'edgeprediction':
        model.eval()
        model = test_data.edge_index
        pred = model(test_data.x)
        pred = gutils.dense_to_sparse(pred)[0]
        acc = (pred == test_data.pos_edge_label_index).sum() / test_data.pos_edge_label_index.shape[1]
        print(f'Test Accuracy: {acc:.4f}')

    ####################
    # Save Loss Curves #
    ####################
    # print("Saving visualizations...")

    # save_result(os.path.join('test', 'testresults'), name, train_loss, val_loss, val_acc)


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
    print(f"Train with {epochs} epochs \n")
    print(f"...with {hidden_size} as hidden layer's size \n")
    print(f"...with {encode_size} as encoding size. \n")

    main(name, dataset, task, epochs, train, validation, test, hidden_size, encode_size)