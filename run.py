# Standard pacakges
import torch
from torch import nn, utils
import numpy as np
import time, re, os, tqdm
import argparse

# Pytroch Geometric
import torch_geometric.transforms as T
from torch_geometric import nn as gnn

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
        filepath = "./data/raw/cora"
        data = loader_cora_torch(filepath=filepath,
                            transform=None,
                            num_train_per_class=train_per_class,
                            num_val=validation,
                            num_test=testing,
                            device=device)

    elif dataset == 'citeseer':
        filepath = "./data/raw/citeseer"
        data = loader_citeseer_torch(filepath=filepath,
                            transform=None,
                            num_train_per_class=train_per_class,
                            num_val=validation,
                            num_test=testing,
                            device=device)
    else:
        filepath = "./data/raw/pubmed"
        data = loader_pubmed_torch(filepath=filepath,
                            transform=None,
                            num_train_per_class=train_per_class,
                            num_val=validation,
                            num_test=testing,
                            device=device)


    if task == 'edgeprediction':
        print('Split data for edge  prediction task')
        data.train_mask = data.val_mask = data.test_mask = None
        data = gutils.train_test_split_edges(data)

    else:
        # Note: we use train/val/test masks to create sub-datasets of edge indices for evaluation
        edge_index, node_features, labels, train_mask, val_mask, test_mask = \
            data.edge_index, data.x, data.y, data.train_mask, data.val_mask, data.test_mask

    print(f"Data loaded in '{filepath}' directory! ")
    time.sleep(2)



    ################################
    # Setup Model, Loss, Optimizer #
    ################################
    if task == 'nodeclassification':
        print('Start Node Classification Task (Model: GCN)')
        model = GCN(input_size=node_features.shape[1], hidden_size_1=hidden_size, encoding_size=encode_size, device=device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
        criterion = nn.CrossEntropyLoss()
        print("Model Initialized!")
    elif task == 'edgeprediction':
        print('Start Edge Prediction Task (Model: GCN-AE)')
        out_channels = 2
        num_features = data.x.shape[1]
        epochs = 100
        # model
        model = gnn.GAE(GCNEncoder(num_features, out_channels))
        # move to GPU (if available)
        model = model.to(device)
        x = data.x.to(device)
        train_pos_edge_index = data.train_pos_edge_index.to(device)
        # inizialize the optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
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
        with tqdm.tqdm(range(epochs), unit="epochs") as tepoch:
            for epoch in (tepoch):
                optimizer.zero_grad()
                out = model(node_features, edge_index)
                loss = criterion(out[train_mask], labels[train_mask])
                loss.backward()
                optimizer.step()

                with torch.no_grad():
                    output = model(node_features, edge_index)
                    vloss = criterion(output[val_mask], labels[val_mask])

                    train_loss.append(loss.cpu().detach().numpy())
                    val_loss.append(vloss.cpu().detach())
                    val_acc.append(np.mean((torch.argmax(output[val_mask], dim=1) == labels[val_mask]).cpu().detach().numpy()))

    elif task == 'edgeprediction':
        def train():
            model.train()
            optimizer.zero_grad()
            z = model.encode(x, train_pos_edge_index)
            loss = model.recon_loss(z, train_pos_edge_index)
            #if args.variational:
            #   loss = loss + (1 / data.num_nodes) * model.kl_loss()
            loss.backward()
            optimizer.step()
            return float(loss)


        def test(pos_edge_index, neg_edge_index):
            model.eval()
            with torch.no_grad():
                z = model.encode(x, train_pos_edge_index)
            return model.test(z, pos_edge_index, neg_edge_index)


        train_loss, val_loss, val_acc = [], [], []
        with tqdm.tqdm(range(epochs), unit="epochs") as tepoch:
            for epoch in tepoch:
                loss = train()
                vauc, vap = test(data.val_pos_edge_index, data.val_neg_edge_index)

                train_loss.append(loss)
                val_loss.append(vap)
                val_acc.append(vauc)

    print("Finish Training!")
    time.sleep(1)

    ##################
    # Test GCN Model #
    ##################
    with torch.no_grad():
        if task == 'nodeclassification':
            pred = model(node_features, edge_index).argmax(dim=1)
            correct = (pred[test_mask] == labels[test_mask]).sum()
            acc = int(correct) / int(test_mask.sum())
            print(f'Test Accuracy: {acc:.4f}')
        elif task == 'edgeprediction':
            tacc, tau = test(data.test_pos_edge_index, data.test_neg_edge_index)
            print(f'Test Accuracy: ', tacc)

    ####################
    # Save Loss Curves #
    ####################
    print("Saving visualizations...")

    save_result(os.path.join('test', 'testresults'), name, train_loss, val_loss, val_acc)
    # save_result(os.path.join('src', 'visualizations'), name, train_loss, val_loss, val_acc)


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
    task = re.sub('[^a-zA-Z]', '', args.task).lower()

    print("Name of the Experiment: ", name, '\n')
    print("Task: ", task)
    print(f"Train with {epochs} epochs \n")
    print(f"...with {hidden_size} as hidden layer's size \n")
    print(f"...with {encode_size} as encoding size. \n")

    main(name, dataset, task, epochs, train, validation, test, hidden_size, encode_size)