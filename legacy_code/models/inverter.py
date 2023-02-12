import torch
import torch.nn as nn
import torch.optim as optim
import time
from scipy.stats import wasserstein_distance
from karateclub import Graph2Vec
from torch_geometric.datasets import TUDataset
from torch_geometric.utils import to_networkx

import numpy as np
import networkx as nx
import ot # Python Optimal Transport
from ot.gromov import gromov_wasserstein

class Inverter(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim):
        """
        Literally just a single-layer network. It takes (fixed-dimension!) embeddings and tries
        to be an inverse to the generator and to enforce a certain distribution (probably normal)
        on the latent space.

        param input_dim: dimension of graph2vec embedding
        param hidden_dim: dimension of hidden layer
        param output_dim: dimension of actual embedding; will be input dimension for generator
        """
        super(Inverter, self).__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, embedding):
        """
        Parameter:
            Embedding: one or a list of graph embeddings trained
        Return:
            Reconstructed graph from using the embedding.
        """
        x = embedding.clone().detach()
        x = self.layer1(x)
        x = nn.ReLU(x)
        x = self.layer2(x)
        return x

def choose_device():
    if torch.cuda.is_available():
        return 'cuda'
    elif torch.backends.mps.is_available():
        return 'mps'
    else:
        return 'cpu'

class WGAN_ReconLoss(nn.Module):
    def __init__(self, lamb: float, loss_func: str='MSE', device=choose_device()) -> None:
        super().__init__()
        self.lamb = lamb
        self.device = device
        if loss_func == 'MSE':
            self.L = nn.MSELoss(reduction='mean').to(self.device)
        elif loss_func == 'CrossEntropy':
            self.L = nn.CrossEntropyLoss().to(self.device)
        else:
            print("Loss function must be 'MSE' or 'CrossEntropy'.")

    def forward(self, x_original, x_reconst, z_original, z_reconst, use_gw1=True):
        """
        From the Generating Natural Adv. Example Paper:
            x_original: original image/graph example
            x_reconst: reconstruction of that image/graph from the generator
            z_original: noise or learned latent feature passed into the inverter
            z_reconst: reconstruction of the noise, output of the inverter
        """
        if use_gw1:
            return (self.L(x_original, x_reconst) \
                + self.lamb * self.L(z_original, z_reconst)).to(self.device)
        else:            
            # initialize measures for each metric measure network
            p_o = ot.unif(x_original.number_of_nodes())
            p_r = ot.unif(x_reconst.number_of_nodes())
            # get adjacency matrices
            C_o = np.asarray(nx.adjacency_matrix(x_original).todense())
            C_r = np.asarray(nx.adjacency_matrix(x_reconst).todense())
            _, log = gromov_wasserstein(
                C_o, C_r, p_o, p_r, 
                'square_loss', verbose=False, log=False)

            return (log['gw_dist'] + (self.lamb * self.L(z_original, z_reconst)).to(self.device))

def train(graphs_name, noise_dim, inverter, generator, epochs=10, lr=0.01, batch_size=1, lamb=0.1, loss_func='MSE', device= choose_device()):
    # initialize noise, optimizer and loss
    i_optimizer = optim.Adam(inverter.parameters(), lr=lr).to(device)
    i_criterion = WGAN_ReconLoss(lamb, loss_func).to(device)
    noise = torch.randn(batch_size, noise_dim).to(device)
    graphs = TUDataset(root='../data/raw/TUDataset', name=graphs_name)

    # initialize Graph2Vec model
    graph2vec = Graph2Vec(wl_iterations=2,
                        attributed=False,
                        dimensions=128,
                        down_sampling=0.0001,
                        workers=-1,
                        epochs=10)

    # train graph2vec with dataset
    graphs_nx = [to_networkx(data) for data in graphs]
    graph2vec.fit(graphs_nx)

    # get all the embeddings for each graph
    graphs_embed = torch.Tensor(graph2vec.get_embedding()).to(device)

    start_time = time.time()
    print("====Start Training Inverter====")
    for e in epochs:
        # for now, treat the input as adj matrices
        for j, (adj_mat, _) in enumerate(graph_loader):
            # graphs
            original_graph = adj_mat
            reconst_graph = adj_mat # placeholder
            # TODO: once we get generator, change with this line
            # reconst_graph = generator(inverter(original_graph))

            # noise
            reconst_noise = noise
            # TODO: once we get generator, uncomment this line
            # reconst_noise = inverter(generator(noise))

            # compute loss
            loss = i_criterion(original_graph, reconst_graph, noise, reconst_noise)

            # reset gradients
            i_optimizer.zero_grad()
            i_criterion.backward()
            i_optimizer.step()
        # Print out training information.
        if (e+1) % 1 == 0:
            elapsed_time = time.time() - start_time
            print('Elapsed time [{:.4f}], Iteration [{}/{}], I Loss: {:.4f}'.format(
                elapsed_time, e+1, epochs, i_criterion.item()))
    print("====End Training Inverter====")
