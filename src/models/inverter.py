"""
Implementation of the inverter and inverter loss
"""
import torch
import torch.nn as nn
from scipy.stats import wasserstein_distance
import time

def choose_device():
    if torch.cuda.is_available():
        return 'cuda'
    elif torch.backends.mps.is_available():
        return 'mps'
    else:
        return 'cpu'

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
            return (self.L(x_original, x_reconst) + self.lamb * self.L(z_original, z_reconst)).to(self.device)
        else:
            return (wasserstein_distance(x_original, x_reconst) + self.lamb * self.L(z_original, z_reconst)).to(self.device)



