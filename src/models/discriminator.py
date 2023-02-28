"""
Implementation of the discriminator

Author: Winston Yu
"""

from datetime import datetime
import numpy as np
import networkx as nx
import torch
import torch.nn as nn


# netD is neural network that plays the role of $f$ in Wasserstein GAN's Theorem 3. It is intended
# to be the function such that when plugged into the Kantorovich-Rubinstein formulation for the
# Wasserstein distance (i.e. \mathbf{E}[netD(X)] - \mathbf{E}[netD(G(Z))]), where X is the true
# distribution, Z is the noise distribution, and G is the generator, the formulation returns the
# true Wasserstein distance betweeen X and G(Z).
#
# netD: \mathcal{X} --> \mathbf{R}, \mathcal{X} being the space that the data inhabits
#
# NOTE: How good of a critic netD is highly depends on the statistics that one chooses. Right now, we
# only have "degree histogram" and "clustering coefficients". Surely, more statistics can better
# describe X and whatever the generator produces.


class NetD(nn.Module):
    def __init__(self, stat_input_dim, stat_hidden_dim, num_stat):
        """
        The discriminator. It computes various statistics from its input, each of which is
        processed by a SimpleNN, takes a weighted sum of the processed values, and outputs a scalar
        in (-1,1). The hope is that true samples get mapped to values closer to 1, and that
        generated samples get mapped to values closer to -1. Of course, we want our generator to
        be able to fool the discriminator into outputting 1's for generated samples too.
        """
        super(NetD, self).__init__()
        self.stat_input_dim = stat_input_dim
        self.stat_hidden_dim = stat_hidden_dim
        self.num_stat = num_stat
        self.stat_NNs = nn.ModuleList([
            SimpleNN(self.stat_input_dim, self.stat_hidden_dim)
            for _ in range(self.num_stat)
        ])

        self.combine = nn.Sequential(
            nn.Linear(self.num_stat, 1, bias=False), # bias=False since this should be an "average"
            nn.Tanh()
        )
        return

    def forward(self, G):
        """
        param G: the input graph
        """
        # when computing univariate statistics for x, pass each through an individual NN with
        # tanh as final activation
        # use a NN with tanh as final activation to combine scores
        # possible modifications:
        # 1) use torch.histogram instead of np.histogram?
        # 2) add lots more statistics, these probably aren't enough to characterize a graph
        graph = [nx.from_numpy_matrix(g.detach().numpy()) for g in G]
        degree_hist = np.array([np.histogram(
            np.array(nx.degree_histogram(g)),
            bins=self.stat_input_dim, range=(0.0, 1.0), density=False)[0] for g in graph])
        degree_hist = torch.from_numpy(degree_hist).type(torch.FloatTensor)
        degree_hist = self.stat_NNs[0](degree_hist)
        clustering_coefs = np.array([np.histogram(
            list(nx.clustering(g).values()),
            bins=self.stat_input_dim, range=(0.0, 1.0), density=False)[0] for g in graph])
        clustering_coefs = torch.from_numpy(clustering_coefs).type(torch.FloatTensor)
        clustering_coefs = self.stat_NNs[1](clustering_coefs)
        stats = torch.cat([degree_hist, clustering_coefs], dim=1)
        out = torch.mean(self.combine(stats))
        return out

class SimpleNN(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        """
        A single-layer neural network with tanh activation (motivation: outputs should lie in
        (-1,1)) whose purpose is to process a vector representing a certain statistic computed on
        a graph. The vector is mapped to a scalar (in (-1,1)).

        param input_dim: the size of the aforementioned vector
        param hidden_dim: the size of the hidden-layer's
        """
        super(SimpleNN, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.reduce = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_dim, 1)
        )
        self.act = nn.Tanh()
        return

    def forward(self, x):
        """
        param x: the vector representing a certain statistic
        """
        x = self.reduce(x)
        return self.act(x)

class TestNN(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        """
        A single-layer neural network with tanh activation (motivation: outputs should lie in
        (-1,1)) whose purpose is to process a vector representing a certain statistic computed on
        a graph. The vector is mapped to a scalar (in (-1,1)).

        param input_dim: the size of the aforementioned vector
        param hidden_dim: the size of the hidden-layer's
        """
        super(TestNN, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.reduce = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_dim, 1)
        )
        self.act = nn.Tanh()
        return

    def forward(self, x):
        """
        param x: the vector representing a certain statistic
        """
        x = self.reduce(x)
        return torch.mean(self.act(x))


