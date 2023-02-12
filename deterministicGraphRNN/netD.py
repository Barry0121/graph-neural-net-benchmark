from datetime import datetime
import numpy as np
import networkx as nx
import torch
import torch.nn as nn
# import pickle as pkl
# import concurrent.futures

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

class netD(nn.Module):
    def __init__(self, stat_input_dim, stat_hidden_dim, num_stat):
        """
        The discriminator. It computes various statistics from its input, each of which is
        processed by a SimpleNN, takes a weighted sum of the processed values, which can be, 
        interpreted as taking into account each feature's importance, and outputs a scalar in 
        (-1,1). The hope is that true samples get mapped to values closer to 1, and that generated
        samples get mapped to values closer to -1. Of course, we want our generator to be able to 
        fool the discriminator into outputting 1's for generated samples too.

        The structure of netD may be strange, but it is equivalent to a fully connected network
        whose weights between statistics are 0. I think this is a reasonable restriction because I 
        do not expect interactions between the histograms of different statistics.
        """
        super(netD, self).__init__()
        self.stat_input_dim = stat_input_dim
        self.stat_hidden_dim = stat_hidden_dim

        self.stat_NNs = [
            SimpleNN(self.stat_input_dim, self.stat_hidden_dim)
            for _ in range(num_stat)
        ]

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
        # 3) add entropy regularization to reduce the computational complexity?
        #       https://alexhwilliams.info/itsneuronalblog/2020/10/09/optimal-transport/

        # stat 0
        degree_hist, _ = np.histogram( 
            np.array(nx.degree_histogram(G)),
            bins=self.stat_input_dim, range=(0.0, 1.0), density=False)
        degree_hist = self.stat_NNs[0](degree_hist)

        clustering_coefs, _ = np.histogram(
            list(nx.clustering(G).values()),
            bins=self.stat_input_dim, range=(0.0, 1.0), density=False)
        clustering_coefs = self.statNNs[1](clustering_coefs)

        # stat 1
        stats = torch.Tensor([degree_hist, clustering_coefs])
        out = self.combine(stats)

        # stat 2??
        # number of nodes, no NN needed

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

###################################################################################################
#                                                                                                 #
# the code below is only for reference, since netD should just take one graph as input            #
#                                                                                                 # 
###################################################################################################

PRINT_TIME = False

def degree_worker(G):
    return np.array(nx.degree_histogram(G))

def degree_stats(true_graphs, generated_graphs):
    """
    Compute degree histograms for two unordered sets of graphs
    Args:
        graph_ref_list, graph_target_list: two lists of networkx graphs to be evaluated
    """
    true_degree_hists = []
    generated_degree_hists = []
    # remove empty generated graphs
    generated_remove_empty = [G for G in generated_graphs if not G.number_of_nodes() == 0]
    
    prev = datetime.now()
    # compute degree histograms
    for i in range(len(true_graphs)):
        degree_temp = np.array(nx.degree_histogram(true_graphs[i]))
        true_degree_hists.append(degree_temp)
    for i in range(len(generated_remove_empty)):
        degree_temp = np.array(nx.degree_histogram(generated_remove_empty[i]))
        generated_degree_hists.append(degree_temp)
    print(len(true_degree_hists), len(generated_degree_hists))

    elapsed = datetime.now() - prev
    if PRINT_TIME:
        print('Time computing degree stats: ', elapsed)
    return true_degree_hists, generated_degree_hists

def clustering_stats(true_graphs, generated_graphs, bins=100):
    true_clustering = []
    generated_clustering = []
    generated_remove_empty = [G for G in generated_graphs if not G.number_of_nodes() == 0]

    prev = datetime.now()
    for i in range(len(true_graphs)):
        clustering_coeffs_list = list(nx.clustering(true_graphs[i]).values())
        hist, _ = np.histogram(
                clustering_coeffs_list, bins=bins, range=(0.0, 1.0), density=False)
        true_clustering.append(hist)

    for i in range(len(generated_remove_empty)):
        clustering_coeffs_list = list(nx.clustering(generated_remove_empty[i]).values())
        hist, _ = np.histogram(
                clustering_coeffs_list, bins=bins, range=(0.0, 1.0), density=False)
        generated_clustering.append(hist)

    elapsed = datetime.now() - prev
    if PRINT_TIME:
        print('Time computing clustering stats: ', elapsed)
    return true_clustering, generated_clustering