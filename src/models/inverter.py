import torch.nn as nn
import torch.optim as optim
# import GW distance

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
        param embedding: the graph2vec embedding
        """
        x = embedding.clone().detach()
        # x = graph2vec(x)
        x = self.layer1(x)
        x = nn.ReLU(x)
        x = self.layer2(x)
        return x

def train(inverter, epochs, lr, batch_size):
    optimizer = optim.Adam(inverter.parameters(), lr=lr)

    # optimize equation 2 from Generating Natural Adversarial Examples
    return