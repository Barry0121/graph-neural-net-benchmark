"""
Adversarial Attack with Latent Perturbation on Graph Classification Task

This file will train the generator, the inverter, and the critic with specified dataset.
"""
import torch
import torch.optim as optim

from utils.dataset import *
from models.critic import *
from models.generator import *
from models.inverter import *


def choose_device():
    if torch.cuda.is_available():
        return 'cuda'
    elif torch.backends.mps.is_available():
        return 'mps'
    else:
        return 'cpu'

def train(dataset, noise_dim, inverter, generator, epochs=10, lr=0.01, batch_size=1, lamb=0.1, loss_func='MSE', device=choose_device()):
    # initialize noise, optimizer and loss
    graph2vec = get_graph2vec(dataset) # use infer() to generate new graph embedding
    i_optimizer = optim.Adam(inverter.parameters(), lr=lr).to(device)
    i_criterion = WGAN_ReconLoss(lamb, loss_func).to(device)
    noise = torch.randn(batch_size, noise_dim).to(device)

    # get the dataset
    graph_loader = None
    # graphs = get_dataset(graphs_name)
    # # get all the embeddings for each graph
    # graphs_embed = get_embeddings(graphs_name)

    start_time = time.time()
    for e in epochs:
        # for now, treat the input as adj matrices
        for j, (adj_mat, _) in enumerate(graph_loader):
            print("====Start Training Discriminator====")


            print("====Start Training Inverter====")
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

            print("====Start Training Generator====")

        # TODO: add loss of all three components
        # Print out training information.
        if (e+1) % 1 == 0:
            elapsed_time = time.time() - start_time
            print('Elapsed time [{:.4f}], Iteration [{}/{}], I Loss: {:.4f}'.format(
                elapsed_time, e+1, epochs, i_criterion.item()))
    print("====End Training Inverter====")

