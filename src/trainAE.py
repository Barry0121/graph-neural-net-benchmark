"""
Adversarial Attack with Latent Perturbation on Graph Classification Task

This file will train the generator, the inverter, and the critic with specified dataset.
"""
import torch
import torch.optim as optim
import torch.nn.functional as F

from models.args import *
from models.dataset import *
from models.discriminator import *
from models.generator import *
from models.inverter import *
from gw_loss import *

from models.GAM.src.param_parser import *
from models.GAM.src.gam import *

from tqdm import tqdm
import warnings

from ot.gromov import gromov_wasserstein

def fxn():
    warnings.warn("deprecated", UserWarning)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    fxn()

def choose_device():
    if torch.cuda.is_available():
        return 'cuda'
    elif torch.backends.mps.is_available():
        return 'mps'
    else:
        return 'cpu'

# outline for AE:
# 1. sample true graphs (label: 0), generate graphs from GraphRNN (label: 1)
# 2. train GraphAttentionMachine on output of (1)
# 3. sample true graphs, generate graphs from GraphRNN again
# 4. pass those graphs only through embedding layers of GAMachine to get their embeddings
# 5. pass embeddings through GraphRNN
# 6. compare results of (5) to results of (3) via GW distance

args = None

GAMachineTrainer = GAMTrainer(args, args.dataset_name) # maps from graphs to latent space (of embeddings)
netG = None
optimizer_trainer = None
optimizer_generator = None

def train(args, data):
    # TODO: figure out what is "data"

    ### from Barry's code
    X = data['x']
    Y = data['y']
    adj_mat = data['adj_mat']
    label = data['label']
    Y_len = data['len']
    ###

    ###############################################################################################
    # Embedding (GAMTrainer) Update
    ###############################################################################################
    
    GAMachineTrainer.model.train()
    netG.eval()
    for e in range(args.embedding_iters):
        noise = torch.randn(args.embedding_batch_size, args.noise_dim)
        fake_graphs = []
        for b in range(args.generator_batch_size):
            fake_graph = netG(noise[b, :], X, Y, Y_len)
            fake_graphs.append(fake_graph)

        # TODO: unpad (a.k.a. pack) fake_graphs

        true_graphs = [] # TODO: sample from true dataset, e.g. MUTAG

        # from process_batch
        optimizer_trainer.zero_grad()
        batch_loss = 0
        for adj in true_graphs:
            batch_loss = self.process_graph(
                batch_loss=batch_loss, 
                already_matrix=True, 
                adj=adj, target=0 # may be worth passing classes instead of generic "true" label
            )
        for adj in fake_graphs:
            batch_loss = self.process_graph(
                batch_loss=batch_loss, 
                already_matrix=True, 
                adj=adj, target=1
            )
        batch_loss.backward(retain_graph=True)
        optimizer_trainer.step()
        
    ###############################################################################################
    # Generator (GraphRNN) Update
    ###############################################################################################

    GAMachineTrainer.model.eval()
    netG.train()

    ### generate fake graphs from noise
    noise = torch.randn(args.generator_batch_size, args.noise_dim)
    fake_graphs = []
    for b in range(args.generator_batch_size):
        fake_graph = netG(noise[b, :], X, Y, Y_len)
        fake_graphs.append(fake_graph)

    # TODO: unpad (a.k.a. pack) fake_graphs

    ### compute embeddings of fake_graphs, then pass embeddings to generator for reconstruction
    recon_fake_graphs = []
    for fake_adj in fake_graphs:
        datadict, features, node = GAMachineTrainer.get_datadict_features_node(adj, target=1) # target unimportant
        fake_embedding = GAMachineTrainer.model(
            datadict, fake_adj, features, node, get_embedding=True
        )
        recon_graph = netG(fake_embedding, X, Y, Y_len)
        recon_fake_graphs.append(recon_graph)

    ### do same for true graphs
    true_graphs = [] # TODO: sample from true dataset, e.g. MUTAG
    
    # compute embeddings of true_graphs, then pass embeddings to generator for reconstruction
    recon_true_graphs = []
    for true_adj in true_graphs:
        datadict, features, node = GAMachineTrainer.get_datadict_features_node(adj, target=0)
        true_embedding = GAMachineTrainer.model(
            datadict, true_adj, features, node, get_embedding=True
        )
        recon_graph = netG(true_embedding, X, Y, Y_len)
        recon_true_graphs.append(recon_graph)

    batch_gw_loss = 0
    # compute GW distance on true_graphs/fake_graphs and recon_true_graphs/recon_fake_graphs
    for adj_o, adj_r in zip(true_graphs, recon_true_graphs):
        batch_gw_loss += GWLoss(adj_o, adj_r)
    for adj_o, adj_r in zip(fake_graphs, recon_fake_graphs):
        batch_gw_loss += GWLoss(adj_o, adj_r)
    batch_gw_loss.backward()
    optimizer_generator.step()

    return


args = Args()
# ===============Test BFS DataLoader==================
# graphs, labels = get_dataset_with_label('MUTAG')
# dataset = Graph_sequence_sampler_pytorch(graphs, labels, args=args)
# dataloader = get_dataloader_labels(dataset, args)
# for i, data in enumerate(dataloader):
#     print(i)
#     print(data['x'].shape)
#     print(data['y'].shape)
#     print(data['adj_mat'].shape)
#     print(data['label'].shape)
#     print(data['len'].shape)
#     break

# ===============Test training function================
train(args=args, train_inverter=False)