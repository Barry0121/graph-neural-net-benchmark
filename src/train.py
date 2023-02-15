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


def choose_device():
    if torch.cuda.is_available():
        return 'cuda'
    elif torch.backends.mps.is_available():
        return 'mps'
    else:
        return 'cpu'


def train(dataset_name, noise_dim, args, num_layers=4, clamp_lower=-0.01, clamp_upper=0.01, epochs=10, lr=1e-3, betas=1e-5, batch_size=1, lamb=0.1, loss_func='MSE', device=choose_device()):
    # get the dataset
    train, labels = get_dataset_with_label(dataset_name) # entire dataset as train
    train_dataset = Graph_sequence_sampler_pytorch_nobfs(train, labels, args)
    train_loader = get_dataloader_labels(train_dataset, args)

    # initialize noise, optimizer and loss
    I = Inverter(input_dim=512, output_dim=noise_dim, hidden_dim=256)
    G = GraphRNN(args=args)
    D = NetD(stat_input_dim=128, stat_hidden_dim=64, num_stat=2)

    graph2vec = get_graph2vec(dataset_name) # use infer() to generate new graph embedding
    optimizerI = optim.Adam(I.parameters(), lr=lr)
    optimizerD = optim.Adam(D.parameters(), lr=lr, betas=[betas for _ in range(2)])
    lossI = WGAN_ReconLoss(lamb, loss_func).to(device)
    G.init_optimizer() # initialize optimizers


    noise = torch.randn(batch_size, noise_dim).to(device)
    one = torch.FloatTensor([1])
    mone = one * -1

    start_time = time.time()
    for e in range(epochs):
        # for now, treat the input as adj matrices

        for i, data in enumerate(train_loader):
            X = data['x']
            Y = data['y']
            label = data['label']
            Y_len = data['len']

            start=time.time()
            print("====Start Training Discriminator====")

            # enable training
            for p in D.parameters(): # reset requires_grad
                p.requires_grad = True # they are set to False below in netG update

            Diters = 10 # number of iterations to train discriminator
            j = 0 # counter for 1, 2, ... Diters
            while j < Diters and i < len(train_loader):
                j += 1
                # weight clipping: clamp parameters to a cube
                for p in D.parameters():
                    p.data.clamp_(clamp_lower, clamp_upper)
                D.zero_grad()

                # train with real
                inputs = torch.torch.empty_like(Y).copy_(Y)
                print(Y[0].shape)
                input_graphs = [nx.from_edgelist(i) for i in inputs.detach().numpy()]
                print(input_graphs[0])
                errD_real = D(input_graphs)
                errD_real.backward(one) # discriminator should assign 1's to true samples

                # train with fake
                input = noise.resize_(batch_size, 1).normal_(0, 1)
                # insert data processing
                fake = G(input)
                errD_fake = D(fake)
                errD_fake.backward(mone) # discriminator should assign -1's to fake samples??

                # compute Wasserstein distance and update parameters
                errD = errD_real - errD_fake
                errD.backward()
                optimizerD.step()

            print(f"====Finished in {(time.time()-start)%60} sec====")

            print("====Start Training Inverter and Generator====")
            G.train()
            G.clear_gradient_opts()
            G.clear_gradient_models()
            I.zero_grad()
            istart = time.time()
            # graphs
            original_graph = Y
            G_pred_graph = G(X=I(original_graph), Y=original_graph, length=Y_len)
            reconst_graph = G_pred_graph[0]  # 0 for prediction, 1 for sorted output
            # noise
            G_pred_noise = G.generate(X=noise, test_batch_size=args.test_batch_size)
            reconst_noise = I(G_pred_noise[0])
            # compute loss and update inverter loss
            loss = lossI(original_graph, reconst_graph, noise, reconst_noise)
            optimizerI.zero_grad()
            loss.backward()
            optimizerI.step()
            # compute loss and update generator loss
            errG = torch.mean(G(reconst_graph))
            errG.backward()
            G.all_steps()
            print(f"====Finished in {(time.time()-istart)%60} sec====")


        # Print out training information.
        if (e+1) % 1 == 0:
            elapsed_time = time.time() - start_time
            print('Elapsed time [{:.4f}], Iteration [{}/{}], I Loss: {:.4f}'.format(
                elapsed_time, e+1, epochs, lossI.item()))
    print("====End of Training====")


name = 'MUTAG'
noise_dim = 8 # TODO: change this
args = Args()
graphs, labels = get_dataset_with_label('MUTAG')
dataset = Graph_sequence_sampler_pytorch_nobfs(graphs, labels, args=args)
dataloader = get_dataloader_labels(dataset)
data = iter(dataloader)
print(next(data))

# train(name, noise_dim, args=args)