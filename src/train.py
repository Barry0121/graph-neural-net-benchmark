"""
Adversarial Attack with Latent Perturbation on Graph Classification Task

This file will train the generator, the inverter, and the critic with specified dataset.
"""
import torch
import torch.optim as optim

from utils.dataset import *
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

def train(dataset_name, noise_dim, clamp_lower=-0.01, clamp_upper=0.01, epochs=10, lr=1e-3, betas=1e-5, batch_size=1, lamb=0.1, loss_func='MSE', device=choose_device()):
    # initialize noise, optimizer and loss
    I = Inverter(input_dim=512, output_dim=noise_dim, hidden_dim=256)
    G_rnn = GRU_plain(input_size=1, embedding_size=64,
                hidden_size=128, num_layers=4, has_input=True,
                has_output=True, output_size=16).to(device)
    G_output = GRU_plain(input_size=1, embedding_size=8,
                           hidden_size=16, num_layers=4, has_input=True,
                           has_output=True, output_size=1).to(device)
    D = NetD(stat_input_dim=128, stat_hidden_dim=64, num_stat=2)

    graph2vec = get_graph2vec(dataset_name) # use infer() to generate new graph embedding
    optimizerI = optim.Adam(i.parameters(), lr=lr).to(device)
    lossI = WGAN_ReconLoss(lamb, loss_func).to(device)
    optimizerD = optim.Adam(D.parameters(), lr=lr, betas=betas).to(device)
    # optimizerG = optim.Adam(G.parameters(), lr=lr, betas=beta).to(device)
    optimizerG_rnn = optim.Adam(list(G_rnn.parameters()), lr=lr, betas=betas)
    optimizerG_output = optim.Adam(list(G_output.parameters()), lr=lr, betas=betas)

    noise = torch.randn(batch_size, noise_dim).to(device)
    one = torch.FloatTensor([1])
    mone = one * -1


    # get the dataset
    train, val, test = get_dataset(dataset_name)
    train_loader = get_dataloader(train, batch_size=64)
    val_loader = get_dataloader(val, batch_size=64)
    test_loader = get_dataloader(test, batch_size=64)

    start_time = time.time()
    for e in epochs:
        # for now, treat the input as adj matrices
        for i, (X, Y) in enumerate(train_loader):
            start=time.time()
            print("====Start Training Discriminator====")

            # enable training
            for p in D.parameters(): # reset requires_grad
                p.requires_grad = True # they are set to False below in netG update

            Diters = 0 # number of iterations to train discriminator
            j = 0 # counter for 1, 2, ... Diters
            while j < Diters and i < len(train_loader):
                j += 1
                # weight clipping: clamp parameters to a cube
                for p in D.parameters():
                    p.data.clamp_(clamp_lower, clamp_upper)
                D.zero_grad()

                # train with real
                input = Y.copy()
                    # insert data processing
                errD_real = D(input)
                errD_real.backward(one) # discriminator should assign 1's to true samples

                # train with fake
                input = noise.resize_(batch_size, 1).normal_(0, 1)
                    # insert data processing
                fake = D(input)
                errD_fake = D(fake)
                errD_fake.backward(mone) # discriminator should assign -1's to fake samples??

                # compute Wasserstein distance and update parameters
                errD = errD_real - errD_fake
                optimizerD.step()

            print(f"====Finished in {(time.time()-start)%60} sec====")

            print("====Start Training Inverter====")
            istart = time.time()
            # graphs
            original_graph = Y
            reconst_graph = G_output(I(original_graph)) # placeholder
            # noise
            reconst_noise = noise
            reconst_noise = I(G_output(noise))
            # compute loss
            loss = lossI(original_graph, reconst_graph, noise, reconst_noise)
            # reset gradients and backprop
            optimizerI.zero_grad()
            loss.backward()
            optimizerI.step()
            print(f"====Finished in {(time.time()-istart)%60} sec====")

            print("====Start Training Generator====")
            gstart = time.time()
            for p in D.parameters():
                p.requires_grad = False # to avoid computation
            # G_rnn.zero_grad()
            G_output.zero_grad()

            input = noise.resize_(batch_size, 1).normal_(0, 1)
                # insert data processing
            fake = G_output(input)
            errG = D(fake) # <- critic's opinion; output of solution f as in WGAN Theorem 3

            # update netG parameters
            errG.backward(one)
            # optimizerG_rnn.step() # TODO: do we need to train the hidden graphRNN?
            optimizerG_output.step()

            gen_iterations += 1
            print(f"====Finished in {(time.time()-gstart)%60} sec====", '\n')

        # Print out training information.
        if (e+1) % 1 == 0:
            elapsed_time = time.time() - start_time
            print('Elapsed time [{:.4f}], Iteration [{}/{}], I Loss: {:.4f}'.format(
                elapsed_time, e+1, epochs, lossI.item()))
    print("====End Training Inverter====")


name = 'MUTAG'
noise_dim = 8
train(name, noise_dim)