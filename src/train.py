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


def train(args, num_layers=4, clamp_lower=-0.01, clamp_upper=0.01, epochs=10, lr=1e-3, betas=1e-5,lamb=0.1, loss_func='MSE', device=choose_device()):
    # save losses
    iloss_lst = []
    dloss_lst = []
    gloss_lst = []

    # get the dataset
    train, labels = get_dataset_with_label(args.graph_type) # entire dataset as train
    train_dataset = Graph_sequence_sampler_pytorch(train, labels, args)
    train_loader = get_dataloader_labels(train_dataset, args)
    noise_dim = args.hidden_size_rnn
    print('noise dimension is: ', noise_dim)

    # initialize noise, optimizer and loss
    I = Inverter(input_dim=512, output_dim=args.hidden_size_rnn, hidden_dim=256)
    G = GraphRNN(args=args)
    D = NetD(stat_input_dim=128, stat_hidden_dim=64, num_stat=2)

    graph2vec = get_graph2vec(args.graph_type, dim=512) # use infer() to generate new graph embedding
    optimizerI = optim.Adam(I.parameters(), lr=lr)
    optimizerD = optim.Adam(D.parameters(), lr=lr, betas=[betas for _ in range(2)])
    lossI = WGAN_ReconLoss(device, lamb, loss_func)
    G.init_optimizer() # initialize optimizers


    noise = torch.randn(args.batch_size, noise_dim).to(device)
    one = torch.FloatTensor([1])
    mone = one * -1

    start_time = time.time()
    for e in range(epochs):
        # for now, treat the input as adj matrices

        for i, data in enumerate(train_loader):
            X = data['x']
            Y = data['y']
            adj_mat = data['adj_mat']
            label = data['label']
            Y_len = data['len']

            start=time.time()
            # print("====Start Training Discriminator====")

            # enable training
            for p in D.parameters(): # reset requires_grad
                p.requires_grad = True # they are set to False below in netG update

            Diters = 1 # number of iterations to train discriminator
            j = 0 # counter for 1, 2, ... Diters
            while j < Diters and i < len(train_loader):
                j += 1
                # weight clipping: clamp parameters to a cube
                for p in D.parameters():
                    p.data.clamp_(clamp_lower, clamp_upper)
                D.zero_grad()

                # train with real
                inputs = torch.torch.empty_like(adj_mat).copy_(adj_mat)
                # print(inputs.shape)
                input_graphs = [nx.from_numpy_matrix(i) for i in inputs.detach().numpy()] # TODO: Error raise NetworkXError(f"Edge tuple {e} must be a 2-tuple or 3-tuple.")
                D_pred = torch.Tensor([D(graph) for graph in input_graphs])
                # print(D_pred.requires_grad)
                errD_real = torch.mean(D_pred)
                # print(errD_real)
                # errD_real.backward(one) # discriminator should assign 1's to true samples
                # errD_real.backward()

                # train with fake
                input = noise.normal_(0,1) # (batch_size, hidden_size)
                # insert data processing
                fake = G.generate(input, args, test_batch_size=args.batch_size)
                fake_tensor = torch.Tensor([D(nx.from_numpy_matrix(f)) for f in fake.detach().numpy()])
                errD_fake = torch.mean(fake_tensor)
                # errD_fake.backward(mone) # discriminator should assign -1's to fake samples??
                # errD_fake.backward()

                # compute Wasserstein distance and update parameters
                errD = Variable(errD_real - errD_fake, requires_grad=True)
                errD.backward()
                optimizerD.step()

            # print(f"====Finished in {(time.time()-start)%60} sec====")

            # print("====Start Training Inverter and Generator====")
            G.train()
            G.clear_gradient_opts()
            G.clear_gradient_models()
            I.zero_grad()
            istart = time.time()
            # graphs
            original_graphs = adj_mat # shape: (batch_size, padded_size, padded_size); in the case for MUTAG, padded_size is 29
            graph_lst = [nx.from_numpy_matrix(am.detach().numpy()) for am in adj_mat]
            embeddings = torch.Tensor(graph2vec.infer(graph_lst))
            I_output = I(torch.reshape(embeddings, (embeddings.shape[0], -1)))
            # print(I_output.shape)
            G_pred_graphs = G.generate(X=I_output, args=args, test_batch_size=args.batch_size)
            reconst_graphs = G_pred_graphs
            # noise
            G_pred_noise = G.generate(X=noise, args=args, test_batch_size=args.batch_size) # shape: (batch_size, padded_size, padded_size)
            # print(G_pred_noise.shape)
            noise_graph_lst = [nx.from_numpy_matrix(am.detach().numpy()) for am in G_pred_noise]
            noise_embeddings = torch.Tensor(graph2vec.infer(noise_graph_lst))
            reconst_noise = I(noise_embeddings)
            # compute loss and update inverter loss
            original_graphs = original_graphs.to(device)
            reconst_graphs = reconst_graphs.to(device)
            noise = noise.to(device)
            reconst_noise = reconst_noise.to(device)
            iloss = lossI(original_graphs, reconst_graphs, noise, reconst_noise)
            optimizerI.zero_grad()
            iloss.backward()
            optimizerI.step()
            # compute loss and update generator loss
            errG = Variable(torch.mean(torch.Tensor([D(nx.from_numpy_matrix(g)) for g in reconst_graphs.cpu().detach().numpy()])), requires_grad=True).to(device)
            errG.backward()
            G.all_steps()
            # print(f"====Finished in {(time.time()-istart)%60} sec====")

        # Print out training information per epoch.
        if (e+1) % 1 == 0:
            elapsed_time = time.time() - start_time
            print('Elapsed time [{:.4f}], Iteration [{}/{}], I Loss: {:.4f}, D Loss: {:.4f}, G Loss {:.4f}'.format(
                elapsed_time, e+1, epochs, iloss.item(), errD, errG))

    # save training loss across
    iloss_lst.append(iloss.item())
    dloss_lst.append(errD)
    gloss_lst.append(errG)
    np.savetxt('./cache/graphrnn/loss_results/inverter_loss.txt', iloss_lst, delimiter=',')
    np.savetxt('./cache/graphrnn/loss_results/discriminator_loss.txt', dloss_lst, delimiter=',')
    np.savetxt('./cache/graphrnn/loss_results/generator_loss.txt', gloss_lst, delimiter=',')

    # save models
    Gpath = './cache/graphrnn/saved_model/generator.pth'
    Ipath = './cache/graphrnn/saved_model/inverter.pth'
    Dpath = './cache/graphrnn/saved_model/discriminator.pth'
    torch.save(G.state_dict(), Gpath)
    torch.save(I.state_dict(), Ipath)
    torch.save(D.state_dict(), Dpath)
    print("====End of Training====")



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
train(args=args)