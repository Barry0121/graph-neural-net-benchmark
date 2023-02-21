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

from tqdm import tqdm
import warnings

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


def train(args, train_inverter=False, num_layers=4, clamp_lower=-0.1, clamp_upper=0.1, lr=1e-3, betas=1e-5, lamb=0.1, loss_func='MSE', device=choose_device()):
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
    netI = Inverter(input_dim=512, output_dim=args.hidden_size_rnn, hidden_dim=256)
    netG = GraphRNN(args=args)
    netD = NetD(stat_input_dim=128, stat_hidden_dim=64, num_stat=2)

    # check model parameters
    # for param in netD.parameters():
    #     print(param.name, param.data, param.requires_grad)
    # for param in netG.parameters():
        # print(param.name, param.data, param.requires_grad)

    graph2vec = get_graph2vec(args.graph_type, dim=512) # use infer() to generate new graph embedding
    optimizerI = optim.Adam(netI.parameters(), lr=lr)
    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=[betas for _ in range(2)])
    lossI = WGAN_ReconLoss(device, lamb, loss_func)
    G_optimizer_rnn, G_optimizer_output, G_scheduler_rnn, G_scheduler_output = netG.init_optimizer(lr=0.1) # initialize optimizers


    noise = torch.randn(args.batch_size, noise_dim).to(device)
    one = torch.tensor(1, dtype=torch.float)
    mone = torch.tensor(-1, dtype=torch.float)

    gen_iterations = 0
    for e in range(args.epochs):
        # for now, treat the input as adj matrices
        start_time = time.time()
        e_errI, e_errD, e_errG, count_batch = 0, 0, 0, 0
        for i, data in tqdm(enumerate(train_loader), desc=f"Training epoch#{e+1}", total=len(train_loader)):
            X = data['x']
            Y = data['y']
            adj_mat = data['adj_mat']
            label = data['label']
            Y_len = data['len']
            # zero grad
            optimizerI.zero_grad()
            optimizerD.zero_grad()
            G_optimizer_rnn.zero_grad()
            G_optimizer_output.zero_grad()
            # netG.clear_gradient_opts()
            # netG.clear_gradient_models()
            # skip uneven batch
            if adj_mat.size(0) != args.batch_size:
                continue

            ######################
            # Discriminator Update
            ######################
            # number of iteration to train the discriminator
            if gen_iterations < 25 or gen_iterations % 500 == 0:
                Diters = 20
            else:
                Diters = 5
            j = 0 # counter for 1, 2, ... Diters

            # enable training
            # for p in netD.parameters(): # reset requires_grad
            #     p.requires_grad = True # they are set to False below in netG update
            netD.train(True)
            # for p in netG.parameters():
            #     p.requires_grad = False
            netG.train(False)
            # print("Check if the model is training: initial value.")
            # for p in netD.parameters():
            #     print("Parameters gradients? :", p.requires_grad)
            #     print("Parameters values: ", p.data)
            #     print("Parameters grad: ", p.grad)
            b_errD = 0
            while j < Diters:

                j += 1
                # weight clipping: clamp parameters to a cube
                # for p in netD.parameters():
                #     p.data.clamp_(clamp_lower, clamp_upper)
                netD.zero_grad()

                # train with real
                inputs = torch.torch.empty_like(adj_mat).copy_(adj_mat)
                input_graphs = [nx.from_numpy_matrix(i) for i in inputs.numpy()]
                D_pred = torch.Tensor([netD(graph) for graph in input_graphs])
                errD_real = Variable(torch.mean(D_pred), requires_grad=True)
                errD_real.backward(one) # discriminator should assign 1's to true samples


                # train with fake
                input = noise.normal_(0,1) # (batch_size, hidden_size)
                # insert data processing
                fake = netG(input, args, output_batch_size=args.batch_size)
                fake_tensor = torch.Tensor([netD(nx.from_numpy_matrix(f)) for f in fake.numpy()])
                errD_fake = Variable(torch.mean(fake_tensor), requires_grad=True)
                errD_fake.backward(mone) # discriminator should assign -1's to fake samples??

                # compute Wasserstein distance and update parameters
                errD = Variable(errD_real - errD_fake, requires_grad=False)

                # print(f"Check if the model is training: iterative value at #{j}.")
                # for p in netD.parameters():
                #     print("Parameters gradients? :", p.requires_grad)
                #     print("Parameters values: ", p.data)
                #     print("Parameters grad: ", p.grad)


                optimizerD.step()
                print(f"Iterative errD {errD.item()}, errD_real {errD_real.item()}, errD_fake {errD_fake.item()}: ")
                b_errD += errD

            # ========== Train Generator ==================
            # for p in netD.parameters():
            #     p.requires_grad = False # to avoid computation
            # for p in netG.parameters():
            #     p.requires_grad = True
            netD.train(False)
            netG.train(True)
            netG.clear_gradient_models()
            G_optimizer_rnn.zero_grad()
            G_optimizer_output.zero_grad()
            # in case our last batch was the tail batch of the dataloader,
            # make sure we feed a full batch of noise
            noisev = Variable(noise.normal_(0,1))
            fake = netG(noisev, args=args, output_batch_size=args.batch_size)
            # fake = netG(noisev, X, Y, Y_len)[0]
            fake_tensor = torch.Tensor([netD(nx.from_numpy_matrix(f)) for f in fake.detach().numpy()])
            errG = Variable(torch.mean(fake_tensor), requires_grad=True)
            errG.backward(one)
            G_optimizer_rnn.step()
            G_optimizer_output.step()
            # netG.all_steps()
            gen_iterations += 1


            # Winston's outline for inverter training
            # 0. train graph2vec on {generator distribution, true distribution}
            # 1. sample true samples x
            # 2. recon_graph = netG(netI(x))
            # 3. sample noise z
            # 4. recon_noise = netI(netG(z))
            # 5. minimize GW_dist(recon_graph - x) + lambda * MSE(recon_noise - z)

            # ========== Train Inverter =================
            # TODO: fix variables, move this into a different training loop
            if train_inverter:
                original_graphs = adj_mat # shape: (batch_size, padded_size, padded_size); in the case for MUTAG, padded_size is 29
                graph_lst = [nx.from_numpy_matrix(am.detach().numpy()) for am in adj_mat]
                # retrain graph2vec
                graph2vec.fit(graph_lst)
                # genearte embedding
                embeddings = torch.Tensor(graph2vec.infer(graph_lst))
                I_output = netI(torch.reshape(embeddings, (embeddings.shape[0], -1)))
                # print(I_output.shape)
                G_pred_graphs = netG(X=I_output, args=args, output_batch_size=args.batch_size)
                reconst_graphs = G_pred_graphs
                # noise
                G_pred_noise = netG(X=noise, args=args, output_batch_size=args.batch_size) # shape: (batch_size, padded_size, padded_size)
                # print(G_pred_noise.shape)
                noise_graph_lst = [nx.from_numpy_matrix(am.detach().numpy()) for am in G_pred_noise]
                noise_embeddings = torch.Tensor(graph2vec.infer(noise_graph_lst))
                reconst_noise = netI(noise_embeddings)
                # compute loss and update inverter loss
                original_graphs = original_graphs.to(device)
                reconst_graphs = reconst_graphs.to(device)
                noise = noise.to(device)
                reconst_noise = reconst_noise.to(device)
                iloss = lossI(original_graphs, reconst_graphs, noise, reconst_noise)
                iloss.backward()
                optimizerI.step()




            # # compute mean error across all batches
            e_errD += b_errD.item()
            e_errG += errG.item()
            if train_inverter:
                e_errI += iloss.item()
            count_batch += 1

        # Print out training information per epoch.
        if train_inverter:
            if (e+1) % 1 == 0:
                elapsed_time = time.time() - start_time
                print('Elapsed time [{:.4f}], Iteration [{}/{}], I Loss: {:.4f}, D Loss: {:.4f}, G Loss {:.4f}'.format(
                    elapsed_time, e+1, args.epochs, e_errI/count_batch, e_errD/count_batch, e_errG/count_batch))
        else:
            if (e+1) % 1 == 0:
                elapsed_time = time.time() - start_time
                print('Elapsed time [{:.4f}], Iteration [{}/{}], D Loss: {:.4f}, G Loss {:.4f}'.format(
                    elapsed_time, e+1, args.epochs, e_errD/count_batch, e_errG/count_batch))


        # append training loss across
        if train_inverter:
            iloss_lst.append(e_errI/count_batch)
        dloss_lst.append(e_errD/count_batch)
        gloss_lst.append(e_errG/count_batch)

    # save loss
    if train_inverter:
        np.savetxt('./cache/graphrnn/loss_results/inverter_loss.txt', iloss_lst, delimiter=',')
    np.savetxt('./cache/graphrnn/loss_results/discriminator_loss.txt', dloss_lst, delimiter=',')
    np.savetxt('./cache/graphrnn/loss_results/generator_loss.txt', gloss_lst, delimiter=',')

    # save models
    Gpath = './cache/graphrnn/saved_model/generator.pth'
    Dpath = './cache/graphrnn/saved_model/discriminator.pth'
    torch.save(netG.state_dict(), Gpath)
    torch.save(netD.state_dict(), Dpath)
    if train_inverter:
        Ipath = './cache/graphrnn/saved_model/inverter.pth'
        torch.save(netI.state_dict(), Ipath)

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