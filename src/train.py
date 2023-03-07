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

from models.GAM.src.param_parser import *
from models.GAM.src.gam import *

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


def gam_forward(adj, gam_trainer, target):
    # from gam.py process_graph()
    num_nodes = adj.shape[0]
    nodes = torch.Tensor(range(num_nodes))
    node = nodes[torch.randint(0, num_nodes, (1, 1))[0, 0]]
    degrees = dict(zip([str(n) for n in range(num_nodes)], adj.sum(dim=0).numpy()))
    inv_degrees = {}
    for k, v in degrees.items():
        if str(v) in inv_degrees.keys(): inv_degrees[str(v)].add(int(k))
        else: inv_degrees[str(v)] = {int(k)}
    # print("inv_degrees: ", inv_degrees)
    data = {
        'target': target,
        'edges': None, # already in adjacency matrix
        'labels': degrees, # should be node degrees
        'inverse_labels': inv_degrees # should be dictionary of what degrees correspond to what nodes
    }
    _, features = create_features(data, gam_trainer.model.identifiers, use_graph=False, adj=adj)
    return gam_trainer.model(data=data, adj=adj, features=features, node=node,  get_embedding=True)


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
    gam_trainer_args = parameter_parser()
    # print(netD_args)

    # netD_args['dataset_name'] = args.graph_type
    netI = Inverter(input_dim=512, output_dim=args.hidden_size_rnn, hidden_dim=256)
    netG = GraphRNN(args=args)
    # netG_rnn = netG.rnn
    # netG_output = netG.output
    netD = NetD(stat_input_dim=128, stat_hidden_dim=64, num_stat=2)
    # netD = SimpleNN(621, 1)


    # check model parameters
    # for param in netD.parameters():
    #     print(param.name, param.data, param.requires_grad)
    # for param in netG.parameters():
        # print(param.name, param.data, param.requires_grad)

    graph2vec = get_graph2vec(args.graph_type, dim=512) # use infer() to generate new graph embedding
    optimizerI = optim.Adam(netI.parameters(), lr=lr)
    optimizerD = optim.Adam(gam_trainer.model.parameters(), lr=lr, betas=[betas for _ in range(2)])
    lossI = WGAN_ReconLoss(device, lamb, loss_func)
    # optimizerG = optim.Adam(netG.parameters(), lr=lr)
    G_optimizer_rnn, G_optimizer_output, G_scheduler_rnn, G_scheduler_output = netG.init_optimizer(lr=0.1) # initialize optimizers

    # Check if netG parameters matches with rnn and output's parameters
    # netG_param = list(netG.parameters())
    # rnn_param = list(netG.rnn.parameters())
    # output_param = list(netG.output.parameters())

    noise = torch.randn(args.batch_size, noise_dim).to(device)
    one = torch.tensor(1, dtype=torch.float)
    mone = torch.tensor(-1, dtype=torch.float)

    gen_iterations = 0
    for e in range(args.epochs):
        # for now, treat the input as adj matrices
        start_time = time.time()
        e_errI, e_errD, e_errG, count_batch = 0, 0, 0, 0
        # for i, data in tqdm(enumerate(train_loader), desc=f"Training epoch#{e+1}", total=len(train_loader)):
        for i, data in enumerate(train_loader):
            X = data['x']
            Y = data['y']
            adj_mat = data['adj_mat']
            label = data['label']
            Y_len = data['len']

            # zero grad
            optimizerI.zero_grad()
            optimizerD.zero_grad()
            # optimizerG.zero_grad()
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
            Diters = 10
            j = 0 # counter for 1, 2, ... Diters

            # enable training
            netD.train(True)
            netG.train(False)
            b_errD = 0
            while j < Diters:
                j += 1
                # TODO: commenting this part out for testing
                # weight clipping: clamp parameters to a cube
                # for p in netD.parameters():
                #     p.data.clamp_(clamp_lower, clamp_upper)
                netD.zero_grad()

                # train with real
                inputs = torch.clone(adj_mat)
                D_pred = netD(inputs)
                errD_real = D_pred
                errD_real.backward() # discriminator should assign 1's to true samples
                # print("Error Real: ", errD_real)
                # print(errD_real.requires_grad, errD_real.grad)

                # train with fake
                input = troch.randn(args.batch_size, noise_dim) # (batch_size, hidden_size)
                # insert data processing
                # print(X.shape, Y.shape)
                fake = netG(noise, X, Y, Y_len)
                fake_tensor = netD(fake)
                errD_fake = fake_tensor
                errD_fake.backward(mone) # discriminator should assign -1's to fake samples??

                # compute Wasserstein distance and update parameters
                errD = errD_real - errD_fake
                optimizerD.step()
                # print(D_pred.grad.size(), fake_tensor.grad.size())

                # print("Error Fake: ", errD_fake)

                # print("Error gradient: ", errD_fake.grad, errD_real.grad)

                # print(f"Check if the model is training: iterative value at #{j}.")
                # for p in netD.parameters():
                #     print("Parameters gradients? :", p.requires_grad) # True
                #     print("Parameters values: ", p.data) # values
                #     print("Parameters grad: ", p.grad) # None
                #     break
                # print('\n')

                print(f"Iterative errD {errD.item()}, errD_real {errD_real.item()}, errD_fake {errD_fake.item()}: ")
                b_errD += errD

            # ========== Train Generator ==================
            netD.train(False)
            netG.train(True)
            # netG.clear_gradient_models()
            G_optimizer_rnn.zero_grad()
            G_optimizer_output.zero_grad()
            # optimizerG.zero_grad()
            # in case our last batch was the tail batch of the dataloader,
            # make sure we feed a full batch of noise
            # noisev = noise.normal_(0,1)
            noisev = torch.randn(args.batch_size, noise_dim)
            fake = netG(noisev, X, Y, Y_len)
            fake_tensor = netD(fake)
            errG = fake_tensor
            errG.backward()
            G_optimizer_rnn.step()
            G_optimizer_output.step()
            # netG.all_steps()


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
                with torch.no_grad():
                    G_pred_graphs = netG(I_output, X, Y, Y_len)
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
                iloss = lossI(original_graphs.float(), reconst_graphs.float(), noise.float(), reconst_noise.float()).float()
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
    # torch.save(netD.state_dict(), Dpath)
    torch.save(gam_trainer.model.state_dict(), Dpath) # for GAM netD
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
train(args=args, train_inverter=False)