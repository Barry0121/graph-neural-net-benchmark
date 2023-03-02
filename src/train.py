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
    train_loader, adj_shape = get_dataloader_labels(train_dataset, args)
    noise_dim = args.hidden_size_rnn
    print('noise dimension is: ', noise_dim)
    print('padded adjacency matrix shape is: ', adj_shape)

    # initialize noise, optimizer and loss
    gam_trainer_args = parameter_parser()
    # print(netD_args)

    # netD_args['dataset_name'] = args.graph_type
    netI = Inverter(input_dim=512, output_dim=args.hidden_size_rnn, hidden_dim=256)
    netG = GraphRNN(args=args)
    # netG_rnn = netG.rnn
    # netG_output = netG.output
    # netD = NetD(stat_input_dim=128, stat_hidden_dim=64, num_stat=2)
    netD = TestNN(gam_trainer_args.combined_dimensions, 1) # matching dimension of Graph2Vec
    gam_trainer = GAMTrainer(gam_trainer_args, args.graph_type)

    # ======Testing=======
    # set up a register_hook to check parameter gradient)
    # hd = list(netD.model.parameters())[0].register_hook(lambda grad: print(f"NetD parameter Update.."))
    hg = list(netG.parameters())[0].register_hook(lambda grad: print(f"NetG parameter Update.."))
    # if train_inverter:
    #     hi = list(netI.parameters())[0].register_hook(lambda grad: print(f"NetI parameter Update.."))

    # ======Testing=======
    # check model parameters
    # for param in netD.parameters():
    #     print(param.name, param.data, param.requires_grad)
    #     break
    # for param in netG.parameters():
    #     print(param.name, param.data, param.requires_grad)
    #     break

    graph2vec = get_graph2vec(args.graph_type, dim=512) # use infer() to generate new graph embedding
    optimizerI = optim.Adam(netI.parameters(), lr=lr)
    optimizerD = optim.Adam(gam_trainer.model.parameters(), lr=lr, betas=[betas for _ in range(2)])
    lossI = WGAN_ReconLoss(device, lamb, loss_func)
    G_optimizer_rnn, G_optimizer_output, G_scheduler_rnn, G_scheduler_output = netG.init_optimizer(lr=0.1) # initialize optimizers

    # ======Testing=======
    # Check if netG parameters matches with rnn and output's parameters
    # netG_param = list(netG.parameters())
    # rnn_param = list(netG.rnn.parameters())
    # output_param = list(netG.output.parameters())


    noise = torch.randn(args.batch_size, noise_dim).to(device)
    one = torch.tensor(1, dtype=torch.float)
    mone = torch.tensor(-1, dtype=torch.float)

    for e in tqdm(range(args.epochs), desc=f"Training...", total=args.epochs):
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
            G_optimizer_rnn.zero_grad()
            G_optimizer_output.zero_grad()

            # skip uneven batch
            if adj_mat.size(0) != args.batch_size:
                continue

            # fit graph2vec
            graph2vec.fit([nx.from_numpy_matrix(decode_adj(y).numpy()) for y in Y])

            ######################
            # Discriminator Update
            ######################
            # number of iteration to train the discriminator
            Diters = 5
            j = 0 # counter for 1, 2, ... Diters
            # enable training
            # netD.train()
            netD.train(True)
            # for param in netG.parameters():
            #     param.requires_grad = False
            netG.eval()

            # fit gam_trainer
            adjs = []
            inputs = torch.clone(adj_mat)
            for i in range(inputs.size(0)):
                adjs.append(inputs[i][:Y_len[i].item(), :Y_len[i].item()])
            gam_trainer.setup_graphs(adjs)
            gam_trainer.fit()

            b_errD = 0
            while j < Diters:
                j += 1
                # weight clipping: clamp parameters to a cube
                # for p in netD.parameters():
                #     p.data.clamp_(clamp_lower, clamp_upper)
                for p in gam_trainer.model.parameters():
                    p.data.clamp_(clamp_lower, clamp_upper)
                optimizerD.zero_grad()

                """=========Original============"""
                # # train with real
                # inputs = torch.clone(adj_mat)
                # # print("netD input shape: ", inputs.dtype)
                # D_pred = netD(inputs.to(torch.float32))
                # errD_real = D_pred
                # errD_real.backward() # discriminator should assign 1's to true samples

                # # train with fake
                # noise = torch.randn(args.batch_size, noise_dim) # (batch_size, hidden_size)
                # # insert data processing
                # with torch.no_grad():
                #     fake = netG(noise, X, Y, Y_len)
                # # embed = graph2vec.infer([nx.from_numpy_matrix(f.detach().numpy()) for f in fake])
                # fake_tensor = netD(fake)
                # errD_fake = -1*fake_tensor
                # errD_fake.backward() # discriminator should assign -1's to fake samples??

                # # compute Wasserstein distance and update parameters
                # errD = errD_real - errD_fake
                # optimizerD.step()

                """==========Test==========="""
                inputs = torch.clone(adj_mat)
                # remove padding on adj_mat
                for i in range(inputs.size(0)):
                    adj = inputs[i][:Y_len[i].item(), :Y_len[i].item()]
                    real_embed = gam_forward(adj, gam_trainer, target=1)
                errD_real = torch.mean(netD(real_embed))
                errD_real.backward(retain_graph=True)
                # errD_real.backward(gradient=torch.tensor([1]).to(torch.float), retain_graph=True)

                # print(" \n\n\n NetG to NetD \n\n\n")
                noise = torch.randn(args.batch_size, noise_dim)
                with torch.no_grad():
                    fake = netG(noise, X, Y, Y_len)
                    # while fake.sum() == 0:
                    #     fake = netG(noise, X, Y, Y_len)
                # remove padding on netG output (might have to torchify it
                for f in fake:
                    # print(f)
                    nonzeros = torch.nonzero(f) # []
                    min_indx, max_indx = torch.min(nonzeros), torch.max(nonzeros)
                    adj = f[min_indx:max_indx+1, min_indx:max_indx+1]
                    fake_embed = gam_forward(adj, gam_trainer, target=-1)
                    # batch_loss = netD.process_graph(batch_loss=batch_loss, already_matrix=True, adj=adj, target=0)
                errD_fake = -1*torch.mean(netD(fake_embed))
                errD_fake.backward(retain_graph=True)
                # errD_fake.backward(gradient=torch.tensor([-1]).to(torch.float), retain_graph=True)

                optimizerD.step()
                errD = errD_real - errD_fake
                b_errD += errD

                # print(f"Iterative errD {errD.item()}, errD_real {errD_real.item()}, errD_fake {errD_fake.item()}: ")

            # ========== Train Generator ==================
            # netD.eval()
            netD.eval()
            netG.train(True)
            G_optimizer_rnn.zero_grad()
            G_optimizer_output.zero_grad()
            noisev = torch.randn(args.batch_size, noise_dim)
            fake = netG(noisev, X, Y, Y_len) # return adjs
            # while fake.sum() == 0:
            #     fake = netG(noise, X, Y, Y_len)
            # fake_tensor = netD(fake)
            """============Test============="""
            # remove padding on netG output
            for f in fake:
                nonzeros = torch.nonzero(f)
                min_indx, max_indx = torch.min(nonzeros), torch.max(nonzeros)
                adj = f[min_indx:max_indx+1, min_indx:max_indx+1]
                fake_embed = gam_forward(adj, gam_trainer, target=-1)
            errG = -1*torch.mean(netD(fake_embed))
            errG.backward()
            # errG.backward(gradient=torch.tensor(-1).to(torch.float), retain_graph=True)
            G_optimizer_rnn.step()
            G_optimizer_output.step()



            # Winston's outline for inverter training
            # 0. train graph2vec on {generator distribution, true distribution}
            # 1. sample true samples x
            # 2. recon_graph = netG(netI(x))
            # 3. sample noise z
            # 4. recon_noise = netI(netG(z))
            # 5. minimize GW_dist(recon_graph - x) + lambda * MSE(recon_noise - z)

            # ========== Train Inverter =================
            if train_inverter:
                netG.eval()
                netI.train(True)
                original_graphs = adj_mat
                graph_lst = [nx.from_numpy_matrix(am.numpy()) for am in adj_mat]
                # retrain graph2vec
                graph2vec.fit(graph_lst)
                # genearte embedding
                embeddings = torch.Tensor(graph2vec.infer(graph_lst))
                I_output = netI(torch.reshape(embeddings, (embeddings.shape[0], -1)))
                with torch.no_grad():
                    G_pred_graphs = netG(I_output, X, Y, Y_len)
                reconst_graphs = G_pred_graphs

                # noise
                noise = torch.randn(args.batch_size, noise_dim)
                with torch.no_grad():
                    G_pred_noise = netG(noise, X, Y, Y_len)
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