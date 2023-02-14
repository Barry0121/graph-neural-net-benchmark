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

def test_rnn_epoch(epoch, num_layers, max_num_node, max_prev_node, rnn, output, test_batch_size=16):
    """From GraphRNN codebase; generate graph for evaluation"""
    rnn.hidden = rnn.init_hidden(test_batch_size)
    rnn.eval()
    output.eval()

    # generate graphs
    y_pred_long = Variable(torch.zeros(test_batch_size, max_num_node, max_prev_node)).to(device) # discrete prediction
    x_step = Variable(torch.ones(test_batch_size,1,max_prev_node)).to(device)
    for i in range(max_num_node):
        h = rnn(x_step)
        # output.hidden = h.permute(1,0,2)
        hidden_null = Variable(torch.zeros(num_layers - 1, h.size(0), h.size(2))).to(device)
        output.hidden = torch.cat((h.permute(1,0,2), hidden_null),
                                  dim=0)  # num_layers, batch_size, hidden_size
        x_step = Variable(torch.zeros(test_batch_size,1,max_prev_node)).to(device)
        output_x_step = Variable(torch.ones(test_batch_size,1,1)).to(device)
        for j in range(min(max_prev_node,i+1)):
            output_y_pred_step = output(output_x_step)
            output_x_step = sample_sigmoid(output_y_pred_step, sample=True, sample_time=1)
            x_step[:,:,j:j+1] = output_x_step
            output.hidden = Variable(output.hidden.data).to(device)
        y_pred_long[:, i:i + 1, :] = x_step
        rnn.hidden = Variable(rnn.hidden.data).to(device)
    y_pred_long_data = y_pred_long.data.long()

    # save graphs as pickle
    G_pred_list = []
    for i in range(test_batch_size):
        adj_pred = decode_adj(y_pred_long_data[i].cpu().numpy())
        G_pred = get_graph(adj_pred) # get a graph from zero-padded adj
        G_pred_list.append(G_pred)

    return G_pred_list

def train(dataset_name, noise_dim, num_layers=4, clamp_lower=-0.01, clamp_upper=0.01, epochs=10, lr=1e-3, betas=1e-5, batch_size=1, lamb=0.1, loss_func='MSE', device=choose_device()):
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

    # From GraphRNN
    optimizerG_rnn = optim.Adam(list(G_rnn.parameters()), lr=lr, betas=betas)
    optimizerG_output = optim.Adam(list(G_output.parameters()), lr=lr, betas=betas)
    # scheduler_rnn = MultiStepLR(optimizer_rnn, milestones=args.milestones, gamma=args.lr_rate)
    # scheduler_output = MultiStepLR(optimizer_output, milestones=args.milestones, gamma=args.lr_rate)



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
        for i, data in enumerate(train_loader):
            X = data['x']
            Y = data['y']
            Y_len = data['len']
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
            # sort training data
            y_len_max = max(y_len_unsorted)
            x_unsorted = X[:, 0:y_len_max, :]
            y_unsorted = Y[:, 0:y_len_max, :]
            max_num_node = train_loader.n
            max_prev_node = 1 # might be problematic

            # freeze discriminator weights
            for p in D.parameters():
                p.requires_grad = False # to avoid computation

            # clear gradients
            G_rnn.zero_grad()
            G_output.zero_grad()

            # initialize hidden states
            rnn.hidden = rnn.init_hidden(batch_size=x_unsorted.size(0))

            # sort input
            y_len,sort_index = torch.sort(y_len_unsorted,0,descending=True)
            y_len = y_len.numpy().tolist()
            x = torch.index_select(x_unsorted,0,sort_index)
            y = torch.index_select(y_unsorted,0,sort_index)

            #======from official training=====
            # input, output for output rnn module
            # a smart use of pytorch builtin function: pack variable--b1_l1,b2_l1,...,b1_l2,b2_l2,...
            y_reshape = pack_padded_sequence(y,y_len,batch_first=True).data
            # reverse y_reshape, so that their lengths are sorted, add dimension
            idx = [i for i in range(y_reshape.size(0)-1, -1, -1)]
            idx = torch.LongTensor(idx)
            y_reshape = y_reshape.index_select(0, idx)
            y_reshape = y_reshape.view(y_reshape.size(0),y_reshape.size(1),1)
            output_x = torch.cat((torch.ones(y_reshape.size(0),1,1),y_reshape[:,0:-1,0:1]),dim=1)
            output_y = y_reshape
            # batch size for output module: sum(y_len)
            output_y_len = []
            output_y_len_bin = np.bincount(np.array(y_len))
            for i in range(len(output_y_len_bin)-1,0,-1):
                count_temp = np.sum(output_y_len_bin[i:]) # count how many y_len is above i
                output_y_len.extend([min(i,y.size(2))]*count_temp) # put them in output_y_len; max value should not exceed y.size(2)
            # pack into variable
            x = Variable(x).to(device)
            y = Variable(y).to(device)
            output_x = Variable(output_x).to(device)
            output_y = Variable(output_y).to(device)
            #=======end=============


            # TODO======confused about what to do about this part==========
            # if using ground truth to train
            h = G_rnn(x, pack=True, input_len=y_len) # pass into the hidden rnn
            h = pack_padded_sequence(h,y_len,batch_first=True).data # get packed hidden vector
            # reverse h
            idx = [i for i in range(h.size(0) - 1, -1, -1)]
            idx = Variable(torch.LongTensor(idx)).to(device)
            h = h.index_select(0, idx)
            hidden_null = Variable(torch.zeros(num_layers-1, h.size(0), h.size(1))).to(device)
            output.hidden = torch.cat((h.view(1,h.size(0),h.size(1)),hidden_null),dim=0) # num_layers, batch_size, hidden_size
            y_pred = G_output(output_x, pack=True, input_len=output_y_len) # pass into the output model
            y_pred = F.sigmoid(y_pred)
            # clean
            y_pred = pack_padded_sequence(y_pred, output_y_len, batch_first=True)
            y_pred = pad_packed_sequence(y_pred, batch_first=True)[0]
            output_y = pack_padded_sequence(output_y,output_y_len,batch_first=True)
            output_y = pad_packed_sequence(output_y,batch_first=True)[0]
            # use cross entropy loss
            loss = binary_cross_entropy_weight(y_pred, output_y)
            loss.backward()


            #=========== my code =======
            input = noise.resize_(batch_size, 1).normal_(0, 1) # TODO: why are we passing fake examples in there?
                # TODO: insert data processing

            fake = G_output(input) # G_output
            errG = D(fake) # <- critic's opinion; output of solution f as in WGAN Theorem 3

            #=========== how to combine the two codes ===============

            # update netG parameters
            errG.backward(one)
            optimizerG_rnn.step() # TODO: do we need to train the hidden graphRNN?
            optimizerG_output.step()

            gen_iterations += 1
            print(f"====Finished in {(time.time()-gstart)%60} sec====", '\n')

            if epoch % 2 == 0 and epoch>=2:
                # for sample_time in range(1,4):
                G_pred = []
                while len(G_pred)<300:
                    G_pred_step = test_rnn_epoch(epoch, num_layers, rnn, output)
                    G_pred.extend(G_pred_step)
                # save graphs
                fname = './cache/graphs/' + dataset_name+'_' + str(e) +'_'+str(sample_time) + '.dat'
                save_graph_list(G_pred, fname)

                print('test done, graphs saved')



            # save generator model weights
            if e % 2 == 0:
                fname0 = dataset_name + '_'
                fname = './cache/' + fname0 + 'lstm_' + str(e) + '.dat'
                torch.save(rnn.state_dict(), fname)
                fname = './cache/' + fname0 + 'output_' + str(e) + '.dat'
                torch.save(output.state_dict(), fname)

        # Print out training information.
        if (e+1) % 1 == 0:
            elapsed_time = time.time() - start_time
            print('Elapsed time [{:.4f}], Iteration [{}/{}], I Loss: {:.4f}'.format(
                elapsed_time, e+1, epochs, lossI.item()))
    print("====End of Training====")


name = 'MUTAG'
noise_dim = 8
train(name, noise_dim)