from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random
import pickle

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence, pad_sequence
from torch.optim.lr_scheduler import MultiStepLR

from collections import OrderedDict
import math
import numpy as np
import time as tm

from .generator_utils import *
from .args import *

################### current adopted model, LSTM+MLP || LSTM+VAE || LSTM+LSTM (where LSTM can be GRU as well) #####
# definition of terms
# h: hidden state of LSTM
# y: edge prediction, model output
# n: noise for generator
# l: whether an output is real or not, binary

# plain LSTM model
class LSTM_plain(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, num_layers, has_input=True, has_output=False, output_size=None):
        super(LSTM_plain, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.has_input = has_input
        self.has_output = has_output

        if has_input:
            self.input = nn.Linear(input_size, embedding_size)
            self.rnn = nn.LSTM(input_size=embedding_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        else:
            self.rnn = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        if has_output:
            self.output = nn.Sequential(
                nn.Linear(hidden_size, embedding_size),
                nn.ReLU(),
                nn.Linear(embedding_size, output_size)
            )

        self.relu = nn.ReLU()
        # initialize
        self.hidden = None # need initialize before forward run

        for name, param in self.rnn.named_parameters():
            if 'bias' in name:
                nn.init.constant(param, 0.25)
            elif 'weight' in name:
                nn.init.xavier_uniform(param,gain=nn.init.calculate_gain('sigmoid'))
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data = init.xavier_uniform(m.weight.data, gain=nn.init.calculate_gain('relu'))

    def init_hidden(self, batch_size):
        return (Variable(torch.zeros(self.num_layers, batch_size, self.hidden_size)).to(choose_device()),
                Variable(torch.zeros(self.num_layers, batch_size, self.hidden_size)).to(choose_device()))

    def forward(self, input_raw, pack=False, input_len=None):
        if self.has_input:
            input = self.input(input_raw)
            input = self.relu(input)
        else:
            input = input_raw
        if pack:
            input = pack_padded_sequence(input, input_len, batch_first=True)
        output_raw, self.hidden = self.rnn(input, self.hidden)
        if pack:
            output_raw = pad_packed_sequence(output_raw, batch_first=True)[0]
        if self.has_output:
            output_raw = self.output(output_raw)
        # return hidden state at each time step
        return output_raw

# plain GRU model
class GRU_plain(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, num_layers, has_input=True, has_output=False, output_size=None):
        super(GRU_plain, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.has_input = has_input
        self.has_output = has_output

        if has_input:
            self.input = nn.Linear(input_size, embedding_size)
            self.rnn = nn.GRU(input_size=embedding_size, hidden_size=hidden_size, num_layers=num_layers,
                              batch_first=True)
        else:
            self.rnn = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        if has_output:
            self.output = nn.Sequential(
                nn.Linear(hidden_size, embedding_size),
                nn.ReLU(),
                nn.Linear(embedding_size, output_size)
            )

        self.relu = nn.ReLU()
        # initialize
        self.hidden = None  # need initialize before forward run

        for name, param in self.rnn.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.25)
            elif 'weight' in name:
                nn.init.xavier_uniform_(param,gain=nn.init.calculate_gain('sigmoid'))
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data = init.xavier_uniform_(m.weight.data, gain=nn.init.calculate_gain('relu'))

    def init_hidden(self, batch_size):
        return Variable(torch.zeros(self.num_layers, batch_size, self.hidden_size)).to(choose_device())

    def forward(self, input_raw, pack=False, input_len=None):
        if self.has_input:
            input = self.input(input_raw)
            input = self.relu(input)
        else:
            input = input_raw
        if pack:
            input = pack_padded_sequence(input, input_len, batch_first=True)
        output_raw, self.hidden = self.rnn(input, self.hidden)
        if pack:
            output_raw = pad_packed_sequence(output_raw, batch_first=True)[0]
        if self.has_output:
            output_raw = self.output(output_raw)
        # return hidden state at each time step
        return output_raw

# a deterministic linear output
class MLP_plain(nn.Module):
    def __init__(self, h_size, embedding_size, y_size):
        super(MLP_plain, self).__init__()
        self.deterministic_output = nn.Sequential(
            nn.Linear(h_size, embedding_size),
            nn.ReLU(),
            nn.Linear(embedding_size, y_size)
        )

        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data = init.xavier_uniform(m.weight.data, gain=nn.init.calculate_gain('relu'))

    def forward(self, h):
        y = self.deterministic_output(h)
        return y

# a deterministic linear output, additional output indicates if the sequence should continue grow
class MLP_token_plain(nn.Module):
    def __init__(self, h_size, embedding_size, y_size):
        super(MLP_token_plain, self).__init__()
        self.deterministic_output = nn.Sequential(
            nn.Linear(h_size, embedding_size),
            nn.ReLU(),
            nn.Linear(embedding_size, y_size)
        )
        self.token_output = nn.Sequential(
            nn.Linear(h_size, embedding_size),
            nn.ReLU(),
            nn.Linear(embedding_size, 1)
        )

        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data = init.xavier_uniform(m.weight.data, gain=nn.init.calculate_gain('relu'))

    def forward(self, h):
        y = self.deterministic_output(h)
        t = self.token_output(h)
        return y,t

# a deterministic linear output (update: add noise)
class MLP_VAE_plain(nn.Module):
    def __init__(self, h_size, embedding_size, y_size):
        super(MLP_VAE_plain, self).__init__()
        self.encode_11 = nn.Linear(h_size, embedding_size) # mu
        self.encode_12 = nn.Linear(h_size, embedding_size) # lsgms

        self.decode_1 = nn.Linear(embedding_size, embedding_size)
        self.decode_2 = nn.Linear(embedding_size, y_size) # make edge prediction (reconstruct)
        self.relu = nn.ReLU()

        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data = init.xavier_uniform(m.weight.data, gain=nn.init.calculate_gain('relu'))

    def forward(self, h):
        # encoder
        z_mu = self.encode_11(h)
        z_lsgms = self.encode_12(h)
        # reparameterize
        z_sgm = z_lsgms.mul(0.5).exp_()
        eps = Variable(torch.randn(z_sgm.size())).to(choose_device())
        z = eps*z_sgm + z_mu
        # decoder
        y = self.decode_1(z)
        y = self.relu(y)
        y = self.decode_2(y)
        return y, z_mu, z_lsgms

# a deterministic linear output (update: add noise)
class MLP_VAE_conditional_plain(nn.Module):
    def __init__(self, h_size, embedding_size, y_size):
        super(MLP_VAE_conditional_plain, self).__init__()
        self.encode_11 = nn.Linear(h_size, embedding_size)  # mu
        self.encode_12 = nn.Linear(h_size, embedding_size)  # lsgms

        self.decode_1 = nn.Linear(embedding_size+h_size, embedding_size)
        self.decode_2 = nn.Linear(embedding_size, y_size)  # make edge prediction (reconstruct)
        self.relu = nn.ReLU()

        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data = init.xavier_uniform(m.weight.data, gain=nn.init.calculate_gain('relu'))

    def forward(self, h):
        # encoder
        z_mu = self.encode_11(h)
        z_lsgms = self.encode_12(h)
        # reparameterize
        z_sgm = z_lsgms.mul(0.5).exp_()
        eps = Variable(torch.randn(z_sgm.size(0), z_sgm.size(1), z_sgm.size(2))).to(choose_device())
        z = eps * z_sgm + z_mu
        # decoder
        y = self.decode_1(torch.cat((h,z),dim=2))
        y = self.relu(y)
        y = self.decode_2(y)
        return y, z_mu, z_lsgms

class DGM_graphs(nn.Module):
    def __init__(self,h_size):
        # h_size: node embedding size
        # h_size*2: graph embedding size

        super(DGM_graphs, self).__init__()
        ### all modules used by the model
        ## 1 message passing, 2 times
        self.m_uv_1 = nn.Linear(h_size*2, h_size*2)
        self.f_n_1 = nn.GRUCell(h_size*2, h_size) # input_size, hidden_size

        self.m_uv_2 = nn.Linear(h_size * 2, h_size * 2)
        self.f_n_2 = nn.GRUCell(h_size * 2, h_size)  # input_size, hidden_size

        ## 2 graph embedding and new node embedding
        # for graph embedding
        self.f_m = nn.Linear(h_size, h_size*2)
        self.f_gate = nn.Sequential(
            nn.Linear(h_size,1),
            nn.Sigmoid()
        )
        # for new node embedding
        self.f_m_init = nn.Linear(h_size, h_size*2)
        self.f_gate_init = nn.Sequential(
            nn.Linear(h_size,1),
            nn.Sigmoid()
        )
        self.f_init = nn.Linear(h_size*2, h_size)

        ## 3 f_addnode
        self.f_an = nn.Sequential(
            nn.Linear(h_size*2,1),
            nn.Sigmoid()
        )

        ## 4 f_addedge
        self.f_ae = nn.Sequential(
            nn.Linear(h_size * 2, 1),
            nn.Sigmoid()
        )

        ## 5 f_nodes
        self.f_s = nn.Linear(h_size*2, 1)

#=======Final GraphRNN model========
class GraphRNN(nn.Module):
    def __init__(self, args, device=choose_device()) -> None:
        super().__init__()
        self.args = args
        self.device = device
        self.rnn = GRU_plain(input_size=self.args.max_prev_node, embedding_size=self.args.embedding_size_rnn,
                        hidden_size=self.args.hidden_size_rnn, num_layers=self.args.num_layers, has_input=True,
                        has_output=True, output_size=self.args.hidden_size_rnn_output).to(self.device)
        self.output = GRU_plain(input_size=1, embedding_size=self.args.embedding_size_rnn_output,
                            hidden_size=self.args.hidden_size_rnn_output, num_layers=self.args.num_layers, has_input=True,
                            has_output=True, output_size=1).to(self.device)

        # load data state
        if args.load:
            fname = args.model_save_path + args.fname + 'lstm_' + str(args.load_epoch) + '.dat'
            self.rnn.load_state_dict(torch.load(fname))
            fname = args.model_save_path + args.fname + 'output_' + str(args.load_epoch) + '.dat'
            self.output.load_state_dict(torch.load(fname))

            args.lr = 0.00001
            epoch = args.load_epoch
            print('model loaded!, lr: {}'.format(args.lr))
        else:
            epoch = 1

    # ====Call these in training loop====
    def init_optimizer(self, lr):
        """Initialize optimizers and schedular for both RNNs"""
        self.optimizer_rnn = optim.Adam(list(self.rnn.parameters()), lr=lr)
        self.optimizer_output = optim.Adam(list(self.output.parameters()), lr=lr)
        self.scheduler_rnn = MultiStepLR(self.optimizer_rnn, milestones=self.args.milestones)
        self.scheduler_output = MultiStepLR(self.optimizer_output, milestones=self.args.milestones)
        return self.optimizer_rnn, self.optimizer_output, self.scheduler_rnn, self.scheduler_output

    def clear_gradient_models(self):
        self.rnn.zero_grad()
        self.output.zero_grad()

    def train(self, flag):
        if flag:
            self.rnn.train(True)
            self.output.train(True)
        else:
            self.rnn.train(False)
            self.output.train(False)

    def clear_gradient_opts(self):
        self.optimizer_rnn.zero_grad()
        self.optimizer_output.zero_grad()

    def all_steps(self):
        self.optimizer_rnn.step()
        self.optimizer_output.step()
        self.scheduler_rnn.step()
        self.scheduler_output.step()

    # ======================================

    def sort_data_per_epoch(self, X, Y, length):
        # print("Sort Data...")
        x_unsorted = X.float()
        y_unsorted = Y.float()
        y_len_unsorted = length
        y_len_max = max(y_len_unsorted)
        # batch-wise clear padding
        # x_unsorted = x_unsorted[:, 0:y_len_max, :]
        # y_unsorted = y_unsorted[:, 0:y_len_max, :]
        # print("y unsorted size: ", y_unsorted.size())

        # sort traget (y) by adj_vector size
        y_len,sort_index = torch.sort(y_len_unsorted,0,descending=True)
        # y_len = y_len.numpy().tolist()
        x = torch.index_select(x_unsorted,0,sort_index)
        y = torch.index_select(y_unsorted,0,sort_index)
        y_len = [y.size(1) for _ in range(y.size(0))] # use uniform length
        # print("y sorted size: ", y.size())

        # create packed seqeunce to pass into the GRU model
        # print('y len size: ', len(y_len))
        # print('y len: ', y_len)
        y_reshape = pack_padded_sequence(y,y_len,batch_first=True).data # This line changes the y shape by stacking across batch vertically
        # print(y_reshape.size())  #( batch*(node, max_prev_node)
        # y_reshape = y
        idx = [i for i in range(y_reshape.size(0)-1, -1, -1)]
        idx = torch.LongTensor(idx)
        y_reshape = y_reshape.index_select(0, idx)
        # print(y_reshape.shape) #( batch*node, max_prev_node)
        y_reshape = y_reshape.view(y_reshape.size(0),y_reshape.size(1),1)
        # print(y_reshape.shape) #( batch*node, max_prev_node, 1)
        output_x = torch.cat((torch.ones(y_reshape.size(0),1,1),y_reshape[:,0:-1,0:1]),dim=1) # x's shape is determined by y's shape
        # print('output x shape: ', output_x.shape) #( batch*node, max_prev_node)
        output_y = y_reshape
        # print('output y shape: ', output_y.shape)
        output_x_len = []
        output_x_len_bin = np.bincount(np.array(y_len))
        for i in range(len(output_x_len_bin)-1,0,-1):
            count_temp = np.sum(output_x_len_bin[i:]) # count how many y_len is above i
            output_x_len.extend([min(i,y.size(2))]*count_temp) # put them in output_x_len; max value should not exceed y.size(2)
        # print('output_x_len: ', output_x_len)
        # pack into variable
        x = Variable(x).to(self.device)
        y = Variable(y).to(self.device)
        output_x = Variable(output_x).to(self.device)
        output_y = Variable(output_y).to(self.device)
        batch_size = x_unsorted.size(0)
        # print("sorted batch size: ", batch_size)
        return x, y, output_x, output_y, y_len, output_x_len, batch_size

    def forward(self, noise, X, Y, length):
        """
        X: noise/latent vector
        args: arguments dictionary
        test_batch_size: number of graphs you want to generate
        """
        # provide a option to change number of graphs generated
        output_batch_size = self.args.test_batch_size
        input_hidden = torch.stack(self.rnn.num_layers*[noise]).to(self.device)
        self.rnn.hidden = input_hidden # expected shape: (num_layer, batch_size, hidden_size)
        # print("X shape: ", X.shape)
        x, y, output_x, output_y, y_len, output_x_len, _ = self.sort_data_per_epoch(X, Y, length)
        # print("Forward pass...")
        # print("output y len :", len(output_x_len))
        # print("ouput_x shape: ", output_x.size())
        h = self.rnn(x, pack=True, input_len=y_len)
        # print('h shape: ', h.shape) # (batch, padded, y_len)
        # TEST: keep the pack_padded_sequence object as input
        h = pack_padded_sequence(h,y_len, batch_first=True).data # get packed hidden vector
        # print('padded h shape: ', h.shape)
        # reverse h
        idx = [i for i in range(h.size(0) - 1, -1, -1)]
        idx = Variable(torch.LongTensor(idx)).cuda()
        h = h.index_select(0, idx)
        hidden_null = Variable(torch.zeros(self.rnn.num_layers-1, h.size(0), h.size(1))).cuda()
        # print('hidden null shape: ', hidden_null.shape)
        self.output.hidden = torch.cat((h.view(1,h.size(0),h.size(1)),hidden_null),dim=0) # num_layers, batch_size, hidden_size
        # print("ouptut x shape: ", output_x.shape)
        y_pred = self.output(output_x, pack=True, input_len=output_x_len)
        y_pred = torch.sigmoid(y_pred)
        # print("y_pred shape (1) : ", y_pred.shape)
        # clean
        y_pred = pack_padded_sequence(y_pred, output_x_len, batch_first=True)
        y_pred = pad_packed_sequence(y_pred, batch_first=True)[0]
        # print("y_pred shape (2) : ", y_pred.shape)
        # print("output y len sum: ", sum(output_x_len))
        # out = decode_adj(y_pred)
        y_output = y_pred[:, :, 0]
        y_output = y_output.reshape(X.size(0), X.size(1), X.size(2))
        # print(y_output.size())
        y_output_adj = torch.stack([decode_adj(op) for op in y_output])
        # print(y_output_adj.size())
        return y_output_adj

    # this is for testing only
    # def forwardv2(self, noise, X, Y, length):
    #     """
    #     noise: noise/latent vector
    #     X: modified sequenced adjcency vectors (with global padding)
    #     Y: original seuqneced adjcency vectors (with global padding)
    #     length: length of each adjcency vectors
    #     """
    #     # Get self.rnn input (sort data, filter by max, etc)
    #     x_unsorted = X.float()
    #     y_unsorted = Y.float()
    #     y_len_unsorted = length

    #     # sort by length
    #     y_len,sort_index = torch.sort(y_len_unsorted,0,descending=True)
    #     # y_len = y_len.numpy().tolist()
    #     x = torch.index_select(x_unsorted,0,sort_index)
    #     y = torch.index_select(y_unsorted,0,sort_index)
    #     output_x = torch.cat((torch.ones(y.size(0),1,1),y[:,0:-1,0:1]),dim=1) # x's shape is determined by y's shape
    #     print(output_x.size(), y.size())

    #     # # output_x_len is needed to count self.output forward output length
    #     # output_x_len = []
    #     # output_x_len_bin = np.bincount(np.array(y_len))
    #     # for i in range(len(output_x_len_bin)-1,0,-1):
    #     #     count_temp = np.sum(output_x_len_bin[i:]) # count how many y_len is above i
    #     #     output_x_len.extend([min(i,y.size(2))]*count_temp) # put them in output_x_len; max value should not exceed y.size(2)

    #     #TODO: might need more transformation, but I don't know how to add them without breaking the pack_padded_sequence object

    #     # pack them to the right device
    #     x = x.to(self.device)
    #     y = y.to(self.device)
    #     output_x = output_x.to(self.device)

    #     # now for the self.rnn forward
    #     self.rnn.hidden = torch.stack(self.rnn.num_layers*[noise]).to(self.device)
    #     h = self.rnn(x, pack=False)
    #     # print("h.shape: ", h.shape)

    #     # create self.output's hidden vector from self.rnn's output h
    #     # h = pad_sequence(h, batch_first=True) # now the input and output would have the same shape as output_x and output_y
    #     h = pack_padded_sequence(h, [h.size(1) for _ in range(h.size(0))], batch_first=True).data # this essentially flatten the first two dimension of h
    #     # print('hidden size: ', self.args.hidden_size_rnn_output)
    #     print("h.shape: ", h.size())
    #     # reverse and pad h (not sure how this will influence the output)
    #     idx = [i for i in range(h.size(0) - 1, -1, -1)]
    #     idx = Variable(torch.LongTensor(idx)).to(self.device)
    #     h = h.index_select(0, idx)
    #     hidden_null = Variable(torch.zeros(self.rnn.num_layers-1, h.size(0), h.size(1))).to(self.device)
    #     self.output.hidden = torch.cat((h.view(1,h.size(0),h.size(1)),hidden_null),dim=0)

    #     # for the self.output forward
    #     y_pred = self.output(output_x, pack=False)
    #     print("y_pred.shape: ", y_pred.size())
    #     return

    def generate(self, X):
        """
        X: noise/latent vector
        args: arguments dictionary
        test_batch_size: number of graphs you want to generate
        """
        # provide a option to change number of graphs generated
        output_batch_size = self.args.test_batch_size
        input_hidden = torch.stack(self.rnn.num_layers*[X]).to(self.device)
        self.rnn.hidden = input_hidden # expected shape: (num_layer, batch_size, hidden_size)

        y_pred_long = Variable(torch.zeros(output_batch_size, self.args.max_num_node, self.args.max_prev_node)).to(self.device) # discrete prediction
        x_step = Variable(torch.ones(output_batch_size, 1, self.args.max_prev_node)).to(self.device)

        # iterative graph generation
        for i in range(self.args.max_num_node):
            # for each node
            # 1. we use rnn to create new node embedding
            # 2. we use output to create new edges

            # (1)
            h = self.rnn(x_step)
            # print('h grad: ', h.grad)
            hidden_null = Variable(torch.zeros(self.args.num_layers - 1, h.size(0), h.size(2))).to(self.device)
            x_step = Variable(torch.zeros(output_batch_size, 1, self.args.max_prev_node)).to(self.device)
            output_x_step = Variable(torch.ones(output_batch_size, 1, 1)).to(self.device)

            # (2)
            self.output.hidden = torch.cat((h.permute(1,0,2), hidden_null), dim=0).to(self.device)
            for j in range(min(self.args.max_prev_node,i+1)):
                output_y_pred_step = self.output(output_x_step)
                # print('output y grad: ', output_y_pred_step.grad)
                output_x_step = sample_sigmoid(output_y_pred_step, sample=True, sample_time=1, device=self.device)
                x_step[:,:,j:j+1] = output_x_step
                # self.output.hidden = Variable(self.output.hidden.data).to(self.device)
            y_pred_long[:, i:i + 1, :] = x_step
            # self.rnn.hidden = Variable(self.rnn.hidden.data).to(self.device)
        y_pred_long_data = y_pred_long.data.long()
        # print("y pred grad: ", y_pred_long_data.grad)

        init_adj_pred = decode_adj(y_pred_long_data[0].cpu())
        adj_pred_list = torch.zeros((output_batch_size, init_adj_pred.size(0), init_adj_pred.size(1)))
        for i in range(output_batch_size):
            # adj_pred = decode_adj(y_pred_long_data[i].cpu().numpy())
            # adj_pred_list = np.append(adj_pred_list, adj_pred)
            # adj_pred_list.append(adj_pred)
            adj_pred_list[i, :, :] = decode_adj(y_pred_long_data[i].cpu())

        # return torch.Tensor(np.array(adj_pred_list))
        adj_pred_list.requires_grad = True
        return adj_pred_list
