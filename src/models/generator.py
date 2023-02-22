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
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
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
    def init_optimizer(self):
        """Initialize optimizers and schedular for both RNNs"""
        self.optimizer_rnn = optim.Adam(list(self.rnn.parameters()), lr=self.args.lr)
        self.optimizer_output = optim.Adam(list(self.output.parameters()), lr=self.args.lr)
        self.scheduler_rnn = MultiStepLR(self.optimizer_rnn, milestones=self.args.milestones, gamma=self.args.lr_rate)
        self.scheduler_output = MultiStepLR(self.optimizer_output, milestones=self.args.milestones, gamma=self.args.lr_rate)

    def clear_gradient_models(self):
        self.rnn.zero_grad()
        self.output.zero_grad()

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
        x_unsorted = X.float()
        y_unsorted = Y.float()
        y_len_unsorted = length
        y_len_max = max(y_len_unsorted)
        x_unsorted = x_unsorted[:, 0:y_len_max, :]
        y_unsorted = y_unsorted[:, 0:y_len_max, :]
        y_len,sort_index = torch.sort(y_len_unsorted,0,descending=True)
        y_len = y_len.numpy().tolist()
        x = torch.index_select(x_unsorted,0,sort_index)
        y = torch.index_select(y_unsorted,0,sort_index)
        y_reshape = pack_padded_sequence(y,y_len,batch_first=True).data
        idx = [i for i in range(y_reshape.size(0)-1, -1, -1)]
        idx = torch.LongTensor(idx)
        y_reshape = y_reshape.index_select(0, idx)
        y_reshape = y_reshape.view(y_reshape.size(0),y_reshape.size(1),1)
        output_x = torch.cat((torch.ones(y_reshape.size(0),1,1),y_reshape[:,0:-1,0:1]),dim=1) # x's shape is determined by y's shape
        output_y = y_reshape
        output_y_len = []
        output_y_len_bin = np.bincount(np.array(y_len))
        for i in range(len(output_y_len_bin)-1,0,-1):
            count_temp = np.sum(output_y_len_bin[i:]) # count how many y_len is above i
            output_y_len.extend([min(i,y.size(2))]*count_temp) # put them in output_y_len; max value should not exceed y.size(2)

        # pack into variable
        x = Variable(x).to(self.device)
        y = Variable(y).to(self.device)
        output_x = Variable(output_x).to(self.device)
        output_y = Variable(output_y).to(self.device)
        batch_size = x_unsorted.size(0)
        return x, y, output_x, output_y, y_len, output_y_len, batch_size

    def forward(self, X, Y, length):
        sorted_x, sorted_y, sorted_output_x, sorted_output_y, y_len, output_y_len, batch_size = self.sort_data_per_epoch(X, Y, length)
        # init hidden for rnn
        self.rnn.hidden = self.rnn.init_hidden(batch_size=batch_size)
        # rnn pass
        h = self.rnn(sorted_x, pack=True, input_len=y_len)
        h = pack_padded_sequence(h, y_len, batch_first=True)
        # reverse hidden
        idx = [i for i in range(h.size(0) - 1, -1, -1)]
        idx = Variable(torch.LongTensor(idx)).to(self.device)
        h = h.index_select(0, idx)
        hidden_null = Variable(torch.zeros(self.args.num_layers-1, h.size(0), h.size(1))).to(self.device)
        # init hidden for output
        self.output.hidden = torch.cat((h.view(1,h.size(0),h.size(1)), hidden_null),dim=0) # num_layers, batch_size, hidden_size
        # output pass
        y_pred = self.output(sorted_output_x, pack=True, input_len=output_y_len)
        y_pred = F.sigmoid(y_pred)
        # clean
        y_pred = pack_padded_sequence(y_pred, output_y_len, batch_first=True)
        y_pred = pad_packed_sequence(y_pred, batch_first=True)[0]
        sorted_output_y = pack_padded_sequence(sorted_output_y, output_y_len, batch_first=True)
        sorted_output_y = pad_packed_sequence(sorted_output_y, batch_first=True)[0]
        return y_pred, sorted_output_y

    def generate(self, X, args, test_batch_size=1):
        """
        X: noise/latent vector
        args: arguments dictionary
        test_batch_size: number of graphs you want to generate
        """
        # provide a option to change number of graphs generated
        if test_batch_size is None:
            test_batch_size = args.test_batch_size

        # self.rnn.hidden = self.rnn.init_hidden(test_batch_size)
        # print(self.rnn.hidden.shape)
        # print(X.shape)
        # return
        self.rnn.hidden = torch.stack(self.rnn.num_layers*[X]).to(self.device)

        # TODO: change this part to noise vector might need resizing
        y_pred_long = Variable(torch.zeros(test_batch_size, args.max_num_node, args.max_prev_node)).to(self.device) # discrete prediction
        # x_step = X.to(self.device) # shape:(batch_size, 1, args.max_prev_node)
        x_step = Variable(torch.ones(test_batch_size, 1, args.max_prev_node)).to(self.device)

        # iterative graph generation
        for i in range(args.max_num_node):
            # for each node
            # 1. we use rnn to create new node embedding
            # 2. we use output to create new edges

            # (1)
            h = self.rnn(x_step)
            hidden_null = Variable(torch.zeros(args.num_layers - 1, h.size(0), h.size(2))).to(self.device)
            x_step = Variable(torch.zeros(test_batch_size, 1, args.max_prev_node)).to(self.device)
            output_x_step = Variable(torch.ones(test_batch_size, 1, 1)).to(self.device)
            # (2)
            self.output.hidden = torch.cat((h.permute(1,0,2), hidden_null), dim=0).to(self.device)
            for j in range(min(args.max_prev_node,i+1)):
                output_y_pred_step = self.output(output_x_step)
                # print(output_y_pred_step.requires_grad)
                output_x_step = sample_sigmoid(output_y_pred_step, sample=True, sample_time=1, device=self.device)
                x_step[:,:,j:j+1] = output_x_step
                self.output.hidden = Variable(self.output.hidden.data).to(self.device)
            y_pred_long[:, i:i + 1, :] = x_step
            self.rnn.hidden = Variable(self.rnn.hidden.data).to(self.device)
        y_pred_long_data = y_pred_long.data.long()

        # TODO: check my work, I am commenting this part out because we don't want graph objects, we want adj_matrix
        # # collect the graphs
        # G_pred_list = []
        # for i in range(test_batch_size):
        #     adj_pred = decode_adj(y_pred_long_data[i].cpu().numpy())
        #     G_pred = get_graph(adj_pred) # get a graph from zero-padded adj
        #     G_pred_list.append(G_pred)
        # return G_pred_list

        adj_pred_list = []
        for i in range(test_batch_size):
            adj_pred = decode_adj(y_pred_long_data[i].cpu().numpy())
            adj_pred_list.append(adj_pred)
        return adj_pred_list