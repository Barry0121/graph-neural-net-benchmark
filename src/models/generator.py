"""
Implementation of the generator

Author: Winston Yu
"""

from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

from collections import OrderedDict
import math
import numpy as np
import time
from args import *
import model 

args = Args()

def choose_device():
    if torch.cuda.is_available():
        return 'cuda'
    elif torch.backends.mps.is_available():
        return 'mps'
    else:
        return 'cpu'

def binary_cross_entropy_weight(y_pred, y,has_weight=False, weight_length=1, weight_max=10):
    '''

    :param y_pred:
    :param y:
    :param weight_length: how long until the end of sequence shall we add weight
    :param weight_value: the magnitude that the weight is enhanced
    :return:
    '''
    if has_weight:
        weight = torch.ones(y.size(0),y.size(1),y.size(2))
        weight_linear = torch.arange(1,weight_length+1)/weight_length*weight_max
        weight_linear = weight_linear.view(1,weight_length,1).repeat(y.size(0),1,y.size(2))
        weight[:,-1*weight_length:,:] = weight_linear
        loss = F.binary_cross_entropy(y_pred, y, weight=weight.to(choose_device()))
    else:
        loss = F.binary_cross_entropy(y_pred, y)
    return loss

def sample_tensor(y,sample=True, thresh=0.5):
    # do sampling
    if sample:
        y_thresh = Variable(torch.rand(y.size())).to(choose_device())
        y_result = torch.gt(y,y_thresh).float()
    # do max likelihood based on some threshold
    else:
        y_thresh = Variable(torch.ones(y.size())*thresh).to(choose_device())
        y_result = torch.gt(y, y_thresh).float()
    return y_result

def gumbel_softmax(logits, temperature, eps=1e-9):
    '''

    :param logits: shape: N*L
    :param temperature:
    :param eps:
    :return:
    '''
    # get gumbel noise
    noise = torch.rand(logits.size())
    noise.add_(eps).log_().neg_()
    noise.add_(eps).log_().neg_()
    noise = Variable(noise).to(choose_device())

    x = (logits + noise) / temperature
    x = F.softmax(x)
    return x

def gumbel_sigmoid(logits, temperature):
    '''

    :param logits:
    :param temperature:
    :param eps:
    :return:
    '''
    # get gumbel noise
    noise = torch.rand(logits.size()) # uniform(0,1)
    noise_logistic = torch.log(noise)-torch.log(1-noise) # logistic(0,1)
    noise = Variable(noise_logistic).to(choose_device())

    x = (logits + noise) / temperature
    x = F.sigmoid(x)
    return x

# made deterministic
def sample_sigmoid(y, sample, thresh=0.5, sample_time=2):
    '''
    do sampling over unnormalized score
    :param y: input
    :param sample: Bool
    :param thresh: if not sample, the threshold
    :param sample_time: how many times do we sample, if =1, do single sample
    :return: sampled result
    '''

    y = F.sigmoid(y) # make y into probabilities
    if sample: # do sampling
        if sample_time>1:
            # if deterministic
            y_result = Variable(y.size(0),y.size(1),y.size(2)).to(choose_device())

            # if random
            # y_result = Variable(torch.rand(y.size(0),y.size(1),y.size(2))).to(choose_device())

            for i in range(y_result.size(0)): # loop over all batches
                for _ in range(sample_time): # do 'multi_sample' times sampling
                    # if deterministic
                    y_thresh = Variable(y.size(0),y.size(1),y.size(2)).to(choose_device())
                    # if random
                    # y_thresh = Variable(torch.rand(y.size(1), y.size(2))).to(choose_device())

                    y_result[i] = torch.gt(y[i], y_thresh).float()
                    if (torch.sum(y_result[i]).data > 0).any():
                        break
                    # else:
                    #     print('all zero',j)
        else:
            # if deterministic
            y_thresh = Variable(y.size(0),y.size(1),y.size(2)).to(choose_device())
            y_result = torch.gt(y,y_thresh).float()

            # if random
            # y_thresh = Variable(torch.rand(y.size(0),y.size(1),y.size(2))).to(choose_device())
            # y_result = torch.gt(y,y_thresh).float()

    # do max likelihood based on some threshold
    else:
        y_thresh = Variable(torch.ones(y.size(0), y.size(1), y.size(2)) * thresh).to(choose_device())
        y_result = torch.gt(y, y_thresh).float()
    return y_result

# made deterministic
def sample_sigmoid_supervised(y_pred, y, current, y_len, sample_time=2):
    '''
    do sampling over unnormalized score
    :param y_pred: input
    :param y: supervision
    :param sample: Bool
    :param thresh: if not sample, the threshold
    :param sampe_time: how many times do we sample, if =1, do single sample
    :return: sampled result
    '''

    y_pred = F.sigmoid(y_pred) # make y_pred into probabilities

    # if deterministic
    y_result = Variable(y_pred.size(0), y_pred.size(1), y_pred.size(2)).to(choose_device())
    # if random
    # y_result = Variable(torch.rand(y_pred.size(0), y_pred.size(1), y_pred.size(2))).to(choose_device()) # do sampling

    for i in range(y_result.size(0)): # loop over all batches
        if current<y_len[i]: # using supervision
            while True:
                # if deterministic
                y_thresh = Variable(y_pred.size(1), y_pred.size(2)).to(choose_device())
                # if random
                # y_thresh = Variable(torch.rand(y_pred.size(1), y_pred.size(2))).to(choose_device())

                y_result[i] = torch.gt(y_pred[i], y_thresh).float()
                y_diff = y_result[i].data - y[i]
                if (y_diff >= 0).all():
                    break
        else: # not using supervision
            # do 'multi_sample' times sampling
            for _ in range(sample_time):
                # if deterministic
                y_thresh = Variable(y_pred.size(1), y_pred.size(2)).to(choose_device())
                # if random
                # y_thresh = Variable(torch.rand(y_pred.size(1), y_pred.size(2))).to(choose_device())

                y_result[i] = torch.gt(y_pred[i], y_thresh).float()
                if (torch.sum(y_result[i]).data > 0).any():
                    break
    return y_result

# made deterministic
def sample_sigmoid_supervised_simple(y_pred, y, current, y_len, sample_time=2):
    '''
    do sampling over unnormalized score
    :param y_pred: input
    :param y: supervision
    :param sample: Bool
    :param thresh: if not sample, the threshold
    :param sampe_time: how many times do we sample, if =1, do single sample
    :return: sampled result
    '''

    y_pred = F.sigmoid(y_pred) # make y_pred into probabilities
    # if deterministic
    y_result = Variable(y_pred.size(0), y_pred.size(1), y_pred.size(2)).to(choose_device())
    # if random
    # y_result = Variable(torch.rand(y_pred.size(0), y_pred.size(1), y_pred.size(2))).to(choose_device())

    for i in range(y_result.size(0)): # loop over all batches
        if current < y_len[i]: # using supervision
            y_result[i] = y[i]
        else: # supervision done
            for _ in range(sample_time): # do 'multi_sample' times sampling
                # if deterministic
                y_thresh = Variable(y_pred.size(1), y_pred.size(2)).to(choose_device())
                # if random
                y_thresh = Variable(torch.rand(y_pred.size(1), y_pred.size(2))).to(choose_device())

                y_result[i] = torch.gt(y_pred[i], y_thresh).float()
                if (torch.sum(y_result[i]).data > 0).any():
                    break
    return y_result

################### current adopted model, LSTM+MLP || LSTM+VAE || LSTM+LSTM (where LSTM can be GRU as well)
#####
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
                nn.init.constant(param, 0.25)
            elif 'weight' in name:
                nn.init.xavier_uniform(param,gain=nn.init.calculate_gain('sigmoid'))
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data = init.xavier_uniform(m.weight.data, gain=nn.init.calculate_gain('relu'))

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

########### baseline model 1: Learning deep generative model of graphs

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

def message_passing(node_neighbor, node_embedding, model):
    node_embedding_new = []
    for i in range(len(node_neighbor)):
        neighbor_num = len(node_neighbor[i])
        if neighbor_num > 0:
            node_self = node_embedding[i].expand(neighbor_num, node_embedding[i].size(1))
            node_self_neighbor = torch.cat([node_embedding[j] for j in node_neighbor[i]], dim=0)
            message = torch.sum(model.m_uv_1(torch.cat((node_self, node_self_neighbor), dim=1)), dim=0, keepdim=True)
            node_embedding_new.append(model.f_n_1(message, node_embedding[i]))
        else:
            message_null = Variable(torch.zeros((node_embedding[i].size(0),node_embedding[i].size(1)*2))).to(choose_device())
            node_embedding_new.append(model.f_n_1(message_null, node_embedding[i]))
    node_embedding = node_embedding_new
    node_embedding_new = []
    for i in range(len(node_neighbor)):
        neighbor_num = len(node_neighbor[i])
        if neighbor_num > 0:
            node_self = node_embedding[i].expand(neighbor_num, node_embedding[i].size(1))
            node_self_neighbor = torch.cat([node_embedding[j] for j in node_neighbor[i]], dim=0)
            message = torch.sum(model.m_uv_1(torch.cat((node_self, node_self_neighbor), dim=1)), dim=0, keepdim=True)
            node_embedding_new.append(model.f_n_1(message, node_embedding[i]))
        else:
            message_null = Variable(torch.zeros((node_embedding[i].size(0), node_embedding[i].size(1) * 2))).to(choose_device())
            node_embedding_new.append(model.f_n_1(message_null, node_embedding[i]))
    return node_embedding_new

def calc_graph_embedding(node_embedding_cat, model):
    node_embedding_graph = model.f_m(node_embedding_cat)
    node_embedding_graph_gate = model.f_gate(node_embedding_cat)
    graph_embedding = torch.sum(torch.mul(node_embedding_graph, node_embedding_graph_gate), dim=0, keepdim=True)
    return graph_embedding

def calc_init_embedding(node_embedding_cat, model):
    node_embedding_init = model.f_m_init(node_embedding_cat)
    node_embedding_init_gate = model.f_gate_init(node_embedding_cat)
    init_embedding = torch.sum(torch.mul(node_embedding_init, node_embedding_init_gate), dim=0, keepdim=True)
    init_embedding = model.f_init(init_embedding)
    return init_embedding



#=============GraphRNN Training + Generating Ref.==================
def train_rnn_epoch(epoch, args, rnn, output, batch_index, data,
                    optimizer_rnn, optimizer_output,
                    scheduler_rnn, scheduler_output):
    rnn.train()
    output.train()
    loss_sum = 0
    rnn.zero_grad()
    output.zero_grad()
    x_unsorted = data['x'].float()
    y_unsorted = data['y'].float()
    y_len_unsorted = data['len']
    y_len_max = max(y_len_unsorted)
    x_unsorted = x_unsorted[:, 0:y_len_max, :]
    y_unsorted = y_unsorted[:, 0:y_len_max, :]
    # initialize lstm hidden state according to batch size
    rnn.hidden = rnn.init_hidden(batch_size=x_unsorted.size(0))
    # output.hidden = output.init_hidden(batch_size=x_unsorted.size(0)*x_unsorted.size(1))

    # sort input
    y_len,sort_index = torch.sort(y_len_unsorted,0,descending=True)
    y_len = y_len.numpy().tolist()
    x = torch.index_select(x_unsorted,0,sort_index)
    y = torch.index_select(y_unsorted,0,sort_index)

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
    x = Variable(x).to(choose_device())
    y = Variable(y).to(choose_device())
    output_x = Variable(output_x).to(choose_device())
    output_y = Variable(output_y).to(choose_device())
    # print(output_y_len)
    # print('len',len(output_y_len))
    # print('y',y.size())
    # print('output_y',output_y.size())


    # if using ground truth to train
    h = rnn(x, pack=True, input_len=y_len)
    h = pack_padded_sequence(h,y_len,batch_first=True).data # get packed hidden vector
    # reverse h
    idx = [i for i in range(h.size(0) - 1, -1, -1)]
    idx = Variable(torch.LongTensor(idx)).to(choose_device())
    h = h.index_select(0, idx)
    hidden_null = Variable(torch.zeros(args.num_layers-1, h.size(0), h.size(1))).to(choose_device())
    output.hidden = torch.cat((h.view(1,h.size(0),h.size(1)),hidden_null),dim=0) # num_layers, batch_size, hidden_size
    y_pred = output(output_x, pack=True, input_len=output_y_len)
    y_pred = F.sigmoid(y_pred)
    # clean
    y_pred = pack_padded_sequence(y_pred, output_y_len, batch_first=True)
    y_pred = pad_packed_sequence(y_pred, batch_first=True)[0]
    output_y = pack_padded_sequence(output_y,output_y_len,batch_first=True)
    output_y = pad_packed_sequence(output_y,batch_first=True)[0]
    # use cross entropy loss
    loss = binary_cross_entropy_weight(y_pred, output_y)
    loss.backward()
    # update deterministic and lstm
    optimizer_output.step()
    optimizer_rnn.step()
    scheduler_output.step()
    scheduler_rnn.step()


    if epoch % args.epochs_log==0 and batch_idx==0: # only output first batch's statistics
        print('Epoch: {}/{}, train loss: {:.6f}, graph type: {}, num_layer: {}, hidden: {}'.format(
            epoch, args.epochs,loss.data[0], args.graph_type, args.num_layers, args.hidden_size_rnn))

    # logging
    log_value('loss_'+args.fname, loss.data[0], epoch*args.batch_ratio+batch_idx)
    feature_dim = y.size(1)*y.size(2)
    loss_sum += loss.data[0]*feature_dim
    return loss_sum/(batch_idx+1)



def test_rnn_epoch(epoch, args, rnn, output, test_batch_size=16):
    rnn.hidden = rnn.init_hidden(test_batch_size)
    rnn.eval()
    output.eval()

    # generate graphs
    max_num_node = int(args.max_num_node)
    y_pred_long = Variable(torch.zeros(test_batch_size, max_num_node, args.max_prev_node)).to(choose_device()) # discrete prediction
    x_step = Variable(torch.ones(test_batch_size,1,args.max_prev_node)).to(choose_device())
    for i in range(max_num_node):
        h = rnn(x_step)
        # output.hidden = h.permute(1,0,2)
        hidden_null = Variable(torch.zeros(args.num_layers - 1, h.size(0), h.size(2))).to(choose_device())
        output.hidden = torch.cat((h.permute(1,0,2), hidden_null),
                                  dim=0)  # num_layers, batch_size, hidden_size
        x_step = Variable(torch.zeros(test_batch_size,1,args.max_prev_node)).to(choose_device())
        output_x_step = Variable(torch.ones(test_batch_size,1,1)).to(choose_device())
        for j in range(min(args.max_prev_node,i+1)):
            output_y_pred_step = output(output_x_step)
            output_x_step = sample_sigmoid(output_y_pred_step, sample=True, sample_time=1)
            x_step[:,:,j:j+1] = output_x_step
            output.hidden = Variable(output.hidden.data).to(choose_device())
        y_pred_long[:, i:i + 1, :] = x_step
        rnn.hidden = Variable(rnn.hidden.data).to(choose_device())
    y_pred_long_data = y_pred_long.data.long()

    # save graphs as pickle
    G_pred_list = []
    for i in range(test_batch_size):
        adj_pred = decode_adj(y_pred_long_data[i].cpu().numpy())
        G_pred = get_graph(adj_pred) # get a graph from zero-padded adj
        G_pred_list.append(G_pred)

    return G_pred_list


def GraphRNN(batch_index, data, args=Args()):
    rnn = GRU_plain(input_size=args.max_prev_node, embedding_size=args.embedding_size_rnn,
                        hidden_size=args.hidden_size_rnn, num_layers=args.num_layers, has_input=True,
                        has_output=True, output_size=args.hidden_size_rnn_output).to(choose_device())
    output = GRU_plain(input_size=1, embedding_size=args.embedding_size_rnn_output,
                           hidden_size=args.hidden_size_rnn_output, num_layers=args.num_layers, has_input=True,
                           has_output=True, output_size=1).to(choose_device()) # if you can't figure how to train with GRU use MLP
    # check if load existing model
    if args.load:
        fname = args.model_save_path + args.fname + 'lstm_' + str(args.load_epoch) + '.dat'
        rnn.load_state_dict(torch.load(fname))
        fname = args.model_save_path + args.fname + 'output_' + str(args.load_epoch) + '.dat'
        output.load_state_dict(torch.load(fname))

        args.lr = 0.00001
        epoch = args.load_epoch
        print('model loaded!, lr: {}'.format(args.lr))
    else:
        epoch = 1

    # initialize optimizer
    optimizer_rnn = optim.Adam(list(rnn.parameters()), lr=args.lr)
    optimizer_output = optim.Adam(list(output.parameters()), lr=args.lr)

    scheduler_rnn = MultiStepLR(optimizer_rnn, milestones=args.milestones, gamma=args.lr_rate)
    scheduler_output = MultiStepLR(optimizer_output, milestones=args.milestones, gamma=args.lr_rate)

    # start main loop
    time_all = np.zeros(args.epochs)
    while epoch<=args.epochs:
        time_start = tm.time()
        # train
        if 'GraphRNN_VAE' in args.note:
            train_vae_epoch(epoch, args, rnn, output, batch_index, data,
                            optimizer_rnn, optimizer_output,
                            scheduler_rnn, scheduler_output)
        elif 'GraphRNN_MLP' in args.note:
            train_mlp_epoch(epoch, args, rnn, output, batch_index, data,
                            optimizer_rnn, optimizer_output,
                            scheduler_rnn, scheduler_output)
        elif 'GraphRNN_RNN' in args.note:
            train_rnn_epoch(epoch, args, rnn, output, batch_index, data,
                            optimizer_rnn, optimizer_output,
                            scheduler_rnn, scheduler_output)
        time_end = tm.time()
        time_all[epoch - 1] = time_end - time_start
        # test
        if epoch % args.epochs_test == 0 and epoch>=args.epochs_test_start:
            for sample_time in range(1,4):
                G_pred = []
                while len(G_pred)<args.test_total_size:
                    if 'GraphRNN_VAE' in args.note:
                        G_pred_step = test_vae_epoch(epoch, args, rnn, output, test_batch_size=args.test_batch_size,sample_time=sample_time)
                    elif 'GraphRNN_MLP' in args.note:
                        G_pred_step = test_mlp_epoch(epoch, args, rnn, output, test_batch_size=args.test_batch_size,sample_time=sample_time)
                    elif 'GraphRNN_RNN' in args.note:
                        G_pred_step = test_rnn_epoch(epoch, args, rnn, output, test_batch_size=args.test_batch_size)
                    G_pred.extend(G_pred_step)
                # save graphs
                fname = args.graph_save_path + args.fname_pred + str(epoch) +'_'+str(sample_time) + '.dat'
                save_graph_list(G_pred, fname)
                if 'GraphRNN_RNN' in args.note:
                    break
            print('test done, graphs saved')


        # save model checkpoint
        if args.save:
            if epoch % args.epochs_save == 0:
                fname = args.model_save_path + args.fname + 'lstm_' + str(epoch) + '.dat'
                torch.save(rnn.state_dict(), fname)
                fname = args.model_save_path + args.fname + 'output_' + str(epoch) + '.dat'
                torch.save(output.state_dict(), fname)
        epoch += 1

    np.save(args.timing_save_path+args.fname,time_all)

    # generate one graph
    return test_rnn_epoch(epoch, args, rnn, output, test_batch_size=1)


#==============GraphVAE, GraphRNN baseline=====================#
class GraphVAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, max_num_nodes, pool='sum'):
        '''
        Args:
            input_dim: input feature dimension for node.
            hidden_dim: hidden dim for 2-layer gcn.
            latent_dim: dimension of the latent representation of graph.
        '''
        super(GraphVAE, self).__init__()
        self.conv1 = model.GraphConv(input_dim=input_dim, output_dim=hidden_dim)
        self.bn1 = nn.BatchNorm1d(input_dim)
        self.conv2 = model.GraphConv(input_dim=hidden_dim, output_dim=hidden_dim)
        self.bn2 = nn.BatchNorm1d(input_dim)
        self.act = nn.ReLU()

        output_dim = max_num_nodes * (max_num_nodes + 1) // 2
        #self.vae = model.MLP_VAE_plain(hidden_dim, latent_dim, output_dim)
        self.vae = model.MLP_VAE_plain(input_dim * input_dim, latent_dim, output_dim)
        #self.feature_mlp = model.MLP_plain(latent_dim, latent_dim, output_dim)

        self.max_num_nodes = max_num_nodes
        for m in self.modules():
            if isinstance(m, model.GraphConv):
                m.weight.data = init.xavier_uniform(m.weight.data, gain=nn.init.calculate_gain('relu'))
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        self.pool = pool

    def recover_adj_lower(self, l):
        # NOTE: Assumes 1 per minibatch
        adj = torch.zeros(self.max_num_nodes, self.max_num_nodes)
        adj[torch.triu(torch.ones(self.max_num_nodes, self.max_num_nodes)) == 1] = l
        return adj

    def recover_full_adj_from_lower(self, lower):
        diag = torch.diag(torch.diag(lower, 0))
        return lower + torch.transpose(lower, 0, 1) - diag

    def edge_similarity_matrix(self, adj, adj_recon, matching_features,
                matching_features_recon, sim_func):
        S = torch.zeros(self.max_num_nodes, self.max_num_nodes,
                        self.max_num_nodes, self.max_num_nodes)
        for i in range(self.max_num_nodes):
            for j in range(self.max_num_nodes):
                if i == j:
                    for a in range(self.max_num_nodes):
                        S[i, i, a, a] = adj[i, i] * adj_recon[a, a] * \
                                        sim_func(matching_features[i], matching_features_recon[a])
                        # with feature not implemented
                        # if input_features is not None:
                else:
                    for a in range(self.max_num_nodes):
                        for b in range(self.max_num_nodes):
                            if b == a:
                                continue
                            S[i, j, a, b] = adj[i, j] * adj[i, i] * adj[j, j] * \
                                            adj_recon[a, b] * adj_recon[a, a] * adj_recon[b, b]
        return S

    def mpm(self, x_init, S, max_iters=50):
        x = x_init
        for it in range(max_iters):
            x_new = torch.zeros(self.max_num_nodes, self.max_num_nodes)
            for i in range(self.max_num_nodes):
                for a in range(self.max_num_nodes):
                    x_new[i, a] = x[i, a] * S[i, i, a, a]
                    pooled = [torch.max(x[j, :] * S[i, j, a, :])
                              for j in range(self.max_num_nodes) if j != i]
                    neigh_sim = sum(pooled)
                    x_new[i, a] += neigh_sim
            norm = torch.norm(x_new)
            x = x_new / norm
        return x

    def deg_feature_similarity(self, f1, f2):
        return 1 / (abs(f1 - f2) + 1)

    def permute_adj(self, adj, curr_ind, target_ind):
        ''' Permute adjacency matrix.
          The target_ind (connectivity) should be permuted to the curr_ind position.
        '''
        # order curr_ind according to target ind
        ind = np.zeros(self.max_num_nodes, dtype=np.int)
        ind[target_ind] = curr_ind
        adj_permuted = torch.zeros((self.max_num_nodes, self.max_num_nodes))
        adj_permuted[:, :] = adj[ind, :]
        adj_permuted[:, :] = adj_permuted[:, ind]
        return adj_permuted

    def pool_graph(self, x):
        if self.pool == 'max':
            out, _ = torch.max(x, dim=1, keepdim=False)
        elif self.pool == 'sum':
            out = torch.sum(x, dim=1, keepdim=False)
        return out

    def forward(self, input_features, adj):
        #x = self.conv1(input_features, adj)
        #x = self.bn1(x)
        #x = self.act(x)
        #x = self.conv2(x, adj)
        #x = self.bn2(x)

        # pool over all nodes
        #graph_h = self.pool_graph(x)
        graph_h = input_features.view(-1, self.max_num_nodes * self.max_num_nodes)
        # vae
        h_decode, z_mu, z_lsgms = self.vae(graph_h)
        out = F.sigmoid(h_decode)
        out_tensor = out.cpu().data
        recon_adj_lower = self.recover_adj_lower(out_tensor)
        recon_adj_tensor = self.recover_full_adj_from_lower(recon_adj_lower)

        # set matching features be degree
        out_features = torch.sum(recon_adj_tensor, 1)

        adj_data = adj.cpu().data[0]
        adj_features = torch.sum(adj_data, 1)

        S = self.edge_similarity_matrix(adj_data, recon_adj_tensor, adj_features, out_features,
                self.deg_feature_similarity)

        # initialization strategies
        init_corr = 1 / self.max_num_nodes
        init_assignment = torch.ones(self.max_num_nodes, self.max_num_nodes) * init_corr
        #init_assignment = torch.FloatTensor(4, 4)
        #init.uniform(init_assignment)
        assignment = self.mpm(init_assignment, S)
        #print('Assignment: ', assignment)

        # matching
        # use negative of the assignment score since the alg finds min cost flow
        row_ind, col_ind = scipy.optimize.linear_sum_assignment(-assignment.numpy())
        print('row: ', row_ind)
        print('col: ', col_ind)
        # order row index according to col index
        #adj_permuted = self.permute_adj(adj_data, row_ind, col_ind)
        adj_permuted = adj_data
        adj_vectorized = adj_permuted[torch.triu(torch.ones(self.max_num_nodes,self.max_num_nodes) )== 1].squeeze_()
        adj_vectorized_var = Variable(adj_vectorized).cuda()

        #print(adj)
        #print('permuted: ', adj_permuted)
        #print('recon: ', recon_adj_tensor)
        adj_recon_loss = self.adj_recon_loss(adj_vectorized_var, out[0])
        print('recon: ', adj_recon_loss)
        print(adj_vectorized_var)
        print(out[0])

        loss_kl = -0.5 * torch.sum(1 + z_lsgms - z_mu.pow(2) - z_lsgms.exp())
        loss_kl /= self.max_num_nodes * self.max_num_nodes # normalize
        print('kl: ', loss_kl)

        loss = adj_recon_loss + loss_kl

        return loss

    def forward_test(self, input_features, adj):
        self.max_num_nodes = 4
        adj_data = torch.zeros(self.max_num_nodes, self.max_num_nodes)
        adj_data[:4, :4] = torch.FloatTensor([[1,1,0,0], [1,1,1,0], [0,1,1,1], [0,0,1,1]])
        adj_features = torch.Tensor([2,3,3,2])

        adj_data1 = torch.zeros(self.max_num_nodes, self.max_num_nodes)
        adj_data1 = torch.FloatTensor([[1,1,1,0], [1,1,0,1], [1,0,1,0], [0,1,0,1]])
        adj_features1 = torch.Tensor([3,3,2,2])
        S = self.edge_similarity_matrix(adj_data, adj_data1, adj_features, adj_features1,
                self.deg_feature_similarity)

        # initialization strategies
        init_corr = 1 / self.max_num_nodes
        init_assignment = torch.ones(self.max_num_nodes, self.max_num_nodes) * init_corr
        #init_assignment = torch.FloatTensor(4, 4)
        #init.uniform(init_assignment)
        assignment = self.mpm(init_assignment, S)
        #print('Assignment: ', assignment)

        # matching
        row_ind, col_ind = scipy.optimize.linear_sum_assignment(-assignment.numpy())
        print('row: ', row_ind)
        print('col: ', col_ind)

        permuted_adj = self.permute_adj(adj_data, row_ind, col_ind)
        print('permuted: ', permuted_adj)

        adj_recon_loss = self.adj_recon_loss(permuted_adj, adj_data1)
        print(adj_data1)
        print('diff: ', adj_recon_loss)

    def adj_recon_loss(self, adj_truth, adj_pred):
        return F.binary_cross_entropy(adj_truth, adj_pred)