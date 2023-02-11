#####################################################################################
# Graph Neural Network Components Module                                            #
#                                                                                   #
# Description:                                                                      #
# This module contains the different graph neural network components, including     #
# models, custom layers, and different optimizer and loss functions.                #
#                                                                                   #
# Code References:                                                                  #
# 1.                                                                                #
#####################################################################################

# Standard pacakges
import torch
from torch.nn import Linear, Parameter, Sequential, ReLU

# Pytroch Geometric
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree

"""
Graph Convolution Layer with MessagePassing
Source with explanation:
    https://zqfang.github.io/2021-08-07-graph-pyg/
    https://pytorch-geometric.readthedocs.io/en/latest/notes/create_gnn.html#the-messagepassing-base-class
"""
class GCNConv(MessagePassing):
    def __init__(self,
                input_dim,
                output_dim,
                device='cuda:0' if torch.cuda.is_available() else 'cpu'):
        super(GCNConv, self).__init__(aggr='add')

        """
        Parameters:
            input_dim: int, size of the input node features.
            output_dim: int, size of the output node feature encodings.
        """
        self.device = device # initialize the hosting device
        self.linear = Linear(input_dim, output_dim, device=self.device, bias=False)
        self.bias = Parameter(torch.Tensor(output_dim))
        self.reset_parameters()

    def forward(self, X, edge_index):
        """
        x has shape [N, in_channels]
        edge_index has shape [2, E]
        """
        # no need for the second edge_attribute list
        edge_index, _ = add_self_loops(edge_index, num_nodes=X.size[0])
        X = self.linear(X)

        # compute normalization
        row, col = edge_index
        deg = degree(col, X.size(0), dtype=X.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0 # eliminate 'inf' error
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        output = self.propagate(edge_index=edge_index, size=None, x=X, norm=norm)
        # note: 'size' is the inferred size of the adjacency matrix; this can be rectangular

        output += self.bias
        return output

    def message(self, x_j, norm):
        """
        x_j has shape [E, out_channels]
        """
        return norm.view(-1, 1) * x_j

    def reset_parameters(self):
        """
        Util function
        """
        self.lin.reset_parameters()
        self.bias.data.zero_()


class EdgeConv(MessagePassing):
    """
    Reference:
        https://pytorch-geometric.readthedocs.io/en/latest/notes/create_gnn.html#implementing-the-edge-convolution
    """
    def __init__(self,
                input_dim,
                output_dim,
                device='cuda:0' if torch.cuda.is_available() else 'cpu'):
        super(EdgeConv).__init__(aggr='max')

        self.device = device
        self.mlp = Sequential(
            Linear(2*input_dim, output_dim), # 2 times the input dimension because of concatenation
            ReLU(),
            Linear(output_dim, output_dim)
        )

    def forward(self, X, edge_index):
        return self.propagate(edge_index, x=X)

    def message(self, X_i, X_j):
        tmp = torch.cat([X_i, X_j - X_i], dim=1)  # tmp has shape [E, 2 * in_channels]
        return self.mlp(tmp)

"""
Graph Convolution Layer without message passing mechanism
"""
# class GCNConvCustom(nn.Module):
#     def __init__(self,
#                 input_dim,
#                 output_dim,
#                 scale=1,
#                 random_init=True,
#                 with_bias=True,
#                 device='cuda:0' if torch.cuda.is_available() else 'mps'):
#         super(GCNConvCustom, self).__init__()

#         """Metadata"""
#         self.device = device # initialize the hosting device
#         self.with_bias = with_bias
#         self.scale = scale

#         self.As = None

#         # initialize learnable weights
#         # the weight should have shape of (N , F) where N is the size of the input, and F is the output dimension
#         self.W, self.b = None, None
#         if random_init:

#             self.W = torch.nn.Parameter(
#                 data=(2 * torch.rand(input_dim, output_dim, device=self.device)-1)*self.scale,
#                 requires_grad=True
#             )
#             # create trainable a bias term for the layer
#             self.b = torch.nn.Parameter(
#                 data=(2 * torch.rand(output_dim, 1, device=self.device)-1)*self.scale,
#                 requires_grad=True
#             )
#         else:
#             self.W = torch.nn.Parameter(
#                 data=torch.zeros(input_dim, output_dim, device=self.device),
#                 requires_grad=True
#             )
#             self.b = torch.nn.Parameter(
#                 data=torch.zeros(output_dim, 1, device=self.device),
#                 requires_grad=True
#             )

#     def forward(self, H, edge_list):
#         self.A_s = self.get_normalized_adj(edge_list)

#         if self.with_bias:
#             return self.A_s @ (H @ self.W) + self.b.T
#         else:
#             return self.A_s @ (H @ self.W)

#     def get_normalized_adj(self, edge_list):
#         """Calculate Matrices"""
#         # the adjacency matrix with self-loop

#         self.A = gutils.to_dense_adj(edge_list).to(self.device)[0]
#         self.A_self = self.A + torch.diag(torch.ones(self.A.shape[0], device=self.device))

#         # calculate the degree matrix with A after added self loop
#         self.D = torch.sum(self.A_self, dim=0).to(self.device)  # Note: these are the elements along the diagonal of D

#         # for diagonal matrix, raising it to any power is the same as raising its diagonal elements to that power
#         # we can just apply the -1/2 power to all element of this degree matrix
#         self.D_half_norm = torch.diag(torch.pow(self.D, -0.5))

#         # normalized adjacency matrix
#         self.A_s = self.D_half_norm @ self.A_self @ self.D_half_norm
#         self.A_s = self.A_s.to(self.device)
#         return self.A_s