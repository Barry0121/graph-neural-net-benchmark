import torch
from ot.gromov import gromov_wasserstein

def GWLoss(adj_o, adj_r):
    """
    adj_o: original adjacency matrix
    adj_r: reconstructed adjacency matrix
    """
    # initialize measures for each metric measure network
    # p_o = ot.unif(adj_o.shape[0])
    p_o = torch.ones(adj_o.shape[0])/adj_o.shape[0]
    p_r = torch.ones(adj_r.shape[0])/adj_r.shape[0]
    # compute GW distance
    _, log = gromov_wasserstein(
        adj_o, adj_r, p_o, p_r,
        'square_loss', verbose=False, log=True)
    # print(log.size())
    return log['gw_dist']