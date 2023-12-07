import math

import torch

import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.nn.functional as F


class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
            + str(self.in_features) + ' -> ' \
            + str(self.out_features) + ')'

'''class Pool(Module):
    def __init__(self, in_feature, top_k_ratio, a_order):
        super(Pool, self).__init__()
        self.y = None
        self.top_k_ratio = top_k_ratio
        self.in_feature = in_feature
        # self.adj = adj
        self.a_order = a_order

        self.parm = nn.Parameter(torch.FloatTensor(in_feature, 1))
        self.act = nn.Sigmoid()

        self.reset_parameter()

    def reset_parameter(self):
        stv = 0.95 / math.sqrt(self.parm.size(0))
        self.parm.data.uniform_(-stv, stv)

    """获得y中的top k， 并返回相应的idx，idx从小到大，保持adj原有顺序"""

    def get_top_k(self, x):
        p = F.normalize(self.parm, p=2, dim=0)
        # print('p:',p)
        self.y = torch.mm(x, self.parm)
        # print(y.shape)
        # print(int(x.shape[0] * self.top_k_ratio))
        _, idx = torch.topk(self.y, int(x.shape[0] * self.top_k_ratio), dim=0)
        # print('idx',idx)
        sort_indices = torch.argsort(idx, dim=0)
        # print('sort:',sort_indices)
        top_k_values = self.y[idx[sort_indices]]
        idx_k = idx[sort_indices].view(-1, 1).squeeze()
        # print("idxk_shape:",idx_k.shape)
        return idx_k

    def forward(self, x, adj):
        adj = adj.to_dense()
        # print(adj.shape)
        idx = self.get_top_k(x)
        # print('idx',idx)

        y_hat = self.act(self.y[idx, :])
        x_hat = x[idx, :]
        new_adj_r = adj[idx]
        # print(new_adj_r.shape)
        ###new_adj_c = self.adj[:, idx]
        new_adj = new_adj_r[:, idx]
        # print("new_adj",new_adj)
        # print("new_adj",new_adj.shape)

        if self.a_order > 1:
            for i in range(self.a_order - 1):
                new_adj = torch.mm(new_adj, new_adj)
        ones = torch.ones_like(torch.empty(self.in_feature), dtype=torch.float).view(1, -1)
        # print("xhat",x_hat.shape)
        # print("ones",ones.shape)
        # print("yhat",y_hat.shape)

        x = x_hat * (torch.mm(y_hat, ones))
        new_adj = new_adj.to_sparse_coo()
        return x, idx, new_adj


class Upool(Module):
    def __init__(self, in_feature, top_k_ratio):
        super(Upool, self).__init__()
        self.in_feature = in_feature
        # self.idx = idx
        self.ratio = 1 / top_k_ratio

    def forward(self, x, idx, adj, num_nodes):
        # n = int(x.size(0) * self.ratio)
        adj = adj.to_dense()
        zeros = torch.zeros(num_nodes, self.in_feature)
        """idx的范围 >> 此时x.size(0)= kxN, 因此在x中取值要从新找到符合范围的idx_idx"""
        idx_idx = torch.arange(idx.size(0))
        # print('idxidx',idx_idx.shape)
        zeros[idx] = x[idx_idx]  # 2708x16
        ##print('zeros',zeros.shape)
        """恢复adj的大小"""
        #
        # print('idx:::',idx.shape)
        adj_zeros = torch.zeros(num_nodes, num_nodes)  # 2708x2708
        # print('adjzeros',adj_zeros.shape)#2708x2708
        # print('adj[idx_idx]',adj[idx_idx].shape)#1895x1895
        # print('adj_zeros[idx]',adj_zeros[idx].shape)
        # 张量形状不匹配
        for i in range(idx.size(0)):
            for j in range(idx.size(0)):
                adj_zeros[idx[i], idx[j]] = adj[idx_idx[i], idx_idx[j]]
        x = zeros
        adj_zeros.to_sparse_coo()

        return x, adj_zeros
'''


"""自己加的CRF模块，用于构建经过gcn卷积得到的隐藏层之间相似的表示关系"""
'''
class CRF(Module):
    def __init__(self, infeature, outfeature, mode='None'):
        super(CRF, self).__init__()

        # self.infeature = infeature
        self.g_matrix = None
        self.s_matrix = None
        # self.outfeature = outfeature
        # self.mode = mode
        self.alpha = nn.Parameter(torch.FloatTensor(1))
        self.beta = nn.Parameter(torch.FloatTensor(1))
        self.W_a = nn.Parameter(torch.FloatTensor(infeature, outfeature))
        self.W_b = nn.Parameter(torch.FloatTensor(infeature, outfeature))
        """
        self.fc1 = nn.Linear(infeature, outfeature)
        self.fc2 = nn.Linear(infeature, outfeature)"""
        self.sftmx = nn.Softmax(1)
        self.reset_parameters()

    def reset_parameters(self):
        # 不同的初始化结果得到不同的值，#下得到82-83%
        # stdv1 = 0.95 / math.sqrt(self.W_a.size(1))
        # self.W_a.data.uniform_(-stdv1, stdv1)
        # stdv2 = 0.95 / math.sqrt(self.W_b.size(1))
        # self.W_b.data.uniform_(-stdv2, stdv2)
        # 正太分布的初始化会使模型不稳定，寻找原因
        self.W_a = nn.init.xavier_normal_(self.W_a, gain=1)
        self.W_b = nn.init.xavier_normal_(self.W_b, gain=1)

        stdv_a = 0.21 / math.sqrt(self.alpha.size(0))
        nn.init.constant_(self.alpha, stdv_a)

        stdv_b = 0.21 / math.sqrt(self.beta.size(0))
        nn.init.constant_(self.beta, stdv_b)

    # def compute_gij(self,x):

    def forward(self, x):
        """compute sij & gij"""
        a = torch.mm(x, self.W_a)
        b = torch.mm(x, self.W_b).t()

        self.s_matrix = torch.mm(a, b)
        self.g_matrix = self.sftmx(self.s_matrix)

        x = self.alpha * x
        """后一项是需要迭代的Hk，为了得到Hk+1，此处用x暂时代替Hk"""
        x = x + self.beta * self.g_matrix.sum(1, keepdim=True) * x
        x = x / (self.alpha + self.beta)

        return x

'''

class CRF_NN(Module):
    def __init__(self, in_dim, out_dim, num_iters, **kwargs):
        super(CRF_NN, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_iters = num_iters
#        self.adj = adj
        self.alpha = nn.Parameter(torch.zeros(1))
        self.beta = nn.Parameter(torch.zeros(1))
        self.w_a = nn.Parameter(torch.nn.init.xavier_uniform_(torch.empty(in_dim, out_dim)))
        self.w_b = nn.Parameter(torch.nn.init.xavier_uniform_(torch.empty(in_dim, out_dim)))

    def forward(self, x):
        output = x
        x1 = torch.matmul(x, self.w_a)
        x2 = torch.matmul(x, self.w_b)

        logits = torch.matmul(x1, x2.t())
        similarity = F.softmax(F.leaky_relu(logits), dim=1)
        gi = torch.sum(similarity, dim=1, keepdim=True)
        gi = F.normalize(gi)
        print(gi.shape)

        alpha = torch.exp(self.alpha)
        beta = torch.exp(self.beta)
        for i in range(self.num_iters):
            output = (alpha * x + beta * gi * output)
            output = output / (alpha + beta * gi)
        return output
