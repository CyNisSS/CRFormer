import math

import torch

import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.nn.functional as F

BIG_CONSTANT = 1e8


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
        # print(gi.shape)

        alpha = torch.exp(self.alpha)
        beta = torch.exp(self.beta)
        for i in range(self.num_iters):
            output = (alpha * x + beta * gi * output)
            output = output / (alpha + beta * gi)
        print(alpha)
        print(beta)
        return output


def create_projection_matrix(m, d, seed=0, scaling=0, struct_mode=False):
    nb_full_blocks = int(m / d)
    block_list = []
    current_seed = seed
    for _ in range(nb_full_blocks):
        torch.manual_seed(current_seed)
        if struct_mode:
            q = create_products_of_givens_rotations(d, current_seed)
        else:
            unstructured_block = torch.randn((d, d))
            q, _ = torch.linalg.qr(unstructured_block)
            q = torch.t(q)
        block_list.append(q)
        current_seed += 1
    remaining_rows = m - nb_full_blocks * d
    if remaining_rows > 0:
        torch.manual_seed(current_seed)
        if struct_mode:
            q = create_products_of_givens_rotations(d, current_seed)
        else:
            unstructured_block = torch.randn((d, d))
            q, _ = torch.linalg.qr(unstructured_block)
            q = torch.t(q)
        block_list.append(q[0:remaining_rows])
    final_matrix = torch.vstack(block_list)

    current_seed += 1
    torch.manual_seed(current_seed)
    if scaling == 0:
        multiplier = torch.norm(torch.randn((m, d)), dim=1)
    elif scaling == 1:
        multiplier = torch.sqrt(torch.tensor(float(d))) * torch.ones(m)
    else:
        raise ValueError("Scaling must be one of {0, 1}. Was %s" % scaling)

    return torch.matmul(torch.diag(multiplier), final_matrix)


def softmax_kernel_transformation(data, is_query, projection_matrix=None, numerical_stabilizer=0.000001):
    data_normalizer = 1.0 / torch.sqrt(torch.sqrt(torch.tensor(data.shape[-1], dtype=torch.float32)))
    data = data_normalizer * data
    ratio = 1.0 / torch.sqrt(torch.tensor(projection_matrix.shape[0], dtype=torch.float32))  # 1/√m
    data_dash = torch.einsum("bnhd,md->bnhm", data, projection_matrix)
    diag_data = torch.square(data)
    diag_data = torch.sum(diag_data, dim=len(data.shape) - 1)
    diag_data = diag_data / 2.0
    diag_data = torch.unsqueeze(diag_data, dim=len(data.shape) - 1)
    last_dims_t = len(data_dash.shape) - 1
    attention_dims_t = len(data_dash.shape) - 3
    if is_query:
        data_dash = ratio * (
                torch.exp(data_dash - diag_data - torch.max(data_dash, dim=last_dims_t, keepdim=True)[
                    0]) + numerical_stabilizer
        )
    else:
        data_dash = ratio * (
                torch.exp(data_dash - diag_data - torch.max(torch.max(data_dash, dim=last_dims_t, keepdim=True)[0],
                                                            dim=attention_dims_t, keepdim=True)[
                    0]) + numerical_stabilizer
        )
    return data_dash


def kernelized_softmax(query, key, value, kernel_transformation, tau, projection_matrix=None, edge_index=None,  return_weight=False):
    '''
    fast computation of all-pair attentive aggregation with linear complexity
    input: query/key/value [B, N, H, D]
    return: updated node emb, attention weight (for computing edge loss)
    B = graph number (always equal to 1 in Node Classification), N = node number, H = head number,
    M = random feature dimension, D = hidden size
    '''
    query = query / math.sqrt(tau)
    key = key / math.sqrt(tau)
    query_prime = kernel_transformation(query, True, projection_matrix)  # [B, N, H, M]
    key_prime = kernel_transformation(key, False, projection_matrix)  # [B, N, H, M]
    query_prime = query_prime.permute(1, 0, 2, 3)  # [N, B, H, M]
    key_prime = key_prime.permute(1, 0, 2, 3)  # [N, B, H, M]
    value = value.permute(1, 0, 2, 3)  # [N, B, H, D]

    # compute updated node emb, this step requires O(N)
    z_num = numerator(query_prime, key_prime, value)
    z_den = denominator(query_prime, key_prime)

    z_num = z_num.permute(1, 0, 2, 3)  # [B, N, H, D]
    z_den = z_den.permute(1, 0, 2)
    z_den = torch.unsqueeze(z_den, len(z_den.shape))
    z_output = z_num / z_den  # [B, N, H, D]

    if return_weight:  # query edge prob for computing edge-level reg loss, this step requires O(E)
        start, end = edge_index
        query_end, key_start = query_prime[end], key_prime[start]  # [E, B, H, M]
        edge_attn_num = torch.einsum("ebhm,ebhm->ebh", query_end, key_start)  # [E, B, H]
        edge_attn_num = edge_attn_num.permute(1, 0, 2)  # [B, E, H]
        attn_normalizer = denominator(query_prime, key_prime)  # [N, B, H]
        edge_attn_dem = attn_normalizer[end]  # [E, B, H]
        edge_attn_dem = edge_attn_dem.permute(1, 0, 2)  # [B, E, H]
        A_weight = edge_attn_num / edge_attn_dem  # [B, E, H]

        return z_output, A_weight

    else:
        return z_output, z_den


def numerator_gumbel(qs, ks, vs):
    kvs = torch.einsum("nbhkm,nbhd->bhkmd", ks, vs)  # kvs refers to U_k in the paper
    return torch.einsum("nbhm,bhkmd->nbhkd", qs, kvs)


def denominator_gumbel(qs, ks):
    all_ones = torch.ones([ks.shape[0]]).to(qs.device)
    ks_sum = torch.einsum("nbhkm,n->bhkm", ks, all_ones)  # ks_sum refers to O_k in the paper
    return torch.einsum("nbhm,bhkm->nbhk", qs, ks_sum)


def kernelized_gumbel_softmax(query, key, value, kernel_transformation, tau, K=10, projection_matrix=None, edge_index=None, return_weight=False):
    '''
    fast computation of all-pair attentive aggregation with linear complexity
    input: query/key/value [B, N, H, D]
    return: updated node emb, attention weight (for computing edge loss)
    B = graph number (always equal to 1 in Node Classification), N = node number, H = head number,
    M = random feature dimension, D = hidden size, K = number of Gumbel sampling
    '''
    query = query / math.sqrt(tau)
    key = key / math.sqrt(tau)
    query_prime = kernel_transformation(query, True, projection_matrix)  # [B, N, H, M]
    key_prime = kernel_transformation(key, False, projection_matrix)  # [B, N, H, M]
    query_prime = query_prime.permute(1, 0, 2, 3)  # [N, B, H, M]
    key_prime = key_prime.permute(1, 0, 2, 3)  # [N, B, H, M]
    value = value.permute(1, 0, 2, 3)  # [N, B, H, D]

    # compute updated node emb, this step requires O(N)
    gumbels = (
                  -torch.empty(key_prime.shape[:-1] + (K,),
                               memory_format=torch.legacy_contiguous_format).exponential_().log()
              ).to(query.device) / tau  # [N, B, H, K]
    key_t_gumbel = key_prime.unsqueeze(3) * gumbels.exp().unsqueeze(4)  # [N, B, H, K, M]
    z_num = numerator_gumbel(query_prime, key_t_gumbel, value)  # [N, B, H, K, D]
    z_den = denominator_gumbel(query_prime, key_t_gumbel)  # [N, B, H, K]

    z_num = z_num.permute(1, 0, 2, 3, 4)  # [B, N, H, K, D]
    z_den = z_den.permute(1, 0, 2, 3)  # [B, N, H, K]
    z_den = torch.unsqueeze(z_den, len(z_den.shape))  # [B,N,H,k,1]

    #z_output = z_num
    z_output = torch.mean(z_num / z_den, dim=3)  # [B, N, H, D]#num_heads 的聚合

    if return_weight:  # query edge prob for computing edge-level reg loss, this step requires O(E)
        start, end = edge_index
        query_end, key_start = query_prime[end], key_prime[start]  # [E, B, H, M]
        edge_attn_num = torch.einsum("ebhm,ebhm->ebh", query_end, key_start)  # [E, B, H]
        edge_attn_num = edge_attn_num.permute(1, 0, 2)  # [B, E, H]
        attn_normalizer = denominator(query_prime, key_prime)  # [N, B, H]
        edge_attn_dem = attn_normalizer[end]  # [E, B, H]
        edge_attn_dem = edge_attn_dem.permute(1, 0, 2)  # [B, E, H]
        A_weight = edge_attn_num / edge_attn_dem  # [B, E, H]

        return z_output, A_weight

    else:
        return z_output, z_den


def numerator(qs, ks, vs):
    kvs = torch.einsum("nbhm,nbhd->bhmd", ks, vs)  # kvs refers to U_k in the paper
    return torch.einsum("nbhm,bhmd->nbhd", qs, kvs)


def denominator(qs, ks):
    all_ones = torch.ones([ks.shape[0]]).to(qs.device)
    ks_sum = torch.einsum("nbhm,n->bhm", ks, all_ones)  # ks_sum refers to O_k in the paper
    return torch.einsum("nbhm,bhm->nbh", qs, ks_sum)


class CRF_Node(Module):
    def __init__(self, in_dim, out_dim, hidden=32, num_iters=2, num_heads=8,
                 kernel_transformation=softmax_kernel_transformation,
                 projection_matrix_type='a', random_features=64, nb_gumbel_sample=10, use_gumbel=True, **kwargs):
        super(CRF_Node, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden = hidden
        self.num_iters = num_iters
        #        self.adj = adj
        self.alpha = nn.Parameter(torch.zeros(1))
        self.beta = nn.Parameter(torch.zeros(1))
        self.num_heads = num_heads
        self.Wq = nn.Linear(in_dim, num_heads * hidden)
        self.Wk = nn.Linear(in_dim, num_heads * hidden)
        self.Wv = nn.Linear(in_dim, num_heads * hidden)
        self.Wo = nn.Linear(num_heads * hidden, out_dim)
        self.kernel_transformation = kernel_transformation
        self.projection_matrix_type = projection_matrix_type
        self.random_features = random_features
        self.nb_gumbel_sample = nb_gumbel_sample
        self.use_gumbel = use_gumbel

    def reset_parameters(self):
        self.Wq.reset_parameters()
        self.Wk.reset_parameters()
        self.Wv.reset_parameters()
        self.Wo.reset_parameters()

    def forward(self, x, adj, tau=0.25):

        N, D = x.size(0), x.size(1)
        #output = x.reshape(-1,N,self.num_heads,self.hidden)#.unsqueeze(-2).expand(-1,-1,-1,10,-1) [B,N,H,D]
        output = x
        print("origin x : ", output.shape)
        query = self.Wq(x).reshape(-1, N, self.num_heads, self.hidden)
        key = self.Wk(x).reshape(-1, N, self.num_heads, self.hidden)
        value = self.Wv(x).reshape(-1, N, self.num_heads, self.hidden)

        seed = torch.ceil(torch.abs(torch.sum(query) * BIG_CONSTANT)).to(torch.int32)
        projection_matrix = create_projection_matrix(self.random_features, self.hidden,
                                                     seed=seed)  # random feature x hidden

        if self.use_gumbel and self.training:  # only using Gumbel noise for training
            x_next, x_den = kernelized_gumbel_softmax(query, key, value, self.kernel_transformation, tau, self.nb_gumbel_sample, projection_matrix,
                                                      adj)
            x_den = torch.mean(x_den, dim=3)
            # x_den = [B, N, H, K]#能不能把 x-den 通过Linear变成->[B,N,Hxk] - > [B,N,K]
        else:
            x_next, x_den = kernelized_softmax(query, key, value, self.kernel_transformation,tau, projection_matrix, adj
                                               )

        # x_next, x_den = kernelized_softmax(query, key, value, self.kernel_transformation, projection_matrix, adj, tau)
        x_next = self.Wo(x_next.flatten(-2, -1)).squeeze(0)  # 1x N x (h*d) -> Nxout_dim

        # print(x_den.shape)
        # x_den = x_den.flatten(-2,-1).squeeze(0)
        # x_den = self.alpha + self.beta * x_den
        alpha = torch.exp(self.alpha)
        beta = torch.exp(self.beta)
        #print('x_den',x_den.shape)
        #print('x-next',x_next.shape)
        for i in range(self.num_iters):
            #output = torch.mean(((alpha * output + beta * x_next)/(alpha + beta * x_den)), dim=3)#[B, N, H, K, D]
            output = (alpha * output + beta * x_next)
            #output = (alpha * output + beta * x_next)/(alpha + beta * x_den)
        #torch.mean(output, dim=3) #[B, N, H, K, D]


        #output = self.Wo(x_next.flatten(-2, -1)).squeeze(0)
        print(alpha)
        print(beta)

        '''x1 = torch.matmul(x, self.w_a)
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
            output = output / (alpha + beta * gi)'''
        return output
