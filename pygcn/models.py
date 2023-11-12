import torch.nn as nn
import torch.nn.functional as F
from pygcn.layers import GraphConvolution, CRF, Pool, Upool, CRF_NN


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.gcpool = GraphConvolution(nhid, nhid)
        self.crf = CRF(nhid, nhid," ")
        self.pool = Pool(nhid, 0.5, 2)
        self.upool = Upool(nhid,0.6)
        self.dropout = dropout
        self.crf_nn = CRF_NN(nhid,nhid,2)

    def forward(self, x, adj):
        n = x.size(0)
        x = F.relu(self.gc1(x, adj))

        x = self.crf_nn(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)

        return F.log_softmax(x, dim=1)