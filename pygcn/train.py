from __future__ import division
from __future__ import print_function

import time
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from scipy.linalg import fractional_matrix_power
from torch_geometric.utils import to_undirected, remove_self_loops, add_self_loops


from pygcn.utils import load_data, accuracy,load_dataset,load_citation_data
from pygcn.models import GCN

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=200,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=64,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.1,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--dataname', type=str, default='citeseer',
                    help='dataset name')
parser.add_argument('--data_dir', type=str, default='../data/',
                    help='data dir')
parser.add_argument('--tau', type=float, default=0.25,
                    help='temperature in gumbel ')


args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

dataset = load_dataset(args.data_dir, args.dataname)

adj_norm, adj, features, labels, idx_train, idx_val, idx_test = load_citation_data(dataset)


# generate A_hat
'''adj = adj + np.eye(adj.shape[0])
row_sum = np.array(np.sum(adj,axis=1))
degree_matrix = np.matrix(np.diag(row_sum))

D = fractional_matrix_power(degree_matrix, -0.5)
adj = D.dot(adj).dot(D)'''

adj = torch.FloatTensor(adj_norm)
print(adj)

adj1 = torch.FloatTensor(adj_norm)
print('if adj_hat == adj_norm: ?', adj==adj1)
#adj = torch.FloatTensor(adj_norm)

#print('idx:',idx_train, idx_val, idx_test)# 0-139; 140-640; 1708-2707

"""print(dataset.graph)
n = dataset.graph['num_nodes']
adjs, _ = remove_self_loops(dataset.graph['edge_index'])
adjs, _ = add_self_loops(adjs, num_nodes=n)
dataset.graph['adj'] = adjs
print('adjs:',adjs)
# Load data
adj, features, labels, idx_train, idx_val, idx_test = load_data()
print(adj)"""

# Model and optimizer
model = GCN(nfeat=features.shape[1],
            nhid=args.hidden,
            nclass=labels.max().item() + 1,
            dropout=args.dropout)
optimizer = optim.Adam(model.parameters(),
                       lr=args.lr, weight_decay=args.weight_decay)

if args.cuda:
    model.cuda()
    features = features.cuda()
    adj = adj.cuda()
    labels = labels.cuda()
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()


def train(epoch):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    output = model(features, adj)
    loss_train = F.nll_loss(output[idx_train], labels[idx_train])
    acc_train = accuracy(output[idx_train], labels[idx_train])
    loss_train.backward()
    optimizer.step()

    if not args.fastmode:
        # Evaluate validation set performance separately,
        # deactivates dropout during validation run.
        model.eval()
        output = model(features, adj)

    loss_val = F.nll_loss(output[idx_val], labels[idx_val])
    acc_val = accuracy(output[idx_val], labels[idx_val])
    print('Epoch: {:04d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'acc_train: {:.4f}'.format(acc_train.item()),
          'loss_val: {:.4f}'.format(loss_val.item()),
          'acc_val: {:.4f}'.format(acc_val.item()),
          'time: {:.4f}s'.format(time.time() - t))


def test():
    model.eval()
    output = model(features, adj)
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])

    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()))
    line_str = 'w_d: %.4f, tau: %.4f,epoch: %d,hidden_dim: %d,drop: %f ,lr: %f,test loss: %.5f,test acc: %.5f\n'
    with open('./result.txt', 'a+') as f:
        f.write(line_str % (args.weight_decay,args.tau, args.epochs,args.hidden, args.dropout, args.lr, loss_test, acc_test))


# Train model
t_total = time.time()
for epoch in range(args.epochs):
    train(epoch)
print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

# Testing
test()
