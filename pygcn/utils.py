import numpy as np
import scipy.sparse as sp
import torch
from scipy.sparse import coo_matrix
from google_drive_downloader import GoogleDriveDownloader as gdd
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid, Amazon, Coauthor
from torch_geometric.utils import degree


def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot


def load_data(path="../data/cora/", dataset="cora"):
    """Load citation network dataset (cora only for now)"""
    print('Loading {} dataset...'.format(dataset))

    idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset),
                                        dtype=np.dtype(str))
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    labels = encode_onehot(idx_features_labels[:, -1])

    # build graph
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset),
                                    dtype=np.int32)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=np.int32).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    features = normalize(features)
    adj = normalize(adj + sp.eye(adj.shape[0]))

    idx_train = range(140)
    idx_val = range(200, 500)
    idx_test = range(500, 1500)

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(np.where(labels)[1])
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_val, idx_test


def load_citation_data(data):
    row, col = data.graph['edge_index']
    num_of_nodes = data.graph['node_feat'].shape[0]
    adj = torch.zeros(num_of_nodes, num_of_nodes)
    for i in np.arange(row.shape[0]):
        adj[row[i]][col[i]] = 1.0
    # adj = edge_index_to_sparse_matrix(data.graph['edge_index'],num_of_nodes)
    adj_norm = generate_adj(adj)

    features = data.graph['node_feat']
    labels = data.label
    idx_train = data.train_idx
    idx_val = data.valid_idx
    idx_test = data.test_idx
    #idx_test = torch.cat(idx_train,idx_test,dim=1)

    '''idx_train = range(140)
    idx_val = range(200, 500)
    idx_test = range(500, 1500)'''

    adj = adj.detach().numpy()
    # adj = sparse_matrix_to_tensor(adj)
    features = features.detach().numpy()
    labels = labels.detach().numpy()
    features = torch.FloatTensor(features)
    labels = torch.LongTensor(labels)
    # adj = torch.FloatTensor(adj)

    return adj_norm, adj, features, labels, idx_train, idx_val, idx_test


def sparse_matrix_to_tensor(sparse_matrix):
    # 使用 torch.sparse_coo_tensor 将稀疏矩阵转换为 PyTorch Tensor
    tensor = torch.sparse_coo_tensor(torch.tensor([sparse_matrix.row, sparse_matrix.col]),
                                     torch.tensor(sparse_matrix.data),
                                     sparse_matrix.shape)

    return tensor


def generate_adj(adj):
    adj = adj.detach().numpy()
    adj = normalize(adj + sp.eye(adj.shape[0]))
    return adj


class NCDataset(object):
    def __init__(self, name):
        """
        based off of ogb NodePropPredDataset
        https://github.com/snap-stanford/ogb/blob/master/ogb/nodeproppred/dataset.py
        Gives torch tensors instead of numpy arrays
            - name (str): name of the dataset
            - root (str): root directory to store the dataset folder
            - meta_dict: dictionary that stores all the meta-information about data. Default is None,
                    but when something is passed, it uses its information. Useful for debugging for external contributers.

        Usage after construction:

        split_idx = dataset.get_idx_split()
        train_idx, valid_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]
        graph, label = dataset[0]

        Where the graph is a dictionary of the following form:
        dataset.graph = {'edge_index': edge_index,
                         'edge_feat': None,
                         'node_feat': node_feat,
                         'num_nodes': num_nodes}
        For additional documentation, see OGB Library-Agnostic Loader https://ogb.stanford.edu/docs/nodeprop/

        """

        self.name = name  # original name, e.g., ogbn-proteins
        self.graph = {}
        self.label = None

    def get_idx_split(self, split_type='random', train_prop=.5, valid_prop=.25, label_num_per_class=20):
        """
        split_type: 'random' for random splitting, 'class' for splitting with equal node num per class
        train_prop: The proportion of dataset for train split. Between 0 and 1.
        valid_prop: The proportion of dataset for validation split. Between 0 and 1.
        label_num_per_class: num of nodes per class
        """

        if split_type == 'random':
            ignore_negative = False if self.name == 'ogbn-proteins' else True
            train_idx, valid_idx, test_idx = rand_train_test_idx(
                self.label, train_prop=train_prop, valid_prop=valid_prop, ignore_negative=ignore_negative)
            split_idx = {'train': train_idx,
                         'valid': valid_idx,
                         'test': test_idx}
        elif split_type == 'class':
            train_idx, valid_idx, test_idx = class_rand_splits(self.label, label_num_per_class=label_num_per_class)
            split_idx = {'train': train_idx,
                         'valid': valid_idx,
                         'test': test_idx}
        return split_idx

    def __getitem__(self, idx):
        assert idx == 0, 'This dataset has only one graph'
        return self.graph, self.label

    def __len__(self):
        return 1

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, len(self))


def load_dataset(data_dir, dataname):
    dataset = load_planetoid_dataset(data_dir, dataname)
    return dataset


def load_planetoid_dataset(data_dir, name):
    transform = T.NormalizeFeatures()
    torch_dataset = Planetoid(root=f'{data_dir}Planetoid',
                              name=name, transform=transform)
    # torch_dataset = Planetoid(root=f'{DATAPATH}Planetoid', name=name)
    data = torch_dataset[0]
    edge_index = data.edge_index
    node_feat = data.x
    label = data.y
    num_nodes = data.num_nodes
    dataset = NCDataset(name)
    dataset.train_idx = torch.where(data.train_mask)[0]
    dataset.valid_idx = torch.where(data.val_mask)[0]
    dataset.test_idx = torch.where(data.test_mask)[0]

    dataset.graph = {'edge_index': edge_index,
                     'node_feat': node_feat,
                     'edge_feat': None,
                     'num_nodes': num_nodes}
    dataset.label = label
    return dataset


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse_coo_tensor(indices, values, shape)


def edge_index_to_sparse_matrix(edge_index, num_nodes):
    # 创建一个稀疏矩阵的坐标表示
    row, col = edge_index
    data = np.ones_like(row)  # 假设每个边的权重都是1

    # 使用 coo_matrix 创建稀疏矩阵
    sparse_matrix = coo_matrix((data, (row, col)), shape=(num_nodes, num_nodes))

    return sparse_matrix
