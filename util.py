import networkx as nx
import numpy as np
import os
import sys
import pandas as pd
import pickle as pk
import scipy.sparse as sp
import random
import torch
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

class S2VGraph(object):
    def __init__(self, g, label, node_tags=None, node_features=None, edge_mat=None, edge_labels=None, edge_features=None):
        '''
            g: a networkx graph
            label: an integer graph label
            node_tags: a list of integer node tags
            node_features: a torch float tensor, one-hot representation of the tag that is used as input to neural nets
            edge_mat: a torch long tensor, contain edge list, will be used to create torch sparse tensor
            neighbors: list of neighbors (without self-loop)
        '''
        self.label = label
        self.g = g
        self.node_tags = node_tags
        self.neighbors = []
        self.node_features = node_features
        self.edge_mat = edge_mat
        self.edge_labels = edge_labels
        self.edge_features = edge_features

        self.max_neighbor = 0
        
    def to(self, device):
        self.node_features = self.node_features.to(device)
        self.edge_mat = self.edge_mat.to(device)
        self.node_tags = torch.LongTensor(self.node_tags).to(device)
        return self


def load_data(dataset, degree_as_tag=False):
    '''
        dataset: name of dataset
    '''

    print('loading data')
    g_list = []
    label_dict = {}
    feat_dict = {}

    with open('./datasets/%s/%s.txt'%(dataset,dataset), 'r') as f:
        n_g = int(f.readline().strip())
        for i in range(n_g):
            row = f.readline().strip().split()
            n, l = [int(w) for w in row]
            if not l in label_dict:
                mapped = len(label_dict)
                label_dict[l] = mapped
            g = nx.Graph()
            node_tags = []
            node_features = []
            n_edges = 0
            for j in range(n):
                g.add_node(j)
                row = f.readline().strip().split()
                tmp = int(row[1]) + 2
                if tmp == len(row):
                    # no node attributes
                    row = [int(w) for w in row]
                    attr = None
                else:
                    row, attr = [int(w) for w in row[:tmp]], np.array([float(w) for w in row[tmp:]])
                if not row[0] in feat_dict:
                    mapped = len(feat_dict)
                    feat_dict[row[0]] = mapped
                node_tags.append(feat_dict[row[0]])

                if tmp > len(row):
                    node_features.append(attr)

                n_edges += row[1]
                for k in range(2, len(row)):
                    g.add_edge(j, row[k])

            if node_features != []:
                node_features = np.stack(node_features)
                node_feature_flag = True
            else:
                node_features = None
                node_feature_flag = False

            assert len(g) == n

            g_list.append(S2VGraph(g, l, node_tags))

    #add labels and edge_mat       
    for g in g_list:
        g.neighbors = [[] for i in range(len(g.g))]
        for i, j in g.g.edges():
            g.neighbors[i].append(j)
            g.neighbors[j].append(i)
        degree_list = []
        for i in range(len(g.g)):
            g.neighbors[i] = g.neighbors[i]
            degree_list.append(len(g.neighbors[i]))
        g.max_neighbor = max(degree_list)

        g.label = label_dict[g.label]

        edges = [list(pair) for pair in g.g.edges()]
        edges.extend([[i, j] for j, i in edges])

        deg_list = list(dict(g.g.degree(range(len(g.g)))).values())
        g.edge_mat = torch.sparse.FloatTensor(torch.LongTensor(edges).transpose(0,1),  \
                                              torch.ones(len(edges)),torch.Size((len(g.g),len(g.g))))

    if degree_as_tag:
        for g in g_list:
            g.node_tags = list(dict(g.g.degree()).values())

    #Extracting unique tag labels   
    tagset = set([])
    for g in g_list:
        tagset = tagset.union(set(g.node_tags))

    tagset = list(tagset)
    tag2index = {tagset[i]:i for i in range(len(tagset))}

    for g in g_list:
        g.node_features = torch.zeros(len(g.node_tags), len(tagset))
        g.node_features[range(len(g.node_tags)), [tag2index[tag] for tag in g.node_tags]] = 1


    print('# classes: %d' % len(label_dict))
    print('# maximum node tag: %d' % len(tagset))

    print("# data: %d" % len(g_list))

    return g_list, len(label_dict)

def separate_data(graph_list, seed, fold_idx):
    assert 0 <= fold_idx and fold_idx < 10, "fold_idx must be from 0 to 9."
    skf = StratifiedKFold(n_splits=10, shuffle = True, random_state = seed)

    labels = [graph.label for graph in graph_list]
    idx_list = []
    for idx in skf.split(np.zeros(len(labels)), labels):
        idx_list.append(idx)
    train_idx, test_idx = idx_list[fold_idx]

    train_graph_list = [graph_list[i] for i in train_idx]
    test_graph_list = [graph_list[i] for i in test_idx]

    return train_graph_list, test_graph_list

def load_train_test(dataset, fold_idx, val=False):
    graphs, num_classes = load_data(dataset)
    with open('./datasets/%s/10fold_idx/train_idx-%d.txt'%(dataset, fold_idx), 'r') as f:
        tr_idx = [int(s.strip()) for s in f.readlines()]      
        if val:
            sss = StratifiedShuffleSplit(n_splits=1, test_size=0.11, random_state=0)
            train_idx, val_idx = next(sss.split(np.zeros(len(tr_idx)), [graphs[i].label for i in tr_idx]))
            train_graphs = [graphs[tr_idx[i]] for i in train_idx]
            val_graphs = [graphs[tr_idx[i]] for i in val_idx]
        else:
            train_graphs=[graphs[i] for i in tr_idx]
            val_graphs = train_graphs
    with open('./datasets/%s/10fold_idx/test_idx-%d.txt'%(dataset, fold_idx), 'r') as f:
        te_idx = [int(s.strip()) for s in f.readlines()]
        test_graphs = [graphs[i] for i in te_idx]
    return train_graphs, test_graphs, val_graphs, num_classes

def sps_block_diag(tensors):
    idx_list = []
    elem_list = []
    start_idx = torch.zeros(2).long().to(tensors[0].device)
    for i, t in enumerate(tensors):        
        idx_list.append(t._indices() + start_idx.unsqueeze(1).expand_as(t._indices()))
        elem_list.append(t._values())
        start_idx += torch.tensor(t.shape).to(t.device)
    block_idx = torch.cat(idx_list, 1)
    block_elem = torch.cat(elem_list)
    block = torch.sparse.FloatTensor(block_idx, block_elem, torch.Size(start_idx))
    return block

class Graph(object):
    def __init__(self, gindex=0, edge_mat=None, label=None, node_tags=None, unique_node=None, node_features=None,  \
                 edge_labels=None, edge_features=None):
        '''
            g: a networkx graph
            label: an integer graph label
            node_tags: a list of integer node tags
            node_features: a torch float tensor, one-hot representation of the tag that is used as input to neural nets
            edge_mat: a torch long tensor, contain edge list, will be used to create torch sparse tensor
            neighbors: list of neighbors (without self-loop)
        '''
        self.label = label
        self.unique_node, self.node_tags = np.unique(node_tags, return_inverse=True)
        self.node_features = node_features
        self.edge_mat = edge_mat.tocoo()
        self.edge_labels = edge_labels
        self.edge_features = edge_features
        self.gind = gindex

        self.max_neighbor = 0 if self.edge_mat is None else (self.edge_mat!=0).sum(0).max()
        
    def to(self, device):
        self.node_features = torch.Tensor(self.node_features).to(device)
        indices = torch.LongTensor(np.vstack((self.edge_mat.row, self.edge_mat.col)))
        values = torch.FloatTensor(self.edge_mat.data)
        self.edge_mat = torch.sparse.FloatTensor(indices, values, torch.Size(self.edge_mat.shape)).to(device)
        self.node_tags = torch.LongTensor(self.node_tags).to(device)
        return self
    
def load_data_general(dataset, tag_as_fea=True):
    print('loading data')

    egs = pd.read_csv('./datasets/%s/%s_A.txt'%(dataset, dataset),header=None)
    gind = pd.read_csv('./datasets/%s/%s_graph_indicator.txt'%(dataset, dataset),header=None)
    glabel = pd.read_csv('./datasets/%s/%s_graph_labels.txt'%(dataset, dataset),header=None)[0]
    if os.path.isfile('./datasets/%s/%s_node_labels.txt'%(dataset, dataset)):    
        ndlabel = pd.read_csv('./datasets/%s/%s_node_labels.txt'%(dataset, dataset),header=None)[0]
    else:
        print('%s, no node label file, all nodes are seen as the same label'%(dataset))
        ndlabel = np.ones(len(gind))
    ndfea = pd.read_csv('./datasets/%s/%s_node_attributes.txt'%(dataset, dataset),header=None).values
#     eglabel = pd.read_csv('./datasets/%s/%s_edge_labels.txt'%(dataset, dataset),header=None)[0]
#     egfea = pd.read_csv('./datasets/%s/%s_edge_attributes.txt'%(dataset, dataset),header=None)
    
    grps = gind.groupby(0).indices
    adj = sp.coo_matrix((np.ones(len(egs)),egs.values.transpose()-1)).tocsr()
    unique_label, glbl = np.unique(glabel, return_inverse=True)
    unique_node, node_idx = np.unique(ndlabel, return_inverse=True)
    emb = np.eye(len(unique_node))
    if tag_as_fea and len(unique_node)>1:
        tag_fea = emb[node_idx]
        ndfea = np.concatenate([tag_fea,ndfea],axis=1)
    
    graphs=[]
    for gindex, idx in grps.items():           
        edge_mat = adj[idx,:][:,idx]
        label = glbl[gindex-1]
        node_tags = node_idx[idx]
        node_features = ndfea[idx]
        edge_labels = None
        edge_features = None
        g = Graph(gindex, edge_mat, label, node_tags, unique_node, node_features, edge_labels, edge_features)
        graphs.append(g)
        
    return graphs, len(unique_label)
        
def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)

def load_graph(dataset_str):
    """
    Loads input graph data for node classification
    ind.dataset_str.x => the feature vectors of the training instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.tx => the feature vectors of the test instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.allx => the feature vectors of both labeled and unlabeled training instances
        (a superset of ind.dataset_str.x) as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.y => the one-hot labels of the labeled training instances as numpy.ndarray object;
    ind.dataset_str.ty => the one-hot labels of the test instances as numpy.ndarray object;
    ind.dataset_str.ally => the labels for instances in ind.dataset_str.allx as numpy.ndarray object;
    ind.dataset_str.graph => a dict in the format {index: [index_of_neighbor_nodes]} as collections.defaultdict
        object;
    ind.dataset_str.test.index => the indices of test instances in graph, for the inductive setting as list object.
    All objects above must be saved using python pickle module.
    :param dataset_str: Dataset name
    :return: All data input files loaded (as well the training/test data).
    """
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("datasets/{}/ind.{}.{}".format(dataset_str, dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pk.load(f, encoding='latin1'))
            else:
                objects.append(pk.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("datasets/{}/ind.{}.test.index".format(dataset_str, dataset_str))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range-min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
    adj = normalize(adj + sp.eye(adj.shape[0]))
    features = normalize(features)

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]
    labels = np.where(labels)[1]

    idx_test = test_idx_range.tolist()
    idx_train = range(len(y))
    idx_val = range(len(y), len(y)+500)
    
    graph = Graph(edge_mat=adj, node_features=features.todense(), node_tags=labels)

    return graph, idx_train, idx_val, idx_test
    
def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)
    
def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1),dtype=float)
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx        
        
def save_obj(obj, name ):
    with open(name, 'wb') as f:
        pk.dump(obj, f, pk.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open( name, 'rb') as f:
        return pk.load(f)

