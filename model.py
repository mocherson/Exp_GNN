import math
import numpy as np
import scipy.sparse as sp

import torch

from torch import nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from util import sps_block_diag


def phi(x, m):
    y = [torch.ones(x.shape[0],1).to(x.device)]
    for i in range(1, m+1):
        
        y.append(x.pow(i))
    return torch.cat(y,dim=1)

class PHI(nn.Module):
    def __init__(self, m):
        super(PHI, self).__init__()
        self.m = m
        # self.p = nn.Parameter(torch.rand(1))
        
    def forward(self, x):
        x = x.abs().pow(1/self.m)*x.sign()
        return phi(x,self.m)
    
def vdphi(x, m):
    y = [torch.ones(x.shape[0],1).to(x.device)]
    y.append(x[:,1:])
    for i in range(1, m+1):        
        y.append(x[:,[0]].pow(i))
        if i<m:
            y.append(x[:,[0]].pow(i)*x[:,1:])
            
    return torch.cat(y,dim=1)

class vdPHI(nn.Module):
    def __init__(self, m):
        super(vdPHI, self).__init__()
        self.m = m
        # self.p = nn.Parameter(torch.rand(1))
        
    def forward(self, x):
        x = torch.cat([x[:,[0]].abs().pow(1/self.m)*x[:,[0]].sign(),x[:,1:]],dim=1)
        return vdphi(x,self.m)

class MLP(nn.Module):
    def __init__(self, in_dim, out_dims, batch_norm=True, dropout=0):
        '''
            in_dim: dimensionality of input features
            out_dim: a list of intgers indicating the dimensionality of hidden and output features
        '''
    
        super(MLP, self).__init__()
        layers = [nn.BatchNorm1d(in_dim), nn.Linear(in_dim, out_dims[0])] if batch_norm else [nn.Linear(in_dim, out_dims[0])] 
        for i in range(len(out_dims)):
            if i+1<len(out_dims):
                layers += [nn.BatchNorm1d(out_dims[i]),nn.ReLU(inplace=True),nn.Dropout(p=dropout),nn.Linear(out_dims[i],out_dims[i+1])]
                
        self.encoder = nn.Sequential(*layers)

    def forward(self, x):
        return self.encoder(x)

class DGNNLayer(nn.Module):
    def __init__(self, in_features, phi_features, out_features, phi=lambda x:x, batch_norm=True, agg='cat'):
        super(DGNNLayer, self).__init__()
        self.in_features = in_features
        self.phi = phi
        self.agg = agg
        if self.agg=='sum':
            self.encoder = MLP(in_features, out_features, batch_norm)  
        elif self.agg=='cat':
            self.encoder = MLP(in_features+phi_features, out_features, batch_norm) 
                       
        
    def forward(self, input, adj):
        assert self.in_features == input.shape[1]
        x = self.phi(input)
        output = torch.spmm(adj, x)

        if self.agg=='sum':
            x = (input + output)  
        elif self.agg=='cat':
            x = torch.cat([input, output], dim=1)
        x = self.encoder(x)
        return x

    def __repr__(self):
        return self.__class__.__name__ +'(in_features={}, phi={}, encoder={})'.format(
            self.in_features, str(self.phi), str(self.encoder) )
            
class AttDGraphNN(nn.Module):    
    def __init__(self, in_features, phi_features, out_features,  n_class,  dropout=0, \
                 phis=lambda x:x, batch_norm=True, agg='cat'):
        super(AttDGraphNN, self).__init__()
        assert len(phi_features)==len(out_features) , "layers mismatch"
            
        if not isinstance(phis,(tuple, list)):
            phis = [phis]*len(phi_features)
            
        if not isinstance(agg,(tuple, list)):
            agg = [agg]*len(phi_features)
            
        self.encoder = nn.ModuleList([DGNNLayer(in_features, phi_features[0],out_features[0],phis[0],batch_norm, agg[0])])
        for i in range(len(phi_features)-1):
            self.encoder.append(DGNNLayer(out_features[i][-1], phi_features[i+1],out_features[i+1],phis[i+1],batch_norm, agg[i+1]))
            
        self.classifier = MLP(out_features[-1][-1],(64, n_class, ), batch_norm=True, dropout=dropout)
        self.dropout=dropout

    def forward(self, graphs):
        x = torch.cat([graph.node_features for graph in graphs], 0)        
        adj = sps_block_diag([graph.edge_mat for graph in graphs])
        n_nodes = [len(graph.node_tags) for graph in graphs]
        for m in self.encoder:
            x = m(x, adj)
            x[x==0]+=1e-8  # avoid infinite gradient
            
        graph_embedding = torch.stack([t.sum(0) for t in x.split(n_nodes)])  
        support = self.classifier(graph_embedding)
        return graph_embedding, support

    
    
class ExpGraphNN_ND(nn.Module):    
    def __init__(self, in_features, phi_features, out_features,  n_class,  dropout=0, \
                 phis=lambda x:x, batch_norm=True, agg='cat'):
        super(ExpGraphNN_ND, self).__init__()
        assert len(phi_features)==len(out_features) , "layers mismatch"
            
        if not isinstance(phis,(tuple, list)):
            phis = [phis]*len(phi_features)
            
        if not isinstance(agg,(tuple, list)):
            agg = [agg]*len(phi_features)
            
        self.encoder = nn.ModuleList([DGNNLayer(in_features, phi_features[0],out_features[0],phis[0],batch_norm, agg[0]) ])
        for i in range(len(phi_features)-1):
            self.encoder.append(DGNNLayer(out_features[i][-1], phi_features[i+1],out_features[i+1],phis[i+1],batch_norm, agg[i+1]))
            
        # self.classifier = MLP(out_features[-1][-1],( n_class, ), batch_norm=False, dropout=dropout)
        self.dropout=dropout 

    def forward(self, graph):
        x = graph.node_features     
        adj = graph.edge_mat

        for i, m in enumerate(self.encoder):            
            x = m(x, adj)  
            if i+1<len(self.encoder): 
                x = F.relu(x)
                x = F.dropout(x,p=self.dropout, training=self.training)

        # support = self.classifier(x)
        support=x
        return x, support
                        
class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.trans = MLP(in_features, (out_features,  ), batch_norm=False)
        # self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        # if bias:
        #     self.bias = Parameter(torch.FloatTensor(out_features))
        # else:
        #     self.register_parameter('bias', None)
        # self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        # support = torch.mm(input, self.weight)     
        output = self.trans(input)
        output = torch.spmm(adj, output)
        # output = torch.cat([input,output],dim=1)
        
        # if self.bias is not None:
        #     return output + self.bias
        # else:
        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'        
        
class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        # self.classifier = nn.Linear(nhid*2+nfeat, nclass)
        self.dropout = dropout

    def forward(self, graph):
        x = graph.node_features     
        adj = graph.edge_mat
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        # x = self.classifier(x)
        return x, x
    
    
    
    