import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import os

from tqdm import tqdm

from util import *
from model import *

parser = argparse.ArgumentParser(description='PyTorch graph neural net for whole-graph classification')
parser.add_argument('--dataset', type=str, default="MUTAG",
                    help='name of dataset (default: MUTAG)')
parser.add_argument('--device', type=int, default=0,
                    help='which gpu to use if any (default: 0)')
parser.add_argument('--batch_size', type=int, default=32,
                    help='input batch size for training (default: 32)')
parser.add_argument('--epochs', type=int, default=300,
                    help='number of epochs to train (default: 300)')
parser.add_argument('--lr', type=float, default=0.01,
                    help='learning rate (default: 0.01)')
parser.add_argument('--seed', type=int, default=0,
                    help='random seed for splitting the dataset into 10 (default: 0)')
parser.add_argument('--fold_idx', type=int, default=0,
                    help='the index of fold in 10-fold validation. Should be less then 10.')
parser.add_argument('--hidden_dim', type=int, default=64,
                    help='number of hidden units (default: 64)')
parser.add_argument('--agg', type=str, default="cat", choices=["cat", "sum"],
                    help='aggregate input and its neighbors, can be extended to other method like mean, max etc.')
parser.add_argument('--attribute', action="store_true",
                    help='Whether it is for attributed graph.')
parser.add_argument('--phi', type=str, default="power", choices=["power", "identical", "MLP","vdmd"],
                    help='transformation before aggregation')
parser.add_argument('--first_phi', action="store_true",
                    help='Whether using phi for first layer. False indicates no transform')
parser.add_argument('--dropout', type=float, default=0,
                        help='final layer dropout (default: 0)')
parser.add_argument('--weight_decay', type=float, default=0.0,
                        help='weight decay in the optimizer (default: 0)')
parser.add_argument('--filename', type = str, default = "",
                    help='save result to file')
args = parser.parse_args()

device = args.device
dataset = args.dataset
fold_idx = args.fold_idx+1
agg = args.agg
hid_dim = args.hidden_dim
dropout = args.dropout
isattr = args.attribute
weight_decay = args.weight_decay
firstphi = args.first_phi

filename = args.filename if not args.filename == "" else "./results/{}/{}_{}_hid{}_wd{}_{}.csv"  \
    .format(dataset,args.phi,fold_idx,hid_dim, weight_decay, agg )
if os.path.isfile(filename):
    print('%s, file exists.'%(filename))
    os._exit(0)

torch.manual_seed(0)
np.random.seed(0)    
device = torch.device("cuda:" + str(device)) if torch.cuda.is_available() else torch.device("cpu")
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)
    
criterion = nn.CrossEntropyLoss()
bgraph=[]    
def train(args, model, device, train_graphs, optimizer, epoch):
    model.train()

    idxs = np.random.permutation(len(train_graphs))
    
    i=0
    loss_accum = 0
    while i<len(idxs):
        selected_idx = idxs[i:i+args.batch_size]
        i = i+args.batch_size

        batch_graph = [train_graphs[idx] for idx in selected_idx]
        _, output = model(batch_graph)

        labels = torch.LongTensor([graph.label for graph in batch_graph]).to(device)

        #compute loss
        loss = criterion(output, labels)

        #backprop
        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        loss = loss.detach().cpu().numpy()
        loss_accum += loss


    average_loss = loss_accum*args.batch_size/len(idxs)
    print("epoch:%d, loss training: %f" % (epoch, average_loss))
    
    return average_loss


def test(args, model, device, train_graphs, test_graphs, epoch):
    model.eval()

    with torch.no_grad():
        acc_train=0
        if sum([len(g.node_tags) for g in train_graphs])<120000:
            emb_tr, output = model(train_graphs)
            pred = output.max(1, keepdim=True)[1]
            labels = torch.LongTensor([graph.label for graph in train_graphs]).to(device)
            correct = pred.eq(labels.view_as(pred)).sum().cpu().item()
            acc_train = correct / float(len(train_graphs))

        emb_te, output = model(test_graphs)
        pred = output.max(1, keepdim=True)[1]
        labels = torch.LongTensor([graph.label for graph in test_graphs]).to(device)
        correct = pred.eq(labels.view_as(pred)).sum().cpu().item()
        acc_test = correct / float(len(test_graphs))
        
        if epoch%10==0 and args.dataset=='SYNTHETICnew':
            save_obj(emb_tr.cpu().numpy(), './results/{}/embeddings/tr_{}_hid{}_ep{}.pkl'.format(dataset, args.phi,hid_dim,epoch))
            save_obj(emb_te.cpu().numpy(), './results/{}/embeddings/te_{}_hid{}_ep{}.pkl'.format(dataset, args.phi,hid_dim,epoch))

    print("accuracy train: %f,  test: %f" % (acc_train,  acc_test))

    return acc_train, acc_test

if isattr:
    graphs, num_classes = load_data_general(dataset)
    train_graphs, test_graphs = separate_data(graphs, args.seed, args.fold_idx)
    train_graphs = [g.to(device) for g in train_graphs]
    test_graphs = [g.to(device) for g in test_graphs]
else:
    train_graphs, test_graphs, val_graphs, num_classes = load_train_test(dataset,  fold_idx)
    # graphs, num_classes = load_data(args.dataset)
    # train_graphs, test_graphs = separate_data(graphs, args.seed, args.fold_idx)
    train_graphs = [g.to(device) for g in train_graphs]
    test_graphs = [g.to(device) for g in test_graphs]

m = max([graph.max_neighbor for graph in train_graphs])
in_dim = train_graphs[0].node_features.shape[1]
out_features = ((hid_dim, hid_dim ), (hid_dim, hid_dim ), (hid_dim, hid_dim ), (hid_dim, hid_dim), (hid_dim, hid_dim))

if args.phi=="power":
    if firstphi:
        phi_features = (in_dim*m+1, hid_dim*m+1,hid_dim*m+1,hid_dim*m+1,hid_dim*m+1)
        ph = [PHI(m) for i in range(5)]
    else:
        phi_features = (in_dim, hid_dim*m+1,hid_dim*m+1,hid_dim*m+1,hid_dim*m+1)
        ph = [lambda x:x]+[PHI(m) for i in range(4)]
elif args.phi=="identical":
    phi_features = (in_dim, hid_dim,hid_dim,hid_dim,hid_dim)
    ph = [lambda x:x]*5
elif args.phi=="MLP":
    phi_features = (in_dim, hid_dim,hid_dim,hid_dim,hid_dim)
    if firstphi:
        ph = [MLP(in_dim,(hid_dim,in_dim), batch_norm=True)]+[MLP(hid_dim,(hid_dim,hid_dim), batch_norm=True) for i in range(4)]
    else:
        ph = [lambda x:x]+[MLP(hid_dim,(hid_dim,hid_dim), batch_norm=True) for i in range(4)]
elif args.phi == "vdmd":
    if firstphi:
        phi_features = (in_dim*m+1, hid_dim*m+1,hid_dim*m+1,hid_dim*m+1,hid_dim*m+1)
        ph = [vdPHI(m) for i in range(5)]
    else:
        phi_features = (in_dim, hid_dim*m+1,hid_dim*m+1,hid_dim*m+1,hid_dim*m+1)
        ph = [lambda x:x]+[vdPHI(m) for i in range(4)]

model = AttDGraphNN(in_dim,phi_features,out_features, n_class=num_classes, dropout=dropout, phis=ph,
                      batch_norm=True, agg=agg).to(device)
    
optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=weight_decay)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

acc_tr=[]
acc_te=[]
loss_tr=[]
bestacc=0
bestloss=np.inf
best_epoc = 0
for epoch in range(1, args.epochs + 1):
    scheduler.step()

    avg_loss = train(args, model, device, train_graphs, optimizer, epoch)
    acc_train,  acc_test = test(args, model, device, train_graphs, test_graphs,  epoch)
   
    acc_tr.append(acc_train)
    acc_te.append(acc_test)
    loss_tr.append(avg_loss)
    
#     if acc_train>bestacc or avg_loss<bestloss:
#         bestacc=max(acc_train, bestacc)
#         bestloss=min(avg_loss, bestloss)
#         best_epoc=epoch
        
#     if epoch-best_epoc>=50:
#         break

res = pd.DataFrame({"acc_tr":acc_tr,"acc_te":acc_te,"loss_tr":loss_tr})    

res.to_csv(filename)

    


