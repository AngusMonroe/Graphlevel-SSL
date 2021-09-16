from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import GINConv, GCNConv, global_add_pool
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, GlobalAttention, Set2Set
from argument import parse_args
from utils import setup_seed
args, unknown = parse_args()
setup_seed(args.seed)


class GINEncoder(torch.nn.Module):
    def __init__(self, num_features, dim, num_gc_layers, device, dataset=None):
        super(GINEncoder, self).__init__()


        self.num_gc_layers = num_gc_layers

        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        self.device = device
        self.dataset = dataset


        for i in range(num_gc_layers):

            if i:
                nn = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
            else:
                nn = Sequential(Linear(num_features, dim), ReLU(), Linear(dim, dim))
            conv = GINConv(nn)
            bn = torch.nn.BatchNorm1d(dim)

            self.convs.append(conv)
            self.bns.append(bn)


    def forward(self, x, edge_index, batch):
        if x is None:
            x = torch.ones((batch.shape[0], 1)).to(self.device)

        xs = []

        for i in range(self.num_gc_layers):
            x = F.relu(self.convs[i](x, edge_index))
            x = self.bns[i](x)
            xs.append(x)

        xpool = [global_add_pool(x, batch) for x in xs]
        x = torch.cat(xpool, 1)
        

        return x, torch.cat(xs, 1) # graph / node

    def get_embeddings(self, loader):

        ret = []
        y = []
        with torch.no_grad():
            for data in loader:
                if len(data) == 2:
                    data = data[0]
                data.to(self.device)
                x, edge_index, batch = data.x, data.edge_index, data.batch
                
                if x is None:
                    x = torch.ones((batch.shape[0],1)).to(self.device)
                x, _ = self.forward(x, edge_index, batch)
                ret.append(x.cpu().numpy())
                y.append(data.y.cpu().numpy())
        ret = np.concatenate(ret, 0)
        y = np.concatenate(y, 0)
        return ret, y


class GCNEncoder(torch.nn.Module):
    def __init__(self, num_features, dim, num_gc_layers, device):
        super(GCNEncoder, self).__init__()

        self.num_gc_layers = num_gc_layers
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        self.device = device
        self.act = nn.PReLU()

        for i in range(num_gc_layers):

            if i:
                conv = GCNConv(dim, dim)
            else:
                conv = GCNConv(num_features, dim)
                bn = torch.nn.BatchNorm1d(dim)

            self.bns.append(bn)
            self.convs.append(conv)
            

    def forward(self, x, edge_index, batch):
        if x is None:
            x = torch.ones((batch.shape[0], 1)).to(self.device)

        xs = []
        for i in range(self.num_gc_layers):
            x = self.convs[i](x, edge_index)
            x = self.act(x)
            
            if i != (self.num_gc_layers-1):
                x = self.bns[i](x)
            
            xs.append(x)

            # if i == 2:
                # feature_map = x2

        xpool = [global_add_pool(x, batch) for x in xs]
        xpooled = torch.cat(xpool, 1)
        
        return xpooled, x


    def get_embeddings(self, loader):

        ret = []
        y = []
        with torch.no_grad():
            for data in loader:
                data = data[0]
                data.to(self.device)
                x, edge_index, batch = data.x, data.edge_index, data.batch
                if x is None:
                    x = torch.ones((batch.shape[0],1)).to(self.device)
                x, _ = self.forward(x, edge_index, batch)
                ret.append(x.cpu().numpy())
                y.append(data.y.cpu().numpy())
        ret = np.concatenate(ret, 0)
        y = np.concatenate(y, 0)
        return ret, y


