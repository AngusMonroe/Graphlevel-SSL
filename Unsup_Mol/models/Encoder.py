from torch.nn import Sequential, Linear, ReLU, PReLU
from torch_geometric.nn import GINConv, GCNConv, global_add_pool
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree, softmax
import rdkit
from ogb.graphproppred.mol_encoder import AtomEncoder
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, GlobalAttention, Set2Set
from torch_sparse import SparseTensor

num_atom_type = 120 #including the extra mask tokens
num_chirality_tag = 3

num_bond_type = 6 #including aromatic and self-loop edge, and extra masked tokens
num_bond_direction = 3


#class GINEncoder(torch.nn.Module):
#    def __init__(self, num_features, dim, num_gc_layers, device, dataset=None):
#        super(GINEncoder, self).__init__()
#
#
#        self.num_gc_layers = num_gc_layers
#
#        # self.nns = []
#        self.convs = torch.nn.ModuleList()
#        self.bns = torch.nn.ModuleList()
#        self.device = device
#        self.dataset = dataset
#
#
#        for i in range(num_gc_layers):
#
#            if i:
#                nn = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
#            else:
#                nn = Sequential(Linear(num_features, dim), ReLU(), Linear(dim, dim))
#            conv = GINConv(nn)
#            bn = torch.nn.BatchNorm1d(dim)
#
#            self.convs.append(conv)
#            self.bns.append(bn)
#
#
#    def forward(self, x, edge_index, batch):
#        if x is None:
#            x = torch.ones((batch.shape[0], 1)).to(self.device)
#
#        xs = []
#
#        for i in range(self.num_gc_layers):
#            x = F.relu(self.convs[i](x, edge_index))
#            x = self.bns[i](x)
#            xs.append(x)
#
#        xpool = [global_add_pool(x, batch) for x in xs]
#        x = torch.cat(xpool, 1)
#        
#
#        return x, torch.cat(xs, 1) # graph / node
#
#    def get_embeddings(self, loader):
#
#        ret = []
#        y = []
#        with torch.no_grad():
#            for data in loader:
#                if len(data) == 2:
#                    data = data[0]
#                data.to(self.device)
#                x, edge_index, batch = data.x, data.edge_index, data.batch
#               
#                if x is None:
#                    x = torch.ones((batch.shape[0],1)).to(self.device)
#                x, _ = self.forward(x, edge_index, batch)
#                ret.append(x.cpu().numpy())
#                y.append(data.y.cpu().numpy())
#        ret = np.concatenate(ret, 0)
#        y = np.concatenate(y, 0)
#        return ret, y


# adopted from https://github.com/illidanlab/MoCL-DK
class GINConv(MessagePassing):
    """
    Extension of GIN aggregation to incorporate edge information by concatenation.
    Args:
        emb_dim (int): dimensionality of embeddings for nodes and edges.
        input_layer (bool): whethe the GIN conv is applied to input layer or not. (Input node labels are uniform...)
    See https://arxiv.org/abs/1810.00826
    """
    def __init__(self, emb_dim, aggr = "add"):#, input_layer = False):
        super(GINConv, self).__init__()
        # multi-layer perceptron
        self.mlp = torch.nn.Sequential(torch.nn.Linear(emb_dim, 2*emb_dim), torch.nn.ReLU(), torch.nn.Linear(2*emb_dim, emb_dim))
        self.edge_embedding1 = torch.nn.Embedding(num_bond_type, emb_dim)
        self.edge_embedding2 = torch.nn.Embedding(num_bond_direction, emb_dim)

        torch.nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        torch.nn.init.xavier_uniform_(self.edge_embedding2.weight.data)
        self.aggr = aggr

        
    def forward(self, x, edge_index, edge_attr):

        #add self loops in the edge space
        edge_index = add_self_loops(edge_index, num_nodes = x.size(0))
        
        #add features corresponding to self-loop edges.
        self_loop_attr = torch.zeros(x.size(0), 2)
        self_loop_attr[:,0] = 4 #bond type for self-loop edge
        self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim = 0)

        edge_embeddings = self.edge_embedding1(edge_attr[:,0]) + self.edge_embedding2(edge_attr[:,1])
        
        return self.propagate(edge_index[0], x=x, edge_attr=edge_embeddings)
        

    def message(self, x_j, edge_attr):
        return x_j + edge_attr

    def update(self, aggr_out):
        return self.mlp(aggr_out)

class GNN(torch.nn.Module):
    """
    Extension of GIN to incorporate edge information by concatenation.
    Args:
        num_layer (int): the number of GNN layers
        emb_dim (int): dimensionality of embeddings
        JK (str): last, concat, max or sum.
        max_pool_layer (int): the layer from which we use max pool rather than add pool for neighbor aggregation
        drop_ratio (float): dropout rate
        gnn_type: gin, gat, graphsage, gcn
        
    See https://arxiv.org/abs/1810.00826
    JK-net: https://arxiv.org/abs/1806.03536
    Output:
        node representations
    """
    def __init__(self, num_layer, emb_dim, device, JK = "last", drop_ratio = 0, gnn_type = "gin"):
        super(GNN, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        self.device = device

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.x_embedding1 = torch.nn.Embedding(num_atom_type, emb_dim)
        self.x_embedding2 = torch.nn.Embedding(num_chirality_tag, emb_dim)

        torch.nn.init.xavier_uniform_(self.x_embedding1.weight.data)
        torch.nn.init.xavier_uniform_(self.x_embedding2.weight.data)

        # for evaluation
        self.pool = global_mean_pool
        
        #self.graph_pred_linear = torch.nn.Linear(2*self.emb_dim, self.num_tasks)

        ###List of message-passing GNN convs
        self.gnns = torch.nn.ModuleList()
        
        for layer in range(num_layer):
            if gnn_type == "gin":
                self.gnns.append(GINConv(emb_dim, aggr = "add")) #, input_layer = input_layer))

        self.batch_norms = torch.nn.ModuleList()
        for layer in range(num_layer):
            self.batch_norms.append(torch.nn.BatchNorm1d(emb_dim))

    #def forward(self, x, edge_index, edge_attr):
    def mask_forward(self, *argv):
        
        x, mask, edge_index, edge_attr = argv[0], argv[1], argv[2], argv[3]
        
        x = self.x_embedding1(x[:,0]) + self.x_embedding2(x[:,1])

        x[:, mask] = 0.


        #h_list = [x]
        h_list = [x]

        for layer in range(self.num_layer):
            h = self.gnns[layer](h_list[layer], edge_index, edge_attr)
            h = self.batch_norms[layer](h)
            if layer == self.num_layer - 1:
                #remove relu from the last layer
                h = F.dropout(h, self.drop_ratio, training = self.training)
            
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training = self.training)
            h_list.append(h)

         ### Different implementations of Jk-concat

        if self.JK == "last":
            node_representation = h_list[-1]
        
        return node_representation

    def forward(self, *argv):
        if len(argv) == 3:
            x, edge_index, edge_attr = argv[0], argv[1], argv[2]
        elif len(argv) == 4:
            x, edge_index, edge_attr, batch = argv[0], argv[1], argv[2], argv[3]
        elif len(argv) == 1:
            data = argv[0]
            x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        else:
            raise ValueError("unmatched number of arguments.")

        x = self.x_embedding1(x[:,0]) + self.x_embedding2(x[:,1])

        #h_list = [x]
        h_list = [x]

        for layer in range(self.num_layer):
            h = self.gnns[layer](h_list[layer], edge_index, edge_attr)
            h = self.batch_norms[layer](h)
            if layer == self.num_layer - 1:
                #remove relu from the last layer
                h = F.dropout(h, self.drop_ratio, training = self.training)
            
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training = self.training)
            h_list.append(h)

         ### Different implementations of Jk-concat
        if self.JK == "concat":
            h_list = h_list[1:]
            node_representation = torch.cat(h_list, 1)
            xpool = [global_add_pool(x, batch) for x in h_list]
            graph_representation = torch.cat(xpool, 1)

            return node_representation, graph_representation

        elif self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "max":
            h_list = [h.unsqueeze_(0) for h in h_list]
            node_representation = torch.max(torch.cat(h_list, dim = 0), dim = 0)[0]
        elif self.JK == "sum":
            h_list = [h.unsqueeze_(0) for h in h_list]
            node_representation = torch.sum(torch.cat(h_list, dim = 0), dim = 0)[0]

        return node_representation
    


class GNN_graphpred(torch.nn.Module):
    """
    Extension of GIN to incorporate edge information by concatenation.
    Args:
        num_layer (int): the number of GNN layers
        emb_dim (int): dimensionality of embeddings
        num_tasks (int): number of tasks in multi-task learning scenario
        drop_ratio (float): dropout rate
        JK (str): last, concat, max or sum.
        graph_pooling (str): sum, mean, max, attention, set2set
        gnn_type: gin, gcn, graphsage, gat
        
    See https://arxiv.org/abs/1810.00826
    JK-net: https://arxiv.org/abs/1806.03536
    """
    def __init__(self, num_layer, emb_dim, num_tasks, args, JK = "last", drop_ratio = 0, graph_pooling = "mean", gnn_type = "gin"):
        super(GNN_graphpred, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        self.emb_dim = emb_dim
        self.num_tasks = num_tasks

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.gnn = GNN(num_layer, emb_dim, args.device, drop_ratio = drop_ratio, gnn_type = gnn_type)
        
        self.pool = global_mean_pool
        
        self.mult = 1
        
        if self.JK == "concat":
            self.graph_pred_linear = torch.nn.Linear(self.mult * (self.num_layer + 1) * self.emb_dim, self.num_tasks)
        else:
            self.graph_pred_linear = torch.nn.Linear(self.mult * self.emb_dim, self.num_tasks)

    def from_pretrained(self, model_file):
        #self.gnn = GNN(self.num_layer, self.emb_dim, JK = self.JK, drop_ratio = self.drop_ratio)
        self.gnn.load_state_dict(torch.load(model_file))

    def forward(self, *argv):
        if len(argv) == 4:
            x, edge_index, edge_attr, batch = argv[0], argv[1], argv[2], argv[3]
        elif len(argv) == 1:
            data = argv[0]
            x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        else:
            raise ValueError("unmatched number of arguments.")

        node_representation = self.gnn(x, edge_index, edge_attr)

        return self.graph_pred_linear(self.pool(node_representation, batch))


class GCNEncoder(torch.nn.Module):
    def __init__(self, num_features, dim, num_gc_layers, device):
        super(GCNEncoder, self).__init__()

        self.num_gc_layers = num_gc_layers
        self.convs = torch.nn.ModuleList()
        #self.bns = torch.nn.ModuleList()
        self.device = device
        self.act = nn.PReLU()

        for i in range(num_gc_layers):

            if i:
                conv = GCNConv(dim, dim)
            else:
                conv = GCNConv(num_features, dim)
                
            self.convs.append(conv)
            

    def forward(self, x, edge_index, batch):
        if x is None:
            x = torch.ones((batch.shape[0], 1)).to(self.device)

        xs = []
        for i in range(self.num_gc_layers):
            x = self.convs[i](x, edge_index)
            x = self.act(x)
            #x = self.bns[i](x)
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