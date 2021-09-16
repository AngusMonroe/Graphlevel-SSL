from torch_geometric.data import DataLoader
import torch
import torch.nn as nn
from torch_geometric.nn import global_mean_pool
import numpy as np
import pandas as pd
from torch import optim

from copy import deepcopy
from tqdm import tqdm

from utils import set_device, setup_seed, task, currentTime, train_cls, eval_cls, config2string
from loader import MoleculeDataset
from .Encoder import GNN, GNN_graphpred
from splitters import *
import torch.nn.functional as F

import networkx as nx
from torch_scatter import scatter
from torch_geometric.utils import dropout_adj, degree, to_undirected, to_networkx

num_atom_type = 120 #including the extra mask tokens
num_chirality_tag = 3

num_bond_type = 6 #including aromatic and self-loop edge, and extra masked tokens
num_bond_direction = 3

class GCA_Trainer():
    def __init__(self, args):
        self._args = args
        self._init()
        self.config_str = config2string(args)
        print("\n[Config] {}\n".format(self.config_str))
        

    def _init(self):
        
        args = self._args

        self._device = set_device(args.device)
        setup_seed(args.seed)

        #path = osp.join(osp.dirname(osp.realpath(__file__)), '.', 'data', args.name)
        path = "dataset/" + args.name
        # path = osp.join(osp.dirname(osp.realpath(__file__)), '.', 'data', args.name)
        dataset = MoleculeDataset(path, dataset=args.name)
        dataset.aug = 'none'
        dataset.shuffle()

        self._loader = DataLoader(dataset=dataset, batch_size=args.batch, shuffle=False)
        
        self.num_tasks = task(args.name)
        
        self.gnn = GNN(args.layer, args.emb_dim, device=args.device, JK = "last", drop_ratio = args.dropout_ratio, gnn_type = 'gin')

        self._model = GCA(self.gnn, args)
        self._model.to(self._device)
        print(self._model)

        self._optimizer = optim.Adam(params=self._model.parameters(), lr=args.lr)#, weight_decay= 1e-5)

        
        # evaluation

        setup_seed(args.runseed)
        eval_dataset = MoleculeDataset(path, dataset=args.name)    
    
        if args.split == "scaffold":
            smiles_list = pd.read_csv('dataset/' + args.dataset + '/processed/smiles.csv', header=None)[0].tolist()
            train_dataset, valid_dataset, test_dataset = scaffold_split(eval_dataset, smiles_list, null_value=0, frac_train=0.8,frac_valid=0.1, frac_test=0.1)
            print("scaffold")
        elif args.split == "random":
            train_dataset, valid_dataset, test_dataset = random_split(eval_dataset, null_value=0, frac_train=0.8,frac_valid=0.1, frac_test=0.1, seed = args.seed)
            print("random")
        elif args.split == "random_scaffold":
            smiles_list = pd.read_csv('dataset/' + args.dataset + '/processed/smiles.csv', header=None)[0].tolist()
            train_dataset, valid_dataset, test_dataset = random_scaffold_split(eval_dataset, smiles_list, null_value=0, frac_train=0.8,frac_valid=0.1, frac_test=0.1, seed = args.seed)
            print("random scaffold")
        else:
            raise ValueError("Invalid split option.")

        self._train_loader = DataLoader(train_dataset, batch_size=args.batch, shuffle=True)#, num_workers = args.num_workers)
        self._val_loader = DataLoader(valid_dataset, batch_size=args.batch, shuffle=False)#, num_workers = args.num_workers)
        self._test_loader = DataLoader(test_dataset, batch_size=args.batch, shuffle=False)#, num_workers = args.num_workers)

        self._eval_model = GNN_graphpred(args.layer, args.emb_dim, self.num_tasks, args, drop_ratio = args.dropout_ratio)


    def experiment(self):
        # get Random Initial accuracy
        
        args = self._args
        
        if args.Scratch:
            pass
        else:
        
            # Start Model Training
            print("Training Start!")
            self._model.train()
            for epoch in range(1, self._args.epochs+1):
                total_pos = 0
                train_loss_accum = 0
                for step, batch in enumerate(tqdm(self._loader, desc="Iteration")):
                    batch = batch.to(self._device)
                    embeddings1 = deepcopy(self._model.encoder.x_embedding1)
                    embeddings2 = deepcopy(self._model.encoder.x_embedding2)
                    embeddings1.weight.requires_grad=False
                    embeddings2.weight.requires_grad=False

                    augmentation = Augmentation(args, batch, embeddings1, embeddings2, float(args.drop_feature_rate_1),float(args.drop_feature_rate_2),float(args.drop_edge_rate_1),float(args.drop_edge_rate_2))
                    mask_1, mask_2, edge_index_1, edge_attr_1, edge_index_2, edge_attr_2 = augmentation.augmentation()
                    
                    self._optimizer.zero_grad()

                    z1 = self._model(batch.x, mask_1, edge_index_1, edge_attr_1, batch.batch)
                    z2 = self._model(batch.x, mask_2, edge_index_2, edge_attr_2, batch.batch)

                    loss = self._model.loss(z1, z2, batch.batch)

                    loss.backward()
                    self._optimizer.step()

                    train_loss_accum += float(loss.detach().cpu().item())
                    

                st = '[{}][Epoch {}/{}] Loss: {:.4f} |'\
                    .format(currentTime(), epoch, args.epochs, train_loss_accum)
                print(st)


            self._eval_model.gnn = self.gnn
            
        self._eval_model.to(self._device)

        #set up optimizer
        #different learning rate for different part of GNN
        model_param_group = []
        model_param_group.append({"params": self._eval_model.gnn.parameters()})
        model_param_group.append({"params": self._eval_model.graph_pred_linear.parameters(), "lr":args.lr*args.lr_scale})
        self._eval_optimizer = optim.Adam(model_param_group, lr=args.lr)#, weight_decay=args.decay)
        print(self._eval_optimizer)


        if args.Scratch:
            print("finetune protocol, train all the layers!")
        else:
            print("linear protocol, only train the top layer!")
            for name, param in self._eval_model.named_parameters():
                if not 'pred_linear' in name:
                    param.requires_grad = False

        # all task info summary
        print('=========task summary=========')
        print('Dataset: ', args.name)

        if args.Scratch:
            print('scratch or finetune: scratch')
            print('loaded model from: - ')
        else:
            print('scratch or finetune: finetune')
            print('loaded model from: ', args.name)

        print('task type:', 'cls')
        print('=========task summary=========')


        # evaluation

        txtfile = f'./results/GCA/{args.name}.txt'

        with open(txtfile, "a") as f:
            f.write(self.config_str)
            f.write('\n')
            f.close()

        wait = 0
        best_auc = 0
        patience = 10
        for epoch in range(1, args.epochs+1):
            print("====epoch " + str(epoch))
            
            train_cls(args, self._eval_model, self._device, self._train_loader, self._eval_optimizer)

            print("====Evaluation")
            
            val_auc = eval_cls(args, self._eval_model, self._device, self._val_loader)
            test_auc = eval_cls(args, self._eval_model, self._device, self._test_loader)

            print("val: %f test: %f" %(val_auc, test_auc))

            # Early Stopping
            if np.greater(val_auc, best_auc):
                best_auc = val_auc
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    print('Early stop at Epoch: {:d} with final val auc: {:.4f}'.format(epoch, val_auc))
                    break

        with open(txtfile, "a") as f:
            f.write('| epoch : {} | val_auc : {} | test_auc : {} |'.format(epoch, val_auc, test_auc))
            f.write('\n')
            f.close()
            


class GCA(nn.Module):
    def __init__(self, gnn, args):
        super(GCA, self).__init__()
        self.encoder = gnn
        self.tau=0.5
        
        self.pool = global_mean_pool
        
        #self.projection_head = nn.Sequential(nn.Linear(args.emb_dim, args.emb_dim), nn.ReLU(inplace=True), nn.Linear(args.emb_dim, args.emb_dim))
        # pred_dim 사용불가(ByoW랑 비교 X)

        self.fc1 = torch.nn.Linear(args.emb_dim, args.emb_dim)
        self.fc2 = torch.nn.Linear(args.emb_dim, args.emb_dim)

    def forward(self, x, mask, edge_index, edge_attr, batch):
        node_rep = self.encoder.mask_forward(x, mask, edge_index, edge_attr)
        return node_rep

    def projection(self, z):
        z = F.elu(self.fc1(z))
        return self.fc2(z)

    def sim(self, z1: torch.Tensor, z2: torch.Tensor):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())

    def semi_loss(self, z1: torch.Tensor, z2: torch.Tensor, batch):
        f = lambda x: torch.exp(x / self.tau)
        
        full_mask = None
        for i in range(len(batch)):

            mask = (batch == batch[i])
            if full_mask == None:
                full_mask = mask.reshape(1, -1)
            else:
                full_mask = torch.cat((full_mask, mask.reshape(1, -1)), dim=0)

        refl_sim = f(self.sim(z1, z1)) * full_mask
        between_sim = f(self.sim(z1, z2)) * full_mask

        return -torch.log(between_sim.diag() / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()))

    def loss(self, z1, z2, batch, mean=True):
        h1 = self.projection(z1)
        h2 = self.projection(z2)

        l1 = self.semi_loss(h1, h2, batch)
        l2 = self.semi_loss(h2, h1, batch)

        ret = (l1 + l2) * 0.5
        ret = ret.mean() if mean else ret.sum()

        return ret


class Augmentation:
    def __init__(self, args, data, embedding1, embedding2, p_f1 = 0.2, p_f2 = 0.2, p_e1 = 0.2, p_e2 = 0.2):
        """
        two simple graph augmentation functions --> "Node feature masking" and "Edge masking"
        Random binary node feature mask following Bernoulli distribution with parameter p_f
        Random binary edge mask following Bernoulli distribution with parameter p_e
        """
        self.p_f1 = p_f1
        self.p_f2 = p_f2
        self.p_e1 = p_e1
        self.p_e2 = p_e2

        self.args = args
        self.data = data

        self.embedding1 = embedding1
        self.embedding2 = embedding2

        # edge weight
        if args.drop_scheme == 'degree':
            self.drop_weights = degree_drop_weights(self.data.edge_index)
        elif args.drop_scheme == 'pr':
            self.drop_weights = pr_drop_weights(self.data.edge_index, aggr='sink', k=200)
        elif args.drop_scheme == 'evc':
            self.drop_weights = evc_drop_weights(self.data)
        else:
            self.drop_weights = None

        # node weight
        if args.drop_scheme == 'degree':
            edge_index_ = to_undirected(self.data.edge_index)
            node_deg = degree(edge_index_[1])

            # 마지막 node에 edge가 하나도 없을 때
            d_node_num = self.data.x.size(0) - ((edge_index_[1].unique())[-1] + 1).item()
            if d_node_num != 0:
                last_nodes = torch.zeros(d_node_num).to(self.args.device)
                node_deg = torch.cat([node_deg, last_nodes], dim=0)

            #if (node_deg==0).sum() != 0:
            #    print((node_deg==0).sum().item())
                

            self.feature_weights = feature_drop_weights_dense(self.data.x, self.embedding1, self.embedding2, node_c=node_deg)
        
        elif args.drop_scheme == 'pr':
            node_pr = compute_pr(self.data.edge_index)
            self.feature_weights = feature_drop_weights_dense(self.data.x, node_c=node_pr)
        
        elif args.drop_scheme == 'evc':
            node_evc = eigenvector_centrality(self.data)
            self.feature_weights = feature_drop_weights_dense(self.data.x, node_c=node_evc)
        
        else:
            self.feature_weights = torch.ones((self.data.x.size(1),))
    

    def drop_edge(self, idx: int):

        if self.args.drop_scheme == 'uniform':
            if idx == 1:
                return dropout_adj(self.data.edge_index, p=self.args.drop_edge_rate_1)[0]
            else :
                return dropout_adj(self.data.edge_index, p=self.args.drop_edge_rate_2)[0]
        elif self.args.drop_scheme in ['degree', 'evc', 'pr']:
            if idx == 1:
                return drop_edge_weighted(self.data.edge_index, self.data.edge_attr, self.drop_weights, p=self.args.drop_edge_rate_1, threshold=0.7)
            else :
                return drop_edge_weighted(self.data.edge_index, self.data.edge_attr, self.drop_weights, p=self.args.drop_edge_rate_1, threshold=0.7)
        else:
            raise Exception(f'undefined drop scheme: {self.args.drop_scheme}')


    def augmentation(self):
        
        # edge drop
        edge_index_1, edge_attr1 = self.drop_edge(1)
        edge_index_2, edge_attr2 = self.drop_edge(2)

        

        # node drop
        if self.args.drop_scheme in ['pr', 'degree', 'evc']:
            x_1 = drop_feature_weighted_2(self.data.x, self.embedding1, self.embedding2, self.feature_weights, self.args.drop_feature_rate_1)
            x_2 = drop_feature_weighted_2(self.data.x, self.embedding1, self.embedding2, self.feature_weights, self.args.drop_feature_rate_2)

        else:
            x_1 = drop_feature(self.data.x, self.args.drop_feature_rate_1)
            x_2 = drop_feature(self.data.x, self.args.drop_feature_rate_2)

        return x_1, x_2, edge_index_1, edge_attr1, edge_index_2, edge_attr2

    def __call__(self):
        
        return self.augmentation()





def drop_feature(x, drop_prob):
    drop_mask = torch.empty((x.size(1),), dtype=torch.float32, device=x.device).uniform_(0, 1) < drop_prob
    x = x.clone()
    x[:, drop_mask] = 0

    return x


def drop_feature_weighted(x, w, p: float, threshold: float = 0.7):
    w = w / w.mean() * p
    w = w.where(w < threshold, torch.ones_like(w) * threshold)
    drop_prob = w.repeat(x.size(0)).view(x.size(0), -1)

    drop_mask = torch.bernoulli(drop_prob).to(torch.bool)

    x = x.clone()
    x[drop_mask] = 0.

    return x


def drop_feature_weighted_2(x, embedding1, embedding2, w, p: float, threshold: float = 0.7):
    
    #x = embedding1(x[:,0]) + embedding2(x[:,1])
    w = w / w.mean() * p
    w = w.where(w < threshold, torch.ones_like(w) * threshold)
    drop_prob = w

    drop_mask = torch.bernoulli(drop_prob).to(torch.bool)

    #x = x.clone()
    #x[:, drop_mask] = 0.

    return drop_mask


def feature_drop_weights(x, node_c):
    x = x.to(torch.bool).to(torch.float32)
    w = x.t() @ node_c
    w = w.log()
    s = (w.max() - w) / (w.max() - w.mean())

    return s


def feature_drop_weights_dense(x, embedding1, embedding2, node_c):
    x = embedding1(x[:,0]) + embedding2(x[:,1])
    x = x.abs()
    w = x.t() @ node_c
    w = w.log()
    s = (w.max() - w) / (w.max() - w.mean())

    return s


def drop_edge_weighted(edge_index, edge_attr, edge_weights, p: float, threshold: float = 1.):
    edge_weights = edge_weights / edge_weights.mean() * p
    edge_weights = edge_weights.where(edge_weights < threshold, torch.ones_like(edge_weights) * threshold)
    sel_mask = torch.bernoulli(1. - edge_weights).to(torch.bool)

    return edge_index[:, sel_mask], edge_attr[sel_mask, :]


def degree_drop_weights(edge_index):
    edge_index_ = to_undirected(edge_index)
    deg = degree(edge_index_[1])
    deg_col = deg[edge_index[1]].to(torch.float32)
    s_col = torch.log(deg_col)
    weights = (s_col.max() - s_col) / (s_col.max() - s_col.mean())

    return weights


def pr_drop_weights(edge_index, aggr: str = 'sink', k: int = 10):
    pv = compute_pr(edge_index, k=k)
    pv_row = pv[edge_index[0]].to(torch.float32)
    pv_col = pv[edge_index[1]].to(torch.float32)
    s_row = torch.log(pv_row)
    s_col = torch.log(pv_col)
    if aggr == 'sink':
        s = s_col
    elif aggr == 'source':
        s = s_row
    elif aggr == 'mean':
        s = (s_col + s_row) * 0.5
    else:
        s = s_col
    weights = (s.max() - s) / (s.max() - s.mean())

    return weights


def evc_drop_weights(data):
    evc = eigenvector_centrality(data)
    evc = evc.where(evc > 0, torch.zeros_like(evc))
    evc = evc + 1e-8
    s = evc.log()

    edge_index = data.edge_index
    s_row, s_col = s[edge_index[0]], s[edge_index[1]]
    s = s_col

    return (s.max() - s) / (s.max() - s.mean())


def compute_pr(edge_index, damp: float = 0.85, k: int = 10):
    num_nodes = edge_index.max().item() + 1
    deg_out = degree(edge_index[0])
    x = torch.ones((num_nodes, )).to(edge_index.device).to(torch.float32)

    for i in range(k):
        edge_msg = x[edge_index[0]] / deg_out[edge_index[0]]
        agg_msg = scatter(edge_msg, edge_index[1], reduce='sum')

        x = (1 - damp) * x + damp * agg_msg

    return x


def eigenvector_centrality(data):
    graph = to_networkx(data)
    x = nx.eigenvector_centrality_numpy(graph)
    x = [x[i] for i in range(data.num_nodes)]
    return torch.tensor(x, dtype=torch.float32).to(data.edge_index.device)






