import os.path as osp

from evaluate_embedding import evaluate_embedding
from .Encoder import GINEncoder
from torch_geometric.data import DataLoader
from torch_geometric.datasets import TUDataset
from torch_geometric.datasets import MoleculeNet

import numpy as np
import os.path as osp
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import *
from cortex_DIM.functions.gan_losses import get_positive_expectation, get_negative_expectation
from argument import parse_args
from utils import setup_seed
args, unknown = parse_args()
setup_seed(args.seed)


class InfoGraph_Trainer():
    def __init__(self, args):
        self._args = args
        self._init()
        self.config_str = config2string(args)
        print("\n[Config] {}\n".format(self.config_str))
        
    def _init(self):
        args = self._args

        self._device = set_device(args.device)
        
        path = osp.join(osp.dirname(osp.dirname(osp.realpath(__file__))), '.', 'data', args.name)
        self._dataset = TUDataset(path, name=args.name).shuffle()
        self._eval_dataset = TUDataset(path, name=args.name).shuffle()

        self._loader = DataLoader(dataset=self._dataset, batch_size=args.batch)  # [self._dataset.data]
        self._eval_loader = DataLoader(dataset=self._eval_dataset, batch_size=args.batch)
        print(f"Data: {self._dataset.data}")

        dataset_num_features = max(self._dataset.num_features, 1)

        self._model = InfoGraph(dataset_num_features, args).to(self._device)
        print(self._model)

        self._optimizer = torch.optim.Adam(params=self._model.parameters(), lr=args.lr)

    def experiment(self):
    
        args = self._args
        
        print("initial accuracy ")
        self._model.eval()
        emb, y = self._model.encoder.get_embeddings(self._eval_loader)
        init_acc = evaluate_embedding(emb, y)

        txtfile = f'./results/InfoGraph/{args.name}.txt'

        with open(txtfile, "a") as f:
            f.write(self.config_str)
            f.write('\n')
            f.write("init --> | val_acc: {} | test_acc : {} |\n".format(init_acc[0], init_acc[1]))
            f.close()

        accuracies = {'val':[] ,'test':[]}
        for epoch in range(1, args.epochs+1):
            loss_all = 0
            self._model.train()
            for data in self._loader:
                data = data.to(self._device)
                self._optimizer.zero_grad()

                loss = self._model(data.x, data.edge_index, data.batch, data.num_graphs)                
                loss_all += loss.item() * data.num_graphs
                loss.backward()
                self._optimizer.step()
                
            st = '[{}][Epoch {}/{}] Loss: {:.4f} |'\
                .format(currentTime(), epoch, args.epochs, loss_all)
            print(st)
            
            if epoch % args.eval_freq == 0:
                self._model.eval()
                emb, y = self._model.encoder.get_embeddings(self._eval_loader)

                acc_val, acc, acc_val_std, acc_std = evaluate_embedding(emb, y)
                accuracies['val'].append(acc_val)
                accuracies['test'].append(acc)
                
                with open(txtfile, "a") as f:
                    f.write('| epoch : {} | val_acc : {} | test_acc : {} |\n'.format(epoch, acc_val, acc))
                    f.close()

        final_acc = accuracies['test'][-1]
        best_test_acc = max(accuracies['test'])
        
        best_val_acc_index = accuracies['val'].index(max(accuracies['val']))
        best_val_test_acc = accuracies['test'][best_val_acc_index]

        print("\nTraining Done!")
        print("[Final] {}".format(final_acc))
        print('################################################')

        with open(txtfile, "a") as f:
            f.write("| final_acc : {} | best_test_acc : {} | best_val_test_acc : {} |\n".format(final_acc, best_test_acc, best_val_test_acc))
            f.close()



class InfoGraph(nn.Module):
    def __init__(self, dataset_num_features, args, alpha=0.5, beta=1., gamma=.1, **kwargs):
        super(InfoGraph, self).__init__()
        self.device = args.device
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.prior = args.prior

        self.embedding_dim = args.hidden_dim * args.layer
        self.encoder = GINEncoder(dataset_num_features, args.hidden_dim, args.layer, args.device)

        self.local_d = FF(self.embedding_dim)
        self.global_d = FF(self.embedding_dim)


        if self.prior:
            self.prior_d = PriorDiscriminator(self.embedding_dim)

        self.init_emb()

    def init_emb(self):
        initrange = -1.5 / self.embedding_dim
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def forward(self, x, edge_index, batch, num_graphs):
        # batch_size = data.num_graphs
        if x is None:
            x = torch.ones(batch.shape[0]).to(self.device)
            x = x.unsqueeze(1)

        y, M = self.encoder(x, edge_index, batch)

        g_enc = self.global_d(y)
        l_enc = self.local_d(M)

        mode='fd'
        measure='JSD'
        local_global_loss = local_global_loss_(l_enc, g_enc, edge_index, batch, measure, self.device)

        if self.prior:
            prior = torch.rand_like(y)
            term_a = torch.log(self.prior_d(prior)).mean()
            term_b = torch.log(1.0 - self.prior_d(y)).mean()
            PRIOR = - (term_a + term_b) * self.gamma
        else:
            PRIOR = 0

        return local_global_loss + PRIOR

class FF(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, input_dim),
            nn.ReLU()
        )
        self.linear_shortcut = nn.Linear(input_dim, input_dim)
        
    def forward(self, x):
        return self.block(x) + self.linear_shortcut(x)

class PriorDiscriminator(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.l0 = nn.Linear(input_dim, input_dim)
        self.l1 = nn.Linear(input_dim, input_dim)
        self.l2 = nn.Linear(input_dim, 1)

    def forward(self, x):
        h = F.relu(self.l0(x))
        h = F.relu(self.l1(h))
        return torch.sigmoid(self.l2(h))

def local_global_loss_(l_enc, g_enc, edge_index, batch, measure, device):
    '''
    Args:
        l: Local feature map.
        g: Global features.
        measure: Type of f-divergence. For use with mode `fd`
        mode: Loss mode. Fenchel-dual `fd`, NCE `nce`, or Donsker-Vadadhan `dv`.
    Returns:
        torch.Tensor: Loss.
    '''
    num_graphs = g_enc.shape[0]
    num_nodes = l_enc.shape[0]

    pos_mask = torch.zeros((num_nodes, num_graphs)).to(device)
    neg_mask = torch.ones((num_nodes, num_graphs)).to(device)
    for nodeidx, graphidx in enumerate(batch):
        pos_mask[nodeidx][graphidx] = 1.
        neg_mask[nodeidx][graphidx] = 0.

    res = torch.mm(l_enc, g_enc.t())

    E_pos = get_positive_expectation(res * pos_mask, measure, average=False).sum()
    E_pos = E_pos / num_nodes
    E_neg = get_negative_expectation(res * neg_mask, measure, average=False).sum()
    E_neg = E_neg / (num_nodes * (num_graphs - 1))

    return E_neg - E_pos