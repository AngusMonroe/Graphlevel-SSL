from torch_geometric.data import DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch import optim
from tensorboardX import SummaryWriter

import os
from utils import *
import copy
import sys

from .Encoder import GINEncoder
from evaluate_embedding import evaluate_embedding

from torch_geometric.datasets import TUDataset
from argument import parse_args
from utils import setup_seed
args, unknown = parse_args()
setup_seed(args.seed)

class GCA_Trainer():
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

        self._loader = DataLoader(dataset=self._dataset, batch_size=args.batch)
        self._eval_loader = DataLoader(dataset=self._eval_dataset, batch_size=args.batch)
        print(f"Data: {self._dataset.data}")

        dataset_num_features = max(self._dataset.num_features, 1)

        self._model = GCA(dataset_num_features, args).to(self._device)
        print(self._model)

        self._optimizer = optim.Adam(params=self._model.parameters(), lr=args.lr)

    def experiment(self):
        
        
        args = self._args

        print("initial accuracy ")
        self._model.eval()
        emb, y = self._model.encoder.get_embeddings(self._eval_loader)
        init_acc = evaluate_embedding(emb, y)

        txtfile = f'./results/GCA/{args.name}.txt'

        with open(txtfile, "a") as f:
            f.write(self.config_str)
            f.write('\n')
            f.write("init --> | val_acc: {} | test_acc : {} |\n".format(init_acc[0], init_acc[1]))
            f.close()


        accuracies = {'val':[] ,'test':[]}
        print("Training Start!")
        for epoch in range(1, args.epochs+1):
            self._model.train()
            for data in self._loader:
                data = data.to(self._device)

                augmentation = Augmentation_GCA(args, data, float(args.drop_feature_rate_1),float(args.drop_feature_rate_2),float(args.drop_edge_rate_1),float(args.drop_edge_rate_2))
                x_1, x_2, edge_index_1, edge_index_2 = augmentation.augmentation()
                _, z1 = self._model(x_1, edge_index_1, data.batch)
                _, z2 = self._model(x_2, edge_index_2, data.batch)

                loss = self._model.loss(z1, z2, data.batch)
                loss.backward()
                self._optimizer.step()

                sys.stdout.write('\rEpoch {}/{}, loss {:.4f}'.format(epoch, args.epochs, loss))
                sys.stdout.flush()

            if (epoch) % args.eval_freq == 0:
                self._model.eval()
                emb, y = self._model.encoder.get_embeddings(self._eval_loader)
                
                acc_val, acc, acc_val_std, acc_std = evaluate_embedding(emb, y)

                st = '  [{}][Epoch {}/{}] Val Acc: {:.4f} | Test Acc : {:.4f} |'\
                    .format(currentTime(), epoch, args.epochs, acc_val, acc)
                print(st)

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



class GCA(torch.nn.Module):
    def __init__(self, dataset_num_features, args):
        super(GCA, self).__init__()
        self.encoder = GINEncoder(dataset_num_features, args.hidden_dim, args.layer, args.device)
        self.tau = 0.5

        num_hidden = args.hidden_dim * args.layer 
        self.fc1 = torch.nn.Linear(num_hidden, num_hidden * 2)
        self.fc2 = torch.nn.Linear(num_hidden * 2, num_hidden)

        self.num_hidden = num_hidden

    def forward(self, x, edge_index, batch) -> torch.Tensor:
        graph, node = self.encoder(x, edge_index, batch)
        return graph, node

    def projection(self, z: torch.Tensor) -> torch.Tensor:
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

    def loss(self, z1: torch.Tensor, z2: torch.Tensor, batch, mean: bool = True ):
        h1 = self.projection(z1)
        h2 = self.projection(z2)

        l1 = self.semi_loss(h1, h2, batch)
        l2 = self.semi_loss(h2, h1, batch)

        ret = (l1 + l2) * 0.5
        ret = ret.mean() if mean else ret.sum()

        return ret

class Augmentation_GCA:
    def __init__(self, args, data, p_f1 = 0.2, p_f2 = 0.2, p_e1 = 0.2, p_e2 = 0.2):
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

        if args.drop_scheme == 'degree':
            self.drop_weights = degree_drop_weights(data.edge_index)
        elif args.drop_scheme == 'pr':
            self.drop_weights = pr_drop_weights(data.edge_index, aggr='sink', k=200)
        elif args.drop_scheme == 'evc':
            self.drop_weights = evc_drop_weights(data)
        else:
            self.drop_weights = None

        if args.drop_scheme == 'degree':
            edge_index_ = to_undirected(data.edge_index)
            node_deg = degree(edge_index_[1])
            self.feature_weights = feature_drop_weights(data.x, node_c=node_deg)
        
        elif args.drop_scheme == 'pr':
            node_pr = compute_pr(data.edge_index)
            self.feature_weights = feature_drop_weights(data.x, node_c=node_pr)
        
        elif args.drop_scheme == 'evc':
            node_evc = eigenvector_centrality(data)
            self.feature_weights = feature_drop_weights(data.x, node_c=node_evc)
        
        else:
            self.feature_weights = torch.ones((data.x.size(1),))
    

    def drop_edge(self, idx: int):

        if self.args.drop_scheme == 'uniform':
            if idx == 1:
                return dropout_adj(self.data.edge_index, p=self.args.drop_edge_rate_1)[0]
            else :
                return dropout_adj(self.data.edge_index, p=self.args.drop_edge_rate_2)[0]
                
        elif self.args.drop_scheme in ['degree', 'evc', 'pr']:
            if idx == 1:
                return drop_edge_weighted(self.data.edge_index, self.drop_weights, p=self.args.drop_edge_rate_1, threshold=0.7)
            else :
                return drop_edge_weighted(self.data.edge_index, self.drop_weights, p=self.args.drop_edge_rate_1, threshold=0.7)
        else:
            raise Exception(f'undefined drop scheme: {self.args.drop_scheme}')


    def augmentation(self):

        edge_index_1 = self.drop_edge(1)
        edge_index_2 = self.drop_edge(2)

        x_1 = drop_feature(self.data.x, self.args.drop_feature_rate_1)
        x_2 = drop_feature(self.data.x, self.args.drop_feature_rate_2)

        if self.args.drop_scheme in ['pr', 'degree', 'evc']:
            x_1 = drop_feature_weighted_2(self.data.x, self.feature_weights, self.args.drop_feature_rate_1)
            x_2 = drop_feature_weighted_2(self.data.x, self.feature_weights, self.args.drop_feature_rate_2)

        return x_1, x_2, edge_index_1, edge_index_2

    def __call__(self):
        
        return self.augmentation()
