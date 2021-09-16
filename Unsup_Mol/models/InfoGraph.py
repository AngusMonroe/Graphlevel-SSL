from torch_geometric.data import DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch import optim
from torch_geometric.nn import global_mean_pool
import pandas as pd
from tqdm import tqdm

from utils import set_device, setup_seed, task, currentTime, train_cls, eval_cls, config2string
from loader import MoleculeDataset
from .Encoder import GNN, GNN_graphpred
from splitters import *
from cortex_DIM.functions.gan_losses import get_positive_expectation, get_negative_expectation


class InfoGraph_Trainer():
    def __init__(self, args):
        self._args = args
        self._init()
        self.config_str = config2string(args)
        print("\n[Config] {}\n".format(self.config_str))
        
        
    def _init(self):
        args = self._args

        self._device = set_device(args.device)
        setup_seed(args.seed)

        path = "dataset/" + args.name
  
        dataset = MoleculeDataset(path, dataset=args.name)
        dataset.aug = 'none'
        dataset.shuffle()
        
        self._loader = DataLoader(dataset=dataset, batch_size=args.batch, shuffle=False)  # [self._dataset.data]
        
        self.num_tasks = task(args.name)
        
        self.gnn = GNN(args.layer, args.emb_dim, device=args.device, JK = "concat", drop_ratio = args.dropout_ratio, gnn_type = 'gin')

        self._model = InfoGraph(self.gnn, args)
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

        args = self._args

        if args.Scratch:
            pass
        else:
            
            print("Training Start!")
            self._model.train()
            for epoch in range(1, args.epochs+1):
                train_acc_accum = 0
                train_loss_accum = 0
                for step, batch in enumerate(tqdm(self._loader, desc="Iteration")):
                    
                    batch = batch.to(self._device)

                    self._optimizer.zero_grad()
                    
                    loss = self._model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
                    
                    loss.backward()
                    self._optimizer.step()

                    train_loss_accum += float(loss.detach().cpu().item())

                st = '[{}][Epoch {}/{}] Loss: {:.4f} |'\
                    .format(currentTime(), epoch, args.epochs, train_loss_accum)
                print(st)

            self._eval_model.gnn = self.gnn
            self._eval_model.gnn.JK = 'last'

        self._eval_model.to(self._device)


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

        txtfile = f'./results/InfoGraph/{args.name}.txt'

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

        

class InfoGraph(nn.Module):

    def __init__(self, gnn, args):
        super(InfoGraph, self).__init__()
        self.encoder = gnn
        self.pool = global_mean_pool
        
        self.projection_head = nn.Sequential(nn.Linear(args.emb_dim, args.emb_dim), nn.ReLU(inplace=True), nn.Linear(args.emb_dim, args.emb_dim))

        training_dim = args.emb_dim * args.layer
        self.local_d = FF(training_dim)
        self.global_d = FF(training_dim)
        self.prior = args.prior
        
        self.args = args

        if self.prior:
            self.prior_d = PriorDiscriminator(training_dim)

        self.init_emb

    def init_emb(self):
        
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def forward(self, x, edge_index, edge_attr, batch):
        node_rep, graph_rep = self.encoder(x, edge_index, edge_attr, batch)
        
        g_enc = self.global_d(graph_rep)
        l_enc = self.local_d(node_rep)

        mode='fd'
        measure='JSD'
        local_global_loss = local_global_loss_(l_enc, g_enc, edge_index, batch, measure, self.args.device)
        
        if self.prior:
            prior = torch.rand_like(graph_rep)
            term_a = torch.log(self.prior_d(prior)).mean()
            term_b = torch.log(1.0 - self.prior_d(graph_rep)).mean()
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