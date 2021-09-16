from .Encoder import GINEncoder, GCNEncoder
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from evaluate_embedding import evaluate_embedding
from utils import *

from torch_geometric.data import DataLoader

from aug import TUDataset_aug as TUDataset
from argument import parse_args
from utils import setup_seed
args, unknown = parse_args()
setup_seed(args.seed)


class MVGRL_Trainer():
    def __init__(self, args):
        
        self._args = args
        self._init()
        self.config_str = config2string(args)
        print("\n[Config] {}\n".format(self.config_str))
        
    def _init(self):
        args = self._args

        self._device = set_device(args.device)

        path = osp.join(osp.dirname(osp.dirname(osp.realpath(__file__))), '.', 'data', args.name)

        self._dataset = TUDataset(path, name=args.name, aug='ppr').shuffle()
        self._eval_dataset = TUDataset(path, name=args.name, aug='none').shuffle()

        self._loader = DataLoader(dataset=self._dataset, batch_size=args.batch)
        self._eval_loader = DataLoader(dataset=self._eval_dataset, batch_size=args.batch)
        print(f"Data: {self._dataset.data}")

        try:
            dataset_num_features = self._dataset.get_num_feature()
        except:
            dataset_num_features = 1

        self._model = MVGRL(dataset_num_features, args).to(self._device) # argument 바꿔야함
        self._optimizer = torch.optim.Adam(self._model.parameters(), lr=args.lr)


    def experiment(self):

        args = self._args
        
        print("initial accuracy ")
        self._model.eval()
        emb, y = self._model.gnn1.get_embeddings(self._eval_loader)
        init_acc = evaluate_embedding(emb, y)

        txtfile = f'./results/MVGRL/{args.name}.txt'

        with open(txtfile, "a") as f:
            f.write(self.config_str)
            f.write('\n')
            f.write("init --> | val_acc: {} | test_acc : {} |\n".format(init_acc[0], init_acc[1]))
            f.close()

        
        accuracies = {'val':[] ,'test':[]}
        # Start Model Training
        print("Training Start!")
        self._model.train()
        for epoch in range(1, args.epochs+1):
            loss_all = 0
            self._model.train()
            for data in self._loader:
                data, data_aug = data
                data = data.to(self._device)
                data_aug = data_aug.to(self._device)
                self._optimizer.zero_grad()
                
                lv1, gv1, lv2, gv2 = self._model(data.x, data.edge_index, data_aug.edge_index, data.batch) #, data.num_graphs)                
                
                loss1 = local_global_loss_(lv1, gv2, data.batch, 'JSD')#, mask)
                loss2 = local_global_loss_(lv2, gv1, data.batch, 'JSD')#, mask)

                loss = loss1 + loss2 

                loss.backward()
                self._optimizer.step()

                loss_all += loss.item() * data.num_graphs

            st = '[{}][Epoch {}/{}] Loss: {:.4f} |'\
            .format(currentTime(), epoch, args.epochs, loss_all)
            print(st)

            if (epoch) % args.eval_freq == 0:
                self._model.eval()
                emb, y = self._model.gnn1.get_embeddings(self._eval_loader)

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




class MVGRL(nn.Module):
    def __init__(self, dataset_num_features, args):
        super(MVGRL, self).__init__()
        self.mlp1 = MLP(1 * args.hidden_dim, args.hidden_dim)
        self.mlp2 = MLP(args.layer * args.hidden_dim, args.hidden_dim)
        self.gnn1 = GCNEncoder(dataset_num_features, args.hidden_dim, args.layer, args.device)
        self.gnn2 = GCNEncoder(dataset_num_features, args.hidden_dim, args.layer, args.device)

    def forward(self, x, edge_index, diff_index, batch):
        gv1, lv1 = self.gnn1(x, edge_index, batch)
        gv2, lv2 = self.gnn2(x, diff_index, batch)

        lv1 = self.mlp1(lv1)
        lv2 = self.mlp1(lv2)

        gv1 = self.mlp2(gv1)
        gv2 = self.mlp2(gv2)

        return lv1, gv1, lv2, gv2


class MLP(nn.Module):
    def __init__(self, in_ft, out_ft):
        super(MLP, self).__init__()
        self.ffn = nn.Sequential(
            nn.Linear(in_ft, out_ft),
            nn.PReLU(),
            nn.Linear(out_ft, out_ft),
            nn.PReLU(),
            nn.Linear(out_ft, out_ft),
            nn.PReLU()
        )
        self.linear_shortcut = nn.Linear(in_ft, out_ft)

    def forward(self, x):
        return self.ffn(x) + self.linear_shortcut(x)


# Borrowed from https://github.com/fanyun-sun/InfoGraph
def get_positive_expectation(p_samples, measure, average=True):
    """Computes the positive part of a divergence / difference.
    Args:
        p_samples: Positive samples.
        measure: Measure to compute for.
        average: Average the result over samples.
    Returns:
        torch.Tensor
    """
    log_2 = np.log(2.)

    if measure == 'GAN':
        Ep = - F.softplus(-p_samples)
    elif measure == 'JSD':
        Ep = log_2 - F.softplus(- p_samples)
    elif measure == 'X2':
        Ep = p_samples ** 2
    elif measure == 'KL':
        Ep = p_samples + 1.
    elif measure == 'RKL':
        Ep = -torch.exp(-p_samples)
    elif measure == 'DV':
        Ep = p_samples
    elif measure == 'H2':
        Ep = 1. - torch.exp(-p_samples)
    elif measure == 'W1':
        Ep = p_samples

    if average:
        return Ep.mean()
    else:
        return Ep

# Borrowed from https://github.com/fanyun-sun/InfoGraph
def get_negative_expectation(q_samples, measure, average=True):
    """Computes the negative part of a divergence / difference.
    Args:
        q_samples: Negative samples.
        measure: Measure to compute for.
        average: Average the result over samples.
    Returns:
        torch.Tensor
    """
    log_2 = np.log(2.)

    if measure == 'GAN':
        Eq = F.softplus(-q_samples) + q_samples
    elif measure == 'JSD':
        Eq = F.softplus(-q_samples) + q_samples - log_2
    elif measure == 'X2':
        Eq = -0.5 * ((torch.sqrt(q_samples ** 2) + 1.) ** 2)
    elif measure == 'KL':
        Eq = torch.exp(q_samples)
    elif measure == 'RKL':
        Eq = q_samples - 1.
    elif measure == 'H2':
        Eq = torch.exp(q_samples) - 1.
    elif measure == 'W1':
        Eq = q_samples

    if average:
        return Eq.mean()
    else:
        return Eq

# Borrowed from https://github.com/fanyun-sun/InfoGraph
def local_global_loss_(l_enc, g_enc, batch, measure):#, mask):
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
    max_nodes = num_nodes // num_graphs

    pos_mask = torch.zeros((num_nodes, num_graphs)).cuda()
    neg_mask = torch.ones((num_nodes, num_graphs)).cuda()
    msk = torch.ones((num_nodes, num_graphs)).cuda()
    for nodeidx, graphidx in enumerate(batch):
        pos_mask[nodeidx][graphidx] = 1.
        neg_mask[nodeidx][graphidx] = 0.

    #for idx, m in enumerate(mask):
    #    msk[idx * max_nodes + m: idx * max_nodes + max_nodes, idx] = 0.

    res = torch.mm(l_enc, g_enc.t())# * msk

    E_pos = get_positive_expectation(res * pos_mask, measure, average=False).sum()
    E_pos = E_pos / num_nodes
    E_neg = get_negative_expectation(res * neg_mask, measure, average=False).sum()
    E_neg = E_neg / (num_nodes * (num_graphs - 1))
    
    return E_neg - E_pos


def global_global_loss_(g1_enc, g2_enc, measure):
    '''
    Args:
        l: Local feature map.
        g: Global features.
        measure: Type of f-divergence. For use with mode `fd`
        mode: Loss mode. Fenchel-dual `fd`, NCE `nce`, or Donsker-Vadadhan `dv`.
    Returns:
        torch.Tensor: Loss.
    '''
    num_graphs = g1_enc.shape[0]

    pos_mask = torch.zeros((num_graphs, num_graphs)).cuda()
    neg_mask = torch.ones((num_graphs, num_graphs)).cuda()
    for graphidx in range(num_graphs):
        pos_mask[graphidx][graphidx] = 1.
        neg_mask[graphidx][graphidx] = 0.

    res = torch.mm(g1_enc, g2_enc.t())

    E_pos = get_positive_expectation(res * pos_mask, measure, average=False).sum()
    E_pos = E_pos / num_graphs
    E_neg = get_negative_expectation(res * neg_mask, measure, average=False).sum()
    E_neg = E_neg / (num_graphs * (num_graphs - 1))
    return E_neg - E_pos