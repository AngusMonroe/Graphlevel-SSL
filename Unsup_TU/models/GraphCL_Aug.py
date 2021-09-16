from torch_geometric.data import DataLoader
import torch
import torch.nn as nn
from torch import optim
import os
from utils import *

from utils import config2string
from aug import TUDataset_aug as TUDataset
from aug import TUDataset_aug2 as TUDataset2
from evaluate_embedding import evaluate_embedding
from .Encoder import GINEncoder
from argument import parse_args
from utils import setup_seed
args, unknown = parse_args()
setup_seed(args.seed)

class GraphCL_Aug_Trainer():
    def __init__(self, args):
        self._args = args
        self._init()
        self.config_str = config2string(args)
        print("\n[Config] {}\n".format(self.config_str))
        

    def _init(self):
        
        args = self._args

        self._device = set_device(args.device)
        

        path = osp.join(osp.dirname(osp.dirname(osp.realpath(__file__))), '.', 'data', args.name)
        
        self._dataset = TUDataset2(path, name=args.name, aug=args.aug, aug2=args.aug2).shuffle()
        self._eval_dataset = TUDataset(path, name=args.name, aug='none').shuffle()

        self._loader = DataLoader(dataset=self._dataset, batch_size=args.batch)
        self._eval_loader = DataLoader(dataset=self._eval_dataset, batch_size=args.batch)
        print(f"Data: {self._dataset.data}")

        try:
            dataset_num_features = self._dataset.get_num_feature()
        except:
            dataset_num_features = 1

        self._model = GraphCL_Aug(dataset_num_features, args).to(self._device)
        print(self._model)

        self._optimizer = optim.Adam(params=self._model.parameters(), lr=args.lr)#, weight_decay= 1e-5)


    def experiment(self):
        
        args = self._args
        
        print("initial accuracy ")
        self._model.eval()
        emb, y = self._model.encoder.get_embeddings(self._eval_loader)
        init_acc = evaluate_embedding(emb, y)

        txtfile = f'./results/GraphCL_Aug/{args.name}.txt'

        with open(txtfile, "a") as f:
            f.write(self.config_str)
            f.write('\n')
            f.write("init --> | val_acc: {} | test_acc : {} |\n".format(init_acc[0], init_acc[1]))
            f.close()


        accuracies = {'val':[] ,'test':[]}
        # Start Model Training
        print("Training Start!")
        self._model.train()
        for epoch in range(1, self._args.epochs+1):
            loss_all = 0
            for data in self._loader:
                data, data_aug1, data_aug2 = data
                node_num = data.x.size(0)
                
                self._optimizer.zero_grad()
                #node_num, _ = data.x.size()
                
                
                if args.aug == 'dnodes' or args.aug == 'subgraph':
                    # node_num_aug, _ = data_aug.x.size()
                    edge_idx = data_aug1.edge_index.numpy()
                    _, edge_num = edge_idx.shape
                    idx_not_missing = [n for n in range(node_num) if (n in edge_idx[0] or n in edge_idx[1])]

                    node_num_aug = len(idx_not_missing)
                    data_aug1.x = data_aug1.x[idx_not_missing]

                    data_aug1.batch = data_aug1.batch[idx_not_missing]
                    idx_dict = {idx_not_missing[n]:n for n in range(node_num_aug)}
                    edge_idx = [[idx_dict[edge_idx[0, n]], idx_dict[edge_idx[1, n]]] for n in range(edge_num) if not edge_idx[0, n] == edge_idx[1, n]]
                    data_aug1.edge_index = torch.tensor(edge_idx).transpose_(0, 1)


                if args.aug2 == 'dnodes' or args.aug2 == 'subgraph':
                    # node_num_aug, _ = data_aug.x.size()
                    edge_idx = data_aug2.edge_index.numpy()
                    _, edge_num = edge_idx.shape
                    idx_not_missing = [n for n in range(node_num) if (n in edge_idx[0] or n in edge_idx[1])]

                    node_num_aug = len(idx_not_missing)
                    data_aug2.x = data_aug2.x[idx_not_missing]

                    data_aug2.batch = data_aug2.batch[idx_not_missing]
                    idx_dict = {idx_not_missing[n]:n for n in range(node_num_aug)}
                    edge_idx = [[idx_dict[edge_idx[0, n]], idx_dict[edge_idx[1, n]]] for n in range(edge_num) if not edge_idx[0, n] == edge_idx[1, n]]
                    data_aug2.edge_index = torch.tensor(edge_idx).transpose_(0, 1)


                data_aug1 = data_aug1.to(self._device)
                x_aug1 = self._model(data_aug1.x, data_aug1.edge_index, data_aug1.batch, data_aug1.num_graphs)


                data_aug2 = data_aug2.to(self._device)
                x_aug2 = self._model(data_aug2.x, data_aug2.edge_index, data_aug2.batch, data_aug2.num_graphs)

                loss = self._model.loss_cal(x_aug1, x_aug2)
                loss_all += loss.item() * data.num_graphs
                loss.backward()
                self._optimizer.step()
                
            st = '[{}][Epoch {}/{}] Loss: {:.4f} |'\
                .format(currentTime(), epoch, args.epochs, loss_all)
            print(st)

            if (epoch) % args.eval_freq == 0:
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


class GraphCL_Aug(nn.Module):
    def __init__(self, dataset_num_features, args, alpha=0.5, beta=1., gamma=.1, **kwargs):
        super(GraphCL_Aug, self).__init__()
        
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        self.embedding_dim = args.hidden_dim * args.layer
        self.encoder = GINEncoder(dataset_num_features, args.hidden_dim, args.layer, args.device)

        self.proj_head = nn.Sequential(nn.Linear(self.embedding_dim, self.embedding_dim), nn.ReLU(inplace=True), nn.Linear(self.embedding_dim, self.embedding_dim))

        self.init_emb()

    def init_emb(self):
        initrange = -1.5 / self.embedding_dim
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def forward(self, x, edge_index, batch, num_graphs):

        # batch_size = data.num_grap
        if x is None:
            x = torch.ones(batch.shape[0]).to(self.device)

        y, M = self.encoder(x, edge_index, batch)

        y = self.proj_head(y)

        return y

    def loss_cal(self, x, x_aug):

        eps = 1e-10

        if x.size(0) != x_aug.size(0):
            x = x[:x_aug.size(0)]

        x += eps
        x_aug += eps

        T = 0.2
        batch_size, _ = x.size()
        x_abs = x.norm(dim=1)
        x_aug_abs = x_aug.norm(dim=1)

        sim_matrix = torch.einsum('ik,jk->ij', x, x_aug) / torch.einsum('i,j->ij', x_abs, x_aug_abs)
        sim_matrix = torch.exp(sim_matrix / T)
        pos_sim = sim_matrix[range(batch_size), range(batch_size)]
        loss = pos_sim / ((sim_matrix.sum(dim=1) - pos_sim))
        loss = - torch.log(loss).mean()

        return loss

