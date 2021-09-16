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


class GraphCL_Trainer():
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
        self._dataset = MoleculeDataset(path, dataset=args.name)
        self._dataset.aug = 'none'
        dataset1 = self._dataset.shuffle()
        dataset2 = deepcopy(dataset1)
        dataset1.aug, dataset1.aug_ratio = args.aug1, args.aug_ratio1
        dataset2.aug, dataset2.aug_ratio = args.aug2, args.aug_ratio2

        self._loader1 = DataLoader(dataset=dataset1, batch_size=args.batch, shuffle=False)  # [self._dataset.data]
        self._loader2 = DataLoader(dataset=dataset2, batch_size=args.batch, shuffle=False)  # [self._dataset.data]
        
        self.num_tasks = task(args.name)
        
        self.gnn = GNN(args.layer, args.emb_dim, device=args.device, JK = "last", drop_ratio = args.dropout_ratio, gnn_type = 'gin')

        self._model = GraphCL(self.gnn, args)
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

        # if args.semi_ratio != 1.0:
        #     n_total, n_sample = len(train_dataset), int(len(train_dataset)*args.semi_ratio)
        #     print('sample {:.2f} = {:d} labels for semi-supervised training!'.format(args.semi_ratio, n_sample))
        #     all_idx = list(range(n_total))
        #     random.seed(0)
        #     idx_semi = random.sample(all_idx, n_sample)
        #     train_dataset = train_dataset[torch.tensor(idx_semi)] #int(len(train_dataset)*args.semi_ratio)
        #     print('new train dataset size:', len(train_dataset))
        # else:
        #     print('finetune using all data!')


        self._train_loader = DataLoader(train_dataset, batch_size=args.batch, shuffle=True)#, num_workers = args.num_workers)
        self._val_loader = DataLoader(valid_dataset, batch_size=args.batch, shuffle=False)#, num_workers = args.num_workers)
        self._test_loader = DataLoader(test_dataset, batch_size=args.batch, shuffle=False)#, num_workers = args.num_workers)

        self._eval_model = GNN_graphpred(args.layer, args.emb_dim, self.num_tasks, args, drop_ratio = args.dropout_ratio)
        # filename = 'graphcl'
        # self.pretrained_path = './tmp/'+filename+'.pth'

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
                train_acc_accum = 0
                train_loss_accum = 0
                for step, batch in enumerate(tqdm(zip(self._loader1, self._loader2), desc="Iteration")):
                    batch1, batch2 = batch
                    batch1 = batch1.to(self._device)
                    batch2 = batch2.to(self._device)

                    self._optimizer.zero_grad()
                    
                    x1 = self._model.forward_cl(batch1.x, batch1.edge_index, batch1.edge_attr, batch1.batch)
                    x2 = self._model.forward_cl(batch2.x, batch2.edge_index, batch2.edge_attr, batch2.batch)
                    loss = self._model.loss_cl(x1, x2)

                    loss.backward()
                    self._optimizer.step()

                    train_loss_accum += float(loss.detach().cpu().item())

                st = '[{}][Epoch {}/{}] Loss: {:.4f} |'\
                    .format(currentTime(), epoch, args.epochs, train_loss_accum)
                print(st)

            #self._eval_model = GNN_graphpred(args.layer, args.emb_dim, self.num_tasks, args, drop_ratio = args.dropout_ratio)
            self._eval_model.gnn = self.gnn
            
            
            #torch.save(self.gnn.state_dict(), self.pretrained_path)
            #self._eval_model = GNN_graphpred(args.layer, args.emb_dim, self.num_tasks, args, drop_ratio = args.dropout_ratio)
            #self._eval_model.from_pretrained(self.pretrained_path)
        
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

        # if linear protocol, fix GNN layers
        #if args.protocol == 'linear':
        #    print("linear protocol, only train the top layer!")
        #    for name, param in self._eval_model.named_parameters():
        #        if not 'pred_linear' in name:
        #            param.requires_grad = False
        #elif args.protocol == 'nonlinear':
        #    print("finetune protocol, train all the layers!")
        #else:
        #    print("invalid protocol!")


        
        
        #if args.semi_ratio == 1.0:
        #    print('full-supervised {:.2f}'.format(args.semi_ratio))
        #else:
        #    print('semi-supervised {:.2f}'.format(args.semi_ratio))
        
        #print('Protocol: ', args.protocol)


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

        txtfile = f'./results/GraphCL/{args.name}.txt'

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
            

        #with open('./logs/GraphCL/best_{}.txt'.format(args.name), 'a+') as f:
        #    f.write('val_acc : {} | test_acc : {}'.format(val_acc_list[-1], test_acc_list[-1]))
        #    f.write('seed : {} | split : {} | best_layer : {} | best_lr : {} | best_epoch : {} | best_aug1 : {} | best_aug2 : {} |\n'
        #                                        .format(args.seed, args.split, args.layer, args.lr, args.epochs, args.aug1, args.aug2))
        #    f.write('\n')


class GraphCL(nn.Module):
    def __init__(self, gnn, args):
        super(GraphCL, self).__init__()
        self.encoder = gnn
        self.pool = global_mean_pool
        
        self.projection_head = nn.Sequential(nn.Linear(args.emb_dim, args.emb_dim), nn.ReLU(inplace=True), nn.Linear(args.emb_dim, args.emb_dim))
        # pred_dim 사용불가(ByoW랑 비교 X)

    def forward_cl(self, x, edge_index, edge_attr, batch):
        x = self.encoder(x, edge_index, edge_attr)
        x = self.pool(x, batch)
        x = self.projection_head(x)
        return x
    
    # GraphCL loss nan
    def loss_cl(self, x1, x2):
        eps = 1e-5
        T = 0.1
        batch_size, _ = x1.size()
        x1_abs = x1.norm(dim=1)
        x2_abs = x2.norm(dim=1)
        
        sim_matrix = torch.einsum('ik,jk->ij', x1, x2) / torch.einsum('i,j->ij', x1_abs, x2_abs)
        sim_matrix = torch.exp(sim_matrix / T)
        
        #if torch.isnan(sim_matrix).sum() !=0:
        #    import pdb
        #    pdb.set_trace()
        
        pos_sim = sim_matrix[range(batch_size), range(batch_size)]

        #if ((sim_matrix.sum(dim=1) - pos_sim) == 0).sum() != 0:
        #    import pdb
        #    pdb.set_trace()

        loss = pos_sim / (sim_matrix.sum(dim=1) - pos_sim + eps )  
        
        loss = - torch.log(loss).mean()

        #if torch.isnan(loss).sum() !=0:
        #    import pdb
        #    pdb.set_trace()

        return loss
