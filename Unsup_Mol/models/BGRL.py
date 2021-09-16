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
from utils import *

class BGRL_Trainer():
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
        
        self._loader = DataLoader(dataset=dataset, batch_size=args.batch, shuffle=False)  # [self._dataset.data]
        
        self.num_tasks = task(args.name)
        
        self.gnn = GNN(args.layer, args.emb_dim, device=args.device, JK = "last", drop_ratio = args.dropout_ratio, gnn_type = 'gin')

        self._model = BGRL(self.gnn, args)
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
            for epoch in range(1, args.epochs+1):
                train_acc_accum = 0
                train_loss_accum = 0
                for step, batch in enumerate(tqdm(self._loader, desc="Iteration")):
                    
                    batch = batch.to(self._device)
                    self._optimizer.zero_grad()

                    augmentation = Augmentation(float(args.drop_feature_rate_1),float(args.drop_feature_rate_2),float(args.drop_edge_rate_1),float(args.drop_edge_rate_2))
                    view1, view2 = augmentation._feature_masking(batch, self._device)

                    loss = self._model(x1=view1.x, x2=view2.x, edge_index_v1=view1.edge_index, edge_index_v2=view2.edge_index, edge_attr1=view1.edge_attr, edge_attr2=view2.edge_attr, batch = view1.batch)

                    loss.backward()
                    self._optimizer.step()
                    self._model.update_moving_average()

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

        txtfile = f'./results/BGRL/{args.name}.txt'

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
            


class BGRL(nn.Module):

    def __init__(self, gnn, args):
        super(BGRL, self).__init__()
        
        self.student_encoder = gnn
        self.teacher_encoder = deepcopy(self.student_encoder)
        set_requires_grad(self.teacher_encoder, False)
        self.teacher_ema_updater = EMA(args.mad, args.epochs)
        
        self.student_predictor = nn.Sequential(nn.Linear(args.emb_dim, args.emb_dim), nn.ReLU(inplace=True), nn.Linear(args.emb_dim, args.emb_dim))
        self.student_predictor.apply(init_weights)

        self.device = args.device

        self.init_emb()

    def init_emb(self):
        #initrange = -1.5 / args.emb_dim
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)
        
    def reset_moving_average(self):
        del self.teacher_encoder
        self.teacher_encoder = None

    def update_moving_average(self):
        assert self.teacher_encoder is not None, 'teacher encoder has not been created yet'
        update_moving_average(self.teacher_ema_updater, self.teacher_encoder, self.student_encoder)
    

    def forward(self, x1, x2, edge_index_v1, edge_index_v2, edge_attr1, edge_attr2, batch):
        v1_student = self.student_encoder(x1, edge_index_v1, edge_attr1)
        v2_student = self.student_encoder(x2, edge_index_v2, edge_attr2)

        v1_pred = self.student_predictor(v1_student)
        v2_pred = self.student_predictor(v2_student)
        
        with torch.no_grad():
            v1_teacher = self.teacher_encoder(x1, edge_index_v1, edge_attr1)
            v2_teacher = self.teacher_encoder(x2, edge_index_v2, edge_attr2)
            
        loss1 = loss_fn(v1_pred, v2_teacher.detach())
        loss2 = loss_fn(v2_pred, v1_teacher.detach())

        loss = loss1 + loss2
        return loss.mean()





class Augmentation:

    def __init__(self, p_f1 = 0.2, p_f2 = 0.1, p_e1 = 0.2, p_e2 = 0.3):
        """
        two simple graph augmentation functions --> "Node feature masking" and "Edge masking"
        Random binary node feature mask following Bernoulli distribution with parameter p_f
        Random binary edge mask following Bernoulli distribution with parameter p_e
        """
        self.p_f1 = p_f1
        self.p_f2 = p_f2
        self.p_e1 = p_e1
        self.p_e2 = p_e2
        self.method = "BGRL"
    
    def _feature_masking(self, data, device):

        if data.x == None :
            edge_index1, edge_attr1 = dropout_adj(data.edge_index, data.edge_attr, p = self.p_e1)
            edge_index2, edge_attr2 = dropout_adj(data.edge_index, data.edge_attr, p = self.p_e2)

            new_data1, new_data2 = data.clone(), data.clone()
            new_data1.edge_index, new_data2.edge_index = edge_index1, edge_index2
            new_data1.batch = data.batch
        
        else :
            feat_mask1 = torch.FloatTensor(data.x.shape[1]).uniform_() > self.p_f1
            feat_mask2 = torch.FloatTensor(data.x.shape[1]).uniform_() > self.p_f2
            feat_mask1, feat_mask2 = feat_mask1.to(device), feat_mask2.to(device)
            x1, x2 = data.x.clone(), data.x.clone()
            x1, x2 = x1 * feat_mask1, x2 * feat_mask2

            edge_index1, edge_attr1 = dropout_adj(data.edge_index, data.edge_attr, p = self.p_e1)
            edge_index2, edge_attr2 = dropout_adj(data.edge_index, data.edge_attr, p = self.p_e2)

            new_data1, new_data2 = data.clone(), data.clone()
            new_data1.x, new_data2.x = x1, x2
            new_data1.edge_index, new_data2.edge_index = edge_index1, edge_index2
            new_data1.edge_attr , new_data2.edge_attr = edge_attr1, edge_attr2
            new_data1.batch = data.batch

        return new_data1, new_data2

    def __call__(self, data):
        
        return self._feature_masking(data)