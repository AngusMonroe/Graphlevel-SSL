from torch_geometric.data import DataLoader
import torch
import torch.nn as nn
from torch import optim

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

class BGRL_Trainer():
    def __init__(self, args):
        #embedder.__init__(self, args)
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

        self._model = BGRL(dataset_num_features, args).to(self._device)
        print(self._model)

        self._optimizer = optim.Adam(params=self._model.parameters(), lr=args.lr)


    def experiment(self):
        # get Random Initial accuracy
        
        args = self._args

        print("initial accuracy ")
        self._model.eval()
        emb, y = self._model.student_encoder.get_embeddings(self._eval_loader)
        init_acc = evaluate_embedding(emb, y)

        txtfile = f'./results/BGRL/{args.name}.txt'

        with open(txtfile, "a") as f:
            f.write(self.config_str)
            f.write('\n')
            f.write("init --> | val_acc: {} | test_acc : {} |\n".format(init_acc[0], init_acc[1]))
            f.close()


        accuracies = {'val':[] ,'test':[]}
        for epoch in range(1, args.epochs+1):
            self._model.train()
            for data in self._loader:
                data = data.to(self._device)
                                
                augmentation = Augmentation(float(args.drop_feature_rate_1),float(args.drop_feature_rate_2),float(args.drop_edge_rate_1),float(args.drop_edge_rate_2))
                view1, view2 = augmentation._feature_masking(data, self._device)

                loss = self._model(x1=view1.x, x2=view2.x, edge_index_v1=view1.edge_index, edge_index_v2=view2.edge_index, batch = view1.batch)

                self._optimizer.zero_grad()
                loss.backward()
                self._optimizer.step()
                self._model.update_moving_average()

                sys.stdout.write('\rEpoch {}/{}, loss {:.4f}'.format(epoch, args.epochs, loss))
                sys.stdout.flush()
               
            if (epoch) % args.eval_freq == 0:
                self._model.eval()
                emb, y = self._model.student_encoder.get_embeddings(self._eval_loader)
                
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



class BGRL(nn.Module):

    def __init__(self, dataset_num_features, args, moving_average_decay = 0.99, **kwargs):
        super().__init__()
        self.student_encoder = GINEncoder(dataset_num_features, args.hidden_dim, args.layer, args.device)
        self.teacher_encoder = copy.deepcopy(self.student_encoder)
        set_requires_grad(self.teacher_encoder, False)
        self.teacher_ema_updater = EMA(moving_average_decay, args.epochs)
        
        self.embedding_dim = args.hidden_dim * args.layer
        self.student_predictor = nn.Sequential(nn.Linear(self.embedding_dim, self.embedding_dim*2), nn.BatchNorm1d(self.embedding_dim*2, momentum = 0.01), nn.PReLU(), nn.Linear(self.embedding_dim*2, self.embedding_dim))
        self.student_predictor.apply(init_weights)
    
    def reset_moving_average(self):
        del self.teacher_encoder
        self.teacher_encoder = None

    def update_moving_average(self):
        assert self.teacher_encoder is not None, 'teacher encoder has not been created yet'
        update_moving_average(self.teacher_ema_updater, self.teacher_encoder, self.student_encoder)

    def forward(self, x1, x2, edge_index_v1, edge_index_v2, batch):
        v1_graph, v1_student = self.student_encoder(x=x1, edge_index=edge_index_v1, batch = batch)
        v2_graph, v2_student = self.student_encoder(x=x2, edge_index=edge_index_v2, batch = batch)

        v1_pred = self.student_predictor(v1_student)
        v2_pred = self.student_predictor(v2_student)
        
        with torch.no_grad():
            _, v1_teacher = self.teacher_encoder(x=x1, edge_index=edge_index_v1, batch = batch)
            _, v2_teacher = self.teacher_encoder(x=x2, edge_index=edge_index_v2, batch = batch)
            
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