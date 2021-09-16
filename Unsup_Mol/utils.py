from torch_geometric.utils import dropout_adj
import torch.nn.functional as F
from collections import Counter
import random
import os.path as osp
import os

import scipy.sparse as sp
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

import torch
import torch.nn as nn

from datetime import datetime

def currentTime():
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')

def to_sparse(x):
    """ converts dense tensor x to sparse format """
    x_typename = torch.typename(x).split('.')[-1]
    sparse_tensortype = getattr(torch.sparse, x_typename)

    indices = torch.nonzero(x)
    if len(indices.shape) == 0:  # if all elements are zeros
        return sparse_tensortype(*x.shape)
    indices = indices.t()
    values = x[tuple(indices[i] for i in range(indices.shape[0]))]
    return sparse_tensortype(indices, values, x.size())



def training_config(args):
    train_config = dict()
    for arg in vars(args):
        train_config[arg] = getattr(args,arg)
    return train_config

def setup_seed(seed):

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)


def create_dirs(dirs):
    for dir_tree in dirs:
        sub_dirs = dir_tree.split("/")
        path = ""
        for sub_dir in sub_dirs:
            path = osp.join(path, sub_dir)
            os.makedirs(path, exist_ok=True)


class EMA:
    def __init__(self, beta, epochs):
        super().__init__()
        self.beta = beta
        self.step = 0
        self.total_steps = epochs

    def update_average(self, old, new):
        if old is None:
            return new

        beta = 1 - (1 - self.beta) * (np.cos(np.pi * self.step / self.total_steps) + 1) / 2.0
        self.step += 1
        return old * beta + (1 - beta) * new

def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

def loss_fn(x, y):
     x = F.normalize(x, dim=-1, p=2)
     y = F.normalize(y, dim=-1, p=2)
     return 2 - 2 * (x * y).sum(dim=-1)


def set_device(d):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(d)
    device = f'cuda:{d}' if torch.cuda.is_available() else "cpu"
    torch.cuda.set_device(device)
    return device

def update_moving_average(ema_updater, ma_model, current_model):
    for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
        old_weight, up_weight = ma_params.data, current_params.data
        ma_params.data = ema_updater.update_average(old_weight, up_weight)


def set_requires_grad(model, val):
    for p in model.parameters():
        p.requires_grad = val


def enumerateConfig(args):
    args_names = []
    args_vals = []
    for arg in vars(args):
        args_names.append(arg)
        args_vals.append(getattr(args, arg))

    return args_names, args_vals

def config2string(args):
    args_names, args_vals = enumerateConfig(args)
    st = ''
    for name, val in zip(args_names, args_vals):
        st_ = "{}_({})_".format(name, val)
        st += st_

    return st[:-1]


def printConfig(args):
    args_names, args_vals = enumerateConfig(args)
    print(args_names)
    print(args_vals)

def repeat_1d_tensor(t, num_reps):
    return t.unsqueeze(1).expand(-1, num_reps)

def task(name):
    if name == "tox21":
        num_tasks = 12
    elif name == "hiv":
        num_tasks = 1
    elif name == "pcba":
        num_tasks = 128
    elif name == "muv":
        num_tasks = 17
    elif name == "bace":
        num_tasks = 1
    elif name == "bbbp":
        num_tasks = 1
    elif name == "toxcast":
        num_tasks = 617
    elif name == "sider":
        num_tasks = 27
    elif name == "clintox":
        num_tasks = 2
    elif name == 'esol':
        num_tasks = 1
    elif name == 'freesolv':
        num_tasks = 1
    elif name == 'mutag':
        num_tasks = 1
    else:
        raise ValueError("Invalid dataset name.")

    return num_tasks


def train_cls(args, model, device, loader, optimizer):
    model.train()
    criterion = nn.BCEWithLogitsLoss(reduction = "none")
    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)
        pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        y = batch.y.view(pred.shape).to(torch.float64)

        #Whether y is non-null or not.
        is_valid = y**2 > 0
        #Loss matrix
        loss_mat = criterion(pred.double(), (y+1)/2)
        #loss matrix after removing null target
        loss_mat = torch.where(is_valid, loss_mat, torch.zeros(loss_mat.shape).to(loss_mat.device).to(loss_mat.dtype))
            
        optimizer.zero_grad()
        loss = torch.sum(loss_mat)/torch.sum(is_valid)
        loss.backward()
        optimizer.step()

### evaluate mol classification

def eval_cls(args, model, device, loader):
    model.eval()
    y_true = []
    y_scores = []

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)

        with torch.no_grad():
            pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)

        y_true.append(batch.y.view(pred.shape))
        y_scores.append(pred)

    y_true = torch.cat(y_true, dim = 0).cpu().numpy()
    y_scores = torch.cat(y_scores, dim = 0).cpu().numpy()

    roc_list = []
    for i in range(y_true.shape[1]):
        #AUC is only defined when there is at least one positive data.
        if np.sum(y_true[:,i] == 1) > 0 and np.sum(y_true[:,i] == -1) > 0:
            is_valid = y_true[:,i]**2 > 0
            roc_list.append(roc_auc_score((y_true[is_valid,i] + 1)/2, y_scores[is_valid,i]))

    if len(roc_list) < y_true.shape[1]:
        print("Some target is missing!")
        print("Missing ratio: %f" %(1 - float(len(roc_list))/y_true.shape[1]))

    return sum(roc_list)/len(roc_list) #y_true.shape[1]




