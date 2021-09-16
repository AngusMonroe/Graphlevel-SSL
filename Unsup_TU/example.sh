#!/bin/bash

device=0

name=MUTAG
lr=0.001

pf=0.1
pe=0.3

aug=pedges
aug2=mask_nodes

python main.py --embedder BGRL --layer 4 --hidden-dim 128 --name $name --lr $lr --device $device --drop_feature_rate_1 $pf --drop_feature_rate_2 $pf --drop_edge_rate_1 $pe --drop_edge_rate_2 $pe
python main.py --embedder GCA --layer 4 --hidden-dim 128 --name $name --lr $lr --device $device --drop_feature_rate_1 $pf --drop_feature_rate_2 $pf --drop_edge_rate_1 $pe --drop_edge_rate_2 $pe
python main.py --embedder InfoGraph --layer 4 --hidden-dim 128 --name $name --lr $lr --device $device
python main.py --embedder MVGRL --layer 4 --hidden-dim 128 --name $name --lr $lr --device $device
python main.py --embedder GraphCL --layer 4 --hidden-dim 128 --name $name --aug $aug --lr $lr --device $device
python main.py --embedder GraphCL_Aug --layer 4 --hidden-dim 128 --name $name --aug $aug --aug2 $aug2 --lr $lr --device $device