#!/bin/bash

name=bace
device=0
runseed=1

emb_dim=512
dropout=0.5

aug1=mask_node
aug2=permute_edge

df1=0.1
df2=0.3
de1=0.1
de2=0.5

python main.py --embedder InfoGraph --name $name --runseed $runseed --device $device --emb_dim $emb_dim --dropout_ratio $dropout
python main.py --embedder GraphCL --name $name --runseed $runseed --device $device --aug1 $aug1 --aug2 $aug2 --emb_dim $emb_dim --dropout_ratio $dropout
python main.py --embedder GCA --name $name --runseed $runseed --device $device --emb_dim $emb_dim --drop_ratio $dropout --drop_feature_rate_1 $df1 --drop_feature_rate_2 $df2 --drop_edge_rate_1 $de1 --drop_edge_rate_2 $de2
python main.py --embedder BGRL --name $name --runseed $runseed --device $device --emb_dim $emb_dim --dropout_ratio $dropout --drop_feature_rate_1 $df1 --drop_feature_rate_2 $df2 --drop_edge_rate_1 $de1 --drop_edge_rate_2 $de2

