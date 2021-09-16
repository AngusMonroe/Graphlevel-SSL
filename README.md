# Graphlevel-SSL

### Overview
Apply Graph Self-Supervised Learning methods to graph-level task(TUDataset, MolculeNet Dataset).
It is unified framework to compare state-of-the art graph-level self-supervised learning method with two well-known dataset(TUDataset, MoleculeNet Dataset).
I only focused on linear protocol which is two step method

1) pretrain with self-supervied method
2) freeze encoder and only train classifier 

### Reference
I adopt various official codes to unify below methods.
I expect that it can ensure very fair comparision.

- InfoGraph : https://github.com/fanyun-sun/InfoGraph
- GraphCL : https://github.com/Shen-Lab/GraphCL
- MVGRL : https://github.com/kavehhassani/mvgrl
- GCA : https://github.com/CRIPAC-DIG/GCA
- BGRL(not official code) : https://github.com/Namkyeong/BGRL_Pytorch
