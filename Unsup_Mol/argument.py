import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--name", type=str, default="bace", help="Name of the dataset. \
                        Supported names TU: MUTAG, PTC_MR, IMDB-BINARY, REDDIT-BINRARY, ... \
                        Supported names Molecule : bace bbbp clintox tox21 sider toxcast")
    parser.add_argument("--embedder", type=str, default="GraphCL", choices=['GraphCL', 'InfoGraph', 'BGRL', 'GCA'])
    parser.add_argument("--Scratch", action="store_true", default=False)


    if 'GraphCL' in parser.parse_known_args()[0].embedder:
        parser.add_argument('--aug1', type=str, default = 'drop_node', choices=['none', 'drop_node', 'subgraph', 'permute_edge', 'mask_node'])
        parser.add_argument('--aug_ratio1', type=float, default = 0.2)
        parser.add_argument('--aug2', type=str, default = 'subgraph', choices=['none', 'drop_node', 'subgraph', 'permute_edge', 'mask_node'])
        parser.add_argument('--aug_ratio2', type=float, default = 0.2)

    if 'InfoGraph' in parser.parse_known_args()[0].embedder:
        parser.add_argument('--prior', dest='prior', action='store_const', const=True, default=False)


    if 'BGRL' in parser.parse_known_args()[0].embedder:
        parser.add_argument('--drop_feature_rate_1', type=float, default=0.2)
        parser.add_argument('--drop_feature_rate_2', type=float, default=0.2)
        parser.add_argument('--drop_edge_rate_1', type=float, default=0.2)
        parser.add_argument('--drop_edge_rate_2', type=float, default=0.2)
        parser.add_argument("--mad", type=float, default=0.9, help="Moving Average Decay for Teacher Network")

    if 'GCA' in parser.parse_known_args()[0].embedder:
        parser.add_argument('--drop_feature_rate_1', type=float, default=0.2)
        parser.add_argument('--drop_feature_rate_2', type=float, default=0.2)
        parser.add_argument('--drop_edge_rate_1', type=float, default=0.2)
        parser.add_argument('--drop_edge_rate_2', type=float, default=0.2)
        parser.add_argument('--drop_scheme', type=str, default='degree')

    parser.add_argument("--layer", type=int, default=3, help="The number of GNN layer")
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument('--lr_scale', type=float, default=1,
                        help='relative learning rate for the feature extraction layer (default: 1)')
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument('--emb_dim', type=int, default=512)
    parser.add_argument('--dropout_ratio', type=float, default=0.5)
    parser.add_argument("--device", type=int, default=3)
    
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--runseed", type=int, default=1)
    
    parser.add_argument('--split', type = str, default="random", help = "random or scaffold or random_scaffold")
    
    
    return parser.parse_known_args()