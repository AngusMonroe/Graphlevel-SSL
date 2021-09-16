import argparse

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--name", type=str, default="MUTAG", help="Name of the dataset. \
                        Supported names TU: MUTAG, PTC_MR, NCI1, PROTEINS, ...")
    parser.add_argument("--embedder", type=str, default="MVGRL", choices=["InfoGraph", 'GraphCL', 'GraphCL_Aug', 'BGRL', 'GCA', 'MVGRL'])
    parser.add_argument("--seed", type=int, default=1)
    
    if 'GraphCL' in parser.parse_known_args()[0].embedder:
        parser.add_argument('--aug', default='dnodes', choices=['subgraph', 'dnodes', 'pedges', 'mask_nodes'])
        
    if 'GraphCL_Aug' in parser.parse_known_args()[0].embedder:
        parser.add_argument('--aug2', default='dnodes', choices=['subgraph', 'dnodes', 'pedges', 'mask_nodes'])

    if 'InfoGraph' in parser.parse_known_args()[0].embedder:
        parser.add_argument('--prior', dest='prior', action='store_const', 
                const=True, default=False)

    if 'GCA' in parser.parse_known_args()[0].embedder:
        parser.add_argument('--drop_feature_rate_1', type=float, default=0.2)
        parser.add_argument('--drop_feature_rate_2', type=float, default=0.2)
        parser.add_argument('--drop_edge_rate_1', type=float, default=0.2)
        parser.add_argument('--drop_edge_rate_2', type=float, default=0.2)
        parser.add_argument('--drop_scheme', type=str, default='degree')

    if 'BGRL' in parser.parse_known_args()[0].embedder:
        parser.add_argument('--drop_feature_rate_1', type=float, default=0.2)
        parser.add_argument('--drop_feature_rate_2', type=float, default=0.2)
        parser.add_argument('--drop_edge_rate_1', type=float, default=0.2)
        parser.add_argument('--drop_edge_rate_2', type=float, default=0.2)
        parser.add_argument("--mad", type=float, default=0.9, help="Moving Average Decay for Teacher Network")

    
    parser.add_argument("--layer", type=int, default=4, help="The number of GNN layer")
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--eval_freq", type=float, default=20)
    parser.add_argument("--device", type=int, default=3)

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument('--hidden-dim', dest='hidden_dim', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default = 4, help='number of workers for dataset loading')
     
        
    return parser.parse_known_args()