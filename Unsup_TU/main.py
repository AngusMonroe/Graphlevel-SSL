from argument import parse_args
from utils import setup_seed
args, unknown = parse_args()
setup_seed(args.seed)

def main():

    args, unknown = parse_args()
    setup_seed(args.seed)
    
    if args.embedder == 'InfoGraph':
        from models import InfoGraph_Trainer
        embedder = InfoGraph_Trainer(args)

    if args.embedder == 'MVGRL':
        from models import MVGRL_Trainer
        embedder = MVGRL_Trainer(args)

    if args.embedder == 'GraphCL':
        from models import GraphCL_Trainer
        embedder = GraphCL_Trainer(args)

    if args.embedder == 'GraphCL_Aug':
        from models import GraphCL_Aug_Trainer
        embedder = GraphCL_Aug_Trainer(args)

    if args.embedder == 'GCA':
        from models import GCA_Trainer
        embedder = GCA_Trainer(args)

    if args.embedder == 'BGRL':
        from models import BGRL_Trainer
        embedder = BGRL_Trainer(args)

    
    embedder.experiment()
    # embedder.writer.close()

if __name__ == "__main__":
    main()