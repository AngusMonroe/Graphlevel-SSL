from argument import parse_args

def main():
    args, unknown = parse_args()

    if args.embedder == 'InfoGraph':
        from models import InfoGraph_Trainer
        embedder = InfoGraph_Trainer(args)

    if args.embedder == 'GraphCL':
        from models import GraphCL_Trainer
        embedder = GraphCL_Trainer(args)

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