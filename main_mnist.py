from __future__ import print_function, absolute_import

import argparse

from model_mnist import ModelBaseline, ModelADA, ModelMEADA


def main():
    train_arg_parser = argparse.ArgumentParser(description="parser")
    train_arg_parser.add_argument("--seed", type=int, default=1,
                                  help="")
    train_arg_parser.add_argument('--algorithm', type=str, default='MEADA', choices=['ERM', 'ADA', 'MEADA'],
                                  help='Choose algorithm.')
    train_arg_parser.add_argument("--test_every", type=int, default=50,
                                  help="")
    train_arg_parser.add_argument("--batch_size", type=int, default=32,
                                  help="")
    train_arg_parser.add_argument("--num_classes", type=int, default=10,
                                  help="")
    train_arg_parser.add_argument("--step_size", type=int, default=1,
                                  help="")
    train_arg_parser.add_argument("--loops_train", type=int, default=100000,
                                  help="")
    train_arg_parser.add_argument("--loops_min", type=int, default=100,
                                  help="")
    train_arg_parser.add_argument("--loops_adv", type=int, default=15,
                                  help="")
    train_arg_parser.add_argument("--seen_index", type=int, default=0,
                                  help="")
    train_arg_parser.add_argument("--lr", type=float, default=0.0001,
                                  help='')
    train_arg_parser.add_argument("--lr_max", type=float, default=1.0,
                                  help='')
    train_arg_parser.add_argument("--weight_decay", type=float, default=0.0,
                                  help='')
    train_arg_parser.add_argument("--logs", type=str, default='logs/',
                                  help='')
    train_arg_parser.add_argument("--model_path", type=str, default='',
                                  help='')
    train_arg_parser.add_argument("--deterministic", type=bool, default=False,
                                  help='')
    train_arg_parser.add_argument("--k", type=int, default=5,
                                  help="")
    train_arg_parser.add_argument("--gamma", type=float, default=1.0,
                                  help="")
    train_arg_parser.add_argument("--eta", type=float, default=1.0,
                                  help="")
    args = train_arg_parser.parse_args()

    if args.algorithm == 'ERM':
        model_obj = ModelBaseline(flags=args)
    elif args.algorithm == 'ADA':
        model_obj = ModelADA(flags=args)
    elif args.algorithm == 'MEADA':
        model_obj = ModelMEADA(flags=args)
    else:
        raise RuntimeError
    model_obj.train(flags=args)


if __name__ == "__main__":
    main()
