import argparse
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import util
from mainCNF import get_datasets, save_args
from cnf_args import DATASETS
import GradualSelfTrain as G

# +
# name: (function, default parameter)
BASELINES = {'sourceonly': (G.SourceOnly, None),
             'gst': (G.GradualSelfTrain, None),
             'goat': (G.GOAT, 3),
             'gift': (G.GIFT, 10),
             'sgift': (G.SequentialGIFT, 10),
             'aux': (G.AuxSelfTrain, 10),
             'saux': (G.SequentialAuxSelfTrain, 10)}


def get_parser():    
    parser = argparse.ArgumentParser(description='baseline methods')
    parser.add_argument("--dataset", type=str, default="moon", choices=DATASETS)
    parser.add_argument("--n_dim", type=int, default=None, help="the number of dimension of dataset")
    parser.add_argument("--method", type=str, default='sourceonly', choices=BASELINES.keys())
    parser.add_argument("--file_name", type=str, required=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--no_inter", type=eval, default=False)
    parser.add_argument("--source_only", type=eval, default=False)
    parser.add_argument("--label_ratio", type=str, default="100-0-0", help="ratio of labeled samples of each domain")
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument("--hidden_dim", type=float, default=64)
    parser.add_argument("--hyper_param", type=int, default=None)
    return parser


# -

if __name__ == '__main__':
    # set args
    parser = get_parser()
    if 'get_ipython' in globals():
        # jupyter notebook env, for debug
        args = parser.parse_args(["--file_name", "debug", "--epochs", "5", "--method", "aux", "--hyper_param", "10"])
    else:
        args = parser.parse_args()

    # load data
    x_all, y_all = get_datasets(args)    

    # load function of baseline method 
    function, hyper_param = BASELINES[args.method]
    if args.hyper_param is None: args.hyper_param = hyper_param

    # save args
    file_name = save_args(args)

    # start training
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.hyper_param is None:
        all_model, _ = function(x_all, y_all, hidden_dim=args.hidden_dim, n_epochs=args.epochs, weight_decay=args.weight_decay)
    else:
        all_model, _ = function(x_all, y_all, args.hyper_param, hidden_dim=args.hidden_dim, n_epochs=args.epochs, weight_decay=args.weight_decay)
    torch.save(all_model[-1].state_dict(), f'./result/state_{file_name}.tar')



