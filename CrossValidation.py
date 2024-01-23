import random
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
import ffjord as C
import util
from mainCNF import parser, get_datasets, get_priors, get_DataLoaders, train

if __name__ == '__main__':

    if 'get_ipython' in globals():
        # jupyter notebook env, for debug
        args = parser.parse_args(["--file_name", "debug", "--epochs", "5"])
    else:
        args = parser.parse_args()
        
    x_all, y_all = get_datasets(args)
    # check args
    _ = [print(f'{arg}: {key}') for arg, key in args._get_kwargs()]
    
    # start cross-validation
    result = []
    loss_all = []
    skf = StratifiedKFold(n_splits=3)
    xs, ys = x_all[0].copy(), y_all[0].copy()

    for train_index, test_index in skf.split(xs, ys):
        x_all[0], y_all[0] = xs[train_index,:].copy(), ys[train_index].copy()

        train_loaders = get_DataLoaders(args, x_all, y_all)
        priors = get_priors(args, x_all)

        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

        # build model
        cnf = C.build_model_tabular(args, args.n_dim, None)
        optimizer = optim.Adam(cnf.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        cnf = util.torch_to(cnf)
        loss_history = np.zeros(shape=(len(x_all), args.epochs))
        min_loss = np.full(shape=len(x_all), fill_value=np.inf)
        opt_param = cnf.state_dict()

        # Train
        for e in tqdm(range(args.epochs)):
            C.update_lr(e, optimizer, args.lr_change)
            cnf, optimizer, loss_epoch = train(args, cnf, optimizer, train_loaders, priors)
            loss_history[:,e] = loss_epoch
            # update min_loss and save model
            if np.all(min_loss > loss_history[:,e]):
                min_loss = loss_history[:,e].copy()
                opt_param = cnf.state_dict()

        # Eval
        cnf.load_state_dict(opt_param)
        s_acc = C.predict_target(cnf, priors[0], xs[test_index,:], ys[test_index], 0)[-1]
        result.append(s_acc)
        loss_all.append(loss_history)
        obj = (args, result, loss_all)
        pd.to_pickle(obj, f'./result/cv_{args.file_name}.pkl')
