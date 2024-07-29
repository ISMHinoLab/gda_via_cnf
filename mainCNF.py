import random
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import ffjord as C
from cnf_args import parser
import util
from Distributions import GaussianMixtureDA, GaussianMixtureEM, Gaussian, knnDistribution

# dataset name: n_dim, n_class, 
settings = {
            'moon': (2, 2),
            'scaled_moon': (2, 2),
            'spiral': (2, 2),
            'block': (2, 5),
            'gaussian': (2, 2),
            'mnist': (4, 10), # mean_r 10, k=15
            'mnist_vae': (4, 10),
            'mnist_dense': (4, 10),
            'portraits': (4, 2), # mean_r 3, k=5
            'tox21a': (4, 2), # mean_r 3, k=5
            'tox21b': (4, 2), # mean_r 3, k=5
            'tox21c': (4, 2), # mean_r 3, k=5
            'rxrx1': (4, 4), # mean_r 10, k=10
            'shift15m': (4, 7), # mean_r 10, k=15
           }


# +
def get_datasets(args, return_eval=False):
    # load dataset
    n_dim, args.n_class = settings[args.dataset]
    if args.n_dim is None: args.n_dim = n_dim 
    x_all, y_all = pd.read_pickle(f'./data/data_{args.dataset}.pkl')[args.n_dim]
    x_eval, y_eval = x_all.pop(), y_all.pop()  # remove evaluation data
    # ablation study
    if args.dataset == 'mnist_dense':
        given_domain = [0, args.inter_index, 28]
        x_all = [x_all[i].copy() for i in given_domain]
        y_all = [y_all[i].copy() for i in given_domain]
    elif args.dataset == 'gaussian':
        given_domain = [0, args.inter_index, 4]
        x_all = [x_all[i].copy() for i in given_domain]
        y_all = [y_all[i].copy() for i in given_domain]
    # unsupervised or semi-supervised
    y_all = util.mask_labels(y_all, args.label_ratio, args.seed)
    # ablation study
    if args.no_inter:
        x_all = [x_all[0].copy(), x_all[-1].copy()]
        y_all = [y_all[0].copy(), y_all[-1].copy()]
    if args.source_only:
        x_all = [x_all[0].copy()]
        y_all = [y_all[0].copy()]
    if return_eval:
        return x_all, y_all, x_eval, y_eval
    else:
        return x_all, y_all


def get_DataLoaders(args, x_all, y_all):
    train_loaders = []
    for x, y in zip(x_all, y_all):
        dataset = util.preprocess_input(x, y)
        if args.batch_size is None:
            batch_size = y.size
        else:
            batch_size = args.batch_size
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        train_loaders.append(train_loader)
    return train_loaders


def get_priors(args, x_all):
    # set prior of base distribution
    if args.base_distribution == 'normal':
        priors = [Gaussian(args.n_dim, args.seed,)]
    elif args.base_distribution == 'gmm':
        priors = [GaussianMixtureDA(args.n_dim, args.n_class, args.seed, args.mean_r)]
    # set prior of each domain's distribution
    # for i in range(num_domain-1):
    for i in range(len(x_all) - 1):
        if args.log_prob_method == 'gmm':
            p = GaussianMixtureEM(x_all[i], args.log_prob_param[i], args.seed)
        elif args.log_prob_method == 'knn':
            p = knnDistribution(x_all[i], args.log_prob_param[i], args.seed)
        priors.append(p)
    return priors


def save_args(args):
    file_name = f'{args.file_name}_{args.seed}'
    pd.to_pickle(args, f'./result/args_{file_name}.pkl')
    _ = [print(f'{arg}: {key}') for arg, key in args._get_kwargs()]
    return file_name
    
    
def train(args, cnf, optimizer, train_loaders, priors):
    loss_epoch = np.zeros(len(train_loaders))
    for batch in zip(*train_loaders):
        # for j in range(num_domain-1, -1, -1):
        for j in reversed(range(len(train_loaders))):  # j: 2 -> 1 -> 0
            x, y = util.torch_to(*batch[j])
            t0 = j
            while t0 > -1:
                x = x if t0 == j else z.detach().to(x)
                delta_logp = torch.zeros(x.shape[0], 1).to(x) if t0 == j else delta_logp.detach().to(x)
                lec = None if (args.poly_coef is None or not cnf.training) else torch.tensor(0.0).to(_x)
                time = torch.tensor([t0, t0+1]).to(x)
                z, delta_logp, lec = C.forward_with_integration_times(cnf, x, delta_logp, lec, time)
                loss = priors[t0].calc_loss(z, delta_logp, y)
                if j == 0: loss *= args.source_coef
                if args.poly_coef is not None: loss += args.poly_coef * lec
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_epoch[j] += loss.item()
                t0 -= 1
    return cnf, optimizer, loss_epoch


# -

if __name__ == '__main__':

    if 'get_ipython' in globals():
        # jupyter notebook env, for debug
        args = parser.parse_args(["--file_name", "debug", "--dataset", "moon", "--epochs", "3"])
    else:
        args = parser.parse_args()

    x_all, y_all = get_datasets(args)
    train_loaders = get_DataLoaders(args, x_all, y_all)
    priors = get_priors(args, x_all)
    file_name = save_args(args)
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    cnf = C.build_model_tabular(args, args.n_dim, None)
    optimizer = optim.Adam(cnf.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    cnf = util.torch_to(cnf)
    loss_history = np.zeros(shape=(len(x_all), args.epochs))
    min_loss = np.full(shape=len(x_all), fill_value=np.inf)

    for e in tqdm(range(args.epochs)):
        C.update_lr(e, optimizer, args.lr_change)
        cnf, optimizer, loss_epoch = train(args, cnf, optimizer, train_loaders, priors)
        loss_history[:,e] = loss_epoch
        pd.to_pickle(loss_history, f'./result/lh_{file_name}.pkl')
        # update min_loss and save model
        if np.all(min_loss > loss_history[:,e]):
            min_loss = loss_history[:,e].copy()
            torch.save(cnf.state_dict(), f'./result/state_{file_name}.tar')
