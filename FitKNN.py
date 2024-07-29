import math
import argparse
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier, KDTree
from sklearn.model_selection import GridSearchCV, StratifiedKFold, KFold
from mainCNF import get_datasets, save_args
from cnf_args import DATASETS


# +
def get_parser():    
    parser = argparse.ArgumentParser(description='baseline methods')
    parser.add_argument("--dataset", type=str, default="moon", choices=DATASETS)
    parser.add_argument("--n_dim", type=int, default=None, help="the number of dimension of dataset")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--no_inter", type=eval, default=False)
    parser.add_argument("--source_only", type=eval, default=False)
    parser.add_argument("--label_ratio", type=str, default="100-0-0", help="ratio of labeled samples of each domain")
    return parser


def estimate_k(x_all, y_all):
    # use source dataset only
    x, y = x_all[0].copy(), y_all[0].copy()
    # construct knn graph with high accuracy settings
    graph = NearestNeighbors(n_neighbors=x.shape[0], algorithm='brute', n_jobs=4)
    graph.fit(x) 
    # estimate
    _, idx = graph.kneighbors(x)
    k = [np.argmin(y[i[0]] == y[i]) for i in idx]
    opt_k = np.quantile(k, 0.25)
    return int(opt_k), k


def fit_knn_classifier(dataset: str, seed=1):
    # set args
    parser = get_parser()
    args = parser.parse_args([])
    args.dataset = dataset
    if dataset == 'block':
        args.label_ratio = '100-0-0-0'
    # use source dataset only
    x_all, y_all = get_datasets(args)
    best_params = []
    for x, y in zip(x_all, y_all):
        # fit knn classifier
        clf = KNeighborsClassifier(weights='distance', algorithm='brute')
        param_grid = {"n_neighbors": [5, 10, 15, 20, 30, 50]}
        # param_grid = {"n_neighbors": np.arange(5, 31, 1)}
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
        cv_model = GridSearchCV(clf, param_grid=param_grid, scoring="accuracy", n_jobs=4, cv=skf)
        _ = cv_model.fit(x, y) 
        best_params.append(cv_model.best_params_['n_neighbors'])
    return best_params


def logp_knn(x_all:list=None, dataset:str=None, ks:list=[5]):
    if dataset is not None:
        # set args
        parser = get_parser()
        args = parser.parse_args([])
        args.dataset = dataset
        if dataset == 'block':
            args.label_ratio = '100-0-0-0'
        # use input data only
        x_all, _ = get_datasets(args)
    result_jack = np.full(shape=(len(x_all), len(ks)), fill_value=np.nan)
    result_mse = np.full_like(result_jack, np.nan)
    result_bias = np.full_like(result_jack, np.nan)
    result_variance = np.full_like(result_jack, np.nan)
    for i, x in enumerate(x_all):
        result_jack[i,:], result_mse[i,:], result_bias[i,:], result_variance[i,:] = logp_knn_screening(x, ks)
    return result_jack, result_mse, result_bias, result_variance


def logp_knn_screening(X: np.ndarray, ks: list):
    
    np.random.seed(1)
    idx = np.random.choice(np.arange(X.shape[0]), size=500, replace=True)
    X = X[idx,:].copy()
        
    result_jack, result_mse, result_bias, result_variance = [], [], [], []
    for k in ks:
        _all = _logp_knn(X, k)
        _sub = []
        for idx in range(X.shape[0]):
            X_sub = np.delete(X, idx, axis=0).copy()
            _sub.append(_logp_knn(X_sub, k))
        _sub = np.array(_sub)
        bias = (X.shape[0] - 1) * (_sub.mean() - _all)
        variance = np.sum((_sub - _sub.mean())**2)
        variance *= (X.shape[0] - 1) / X.shape[0]
        jack = _all - bias
        mse = bias**2 + variance
        result_jack.append(jack)
        result_mse.append(mse)
        result_bias.append(bias)
        result_variance.append(variance)
    return np.array(result_jack), np.array(result_mse), np.array(result_bias), np.array(result_variance)


def _logp_knn(X, k):
    k_buffer = 30 if k * 2 < 30 else k * 2    
    Gs = KDTree(X, metric='euclidean')
    dist_buffer, _ = Gs.query(X, k=k_buffer) # return index and distance
    n_zero = (dist_buffer == 0).sum(axis=1)
    if np.all(n_zero == 1): n_zero *= 0
    dist = np.array([_d[k + _nz] for _d, _nz in zip(dist_buffer, n_zero)])
    n = X.shape[0] - n_zero
    d = X.shape[1]
    const = np.log(k / (n-1)) + np.log(math.gamma(1 + 0.5*d)) - 0.5 * d * np.log(np.pi)
    logp = const - d * np.log(dist)
    return logp.mean()



# def logp_knn_cv(x_all:list=None, dataset:str=None, ks:list=[5], seed:int=1):
#     if dataset is not None:
#         # set args
#         parser = get_parser()
#         args = parser.parse_args([])
#         args.dataset = dataset
#         if dataset == 'block':
#             args.label_ratio = '100-0-0-0'
#         # use input data only
#         x_all, _ = get_datasets(args)
#     result = np.full(shape=(len(x_all), len(ks)), fill_value=np.nan)
#     for i, x in enumerate(x_all):
#         result[i,:] = logp_knn_screening_cv(x, ks, seed)
#     return result


# def logp_knn_screening_cv(X: np.ndarray, ks: list, seed:int):
    
#     np.random.seed(seed)
#     index = np.arange(X.shape[0])
#     np.random.shuffle(index)
#     split_idx = np.array_split(np.arange(X.shape[0]), 10)
        
#     result = []
#     for k in ks:
#         logp = []
#         for i in split_idx:
#             X_tr = np.delete(X, i, axis=0).copy()
#             X_te = X[i].copy()
#             _logp = _logp_knn_cv(X_tr, X_te, k)
#             logp.append(_logp)
#         result.append(np.mean(logp))
        
#     return np.array(result)


# def _logp_knn_cv(X_tr, X_te, k):
#     n, d = X_tr.shape
#     const = np.log(k / n) + np.log(math.gamma(1 + 0.5*d)) - 0.5 * d * np.log(np.pi)
#     Gs = KDTree(X_tr, metric='euclidean')
#     dist, _ = Gs.query(X_te, k=k) # return index and distance
#     dist = dist[:,-1]
#     logp = const - d * np.log(dist)
#     return logp.mean()

# +
if __name__ == '__main__':
    data = {'mnist':'Rotating MNIST', 'portraits':'Portraits', 'shift15m':'SHIFT15M', 'rxrx1':'RxRx1',
            'tox21a':'Tox21 NHOHCount', 'tox21b':'Tox21 RingCount', 'tox21c':'Tox21 NumHDonors'}
    
    ks = [5, 10, 15, 20, 30] 
    
    for d in data:
        jack, mse, bias, variance = logp_knn(dataset=d, ks=ks)
        result = {'jack': jack, 'mse': mse, 'bias': bias, 'variance': variance}
        pd.to_pickle(result, f'./result/likelihood_{d}.pkl')
    
    # for mnist index exp
    x_all, _ = pd.read_pickle('./data/data_mnist_dense.pkl')[4]
    index = [4, 8, 12, 14, 16, 20, 24]
    x_all = [x_all[i].copy() for i in index]
    jack, mse, bias, variance = logp_knn(x_all=x_all, ks=ks)
    result = {'jack': jack, 'mse': mse, 'bias': bias, 'variance': variance, 'index': index}
    pd.to_pickle(result, f'./result/likelihood_mnist_index.pkl')
    
    
#     import plotly.express as px
#     result = pd.read_pickle('./result/likelihood_tox21a.pkl')    
#     j = 0
    
#     fig = px.scatter(x=ks, y=result["bias"][j]**2, title='bias')
#     fig.show()

#     fig = px.scatter(x=ks, y=result["variance"][j], title='variance')
#     fig.show()

#     fig = px.scatter(x=ks, y=result["mse"][j], title='mse')
#     fig.show()

#     fig = px.scatter(x=ks, y=result["jack"][j], title='jack')
#     fig.show()

