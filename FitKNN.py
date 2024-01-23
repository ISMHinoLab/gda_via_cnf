import argparse
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold 
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


def fit_knn_classifier(dataset:str):
    # set args
    parser = get_parser()
    args = parser.parse_args([])
    args.dataset = dataset
    if dataset == 'block':
        args.label_ratio = '100-0-0-0'
    # use source dataset only
    x_all, y_all = get_datasets(args)
    x, y = x_all[0].copy(), y_all[0].copy()
    # fit knn classifier
    clf = KNeighborsClassifier(weights='distance', algorithm='brute')
    param_grid = {"n_neighbors":[5, 10, 15, 20, 30, 50]}
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=1)
    cv_model = GridSearchCV(clf, param_grid=param_grid, scoring="accuracy", n_jobs=4, cv=skf)
    _ = cv_model.fit(x, y)
    return cv_model
