import warnings
import numpy as np
import pandas as pd
import umap
import datasets2


def fit_umap(x_all, y_all, **umap_kwargs) -> list:
    umap_settings = dict(n_components=2, n_neighbors=15, min_dist=0.1, metric='euclidean')
    umap_settings.update(umap_kwargs)
    X = np.vstack(x_all)
    X = X.reshape(X.shape[0], -1)
    # use source label as semi-superviesd UMAP
    Y_semi_supervised = [np.full(shape=y.shape[0], fill_value=-1) for y in y_all]
    Y_semi_supervised[0] = y_all[0].copy()
    Y_semi_supervised = np.hstack(Y_semi_supervised)
    # fit UMAP
    encoder = umap.UMAP(random_state=1234, **umap_settings)
    Z = encoder.fit_transform(X, Y_semi_supervised)
    z_idx = np.cumsum([i.shape[0] for i in x_all])
    z_all = np.vsplit(Z, z_idx)[:-1]
    return z_all, encoder


settings = {
            'portraits': (datasets2.load_Portraits, None),
            'mnist': (datasets2.load_RotatedMNIST2, [0, 14, 28, 29]),
            'mnist_dense': (datasets2.load_RotatedMNIST2, None),
            'rxrx1': (datasets2.load_RxRx1, None),
            'shift15m': (datasets2.load_shift15m, None),
            'tox21a': (datasets2.load_Tox21a, None),
            'tox21b': (datasets2.load_Tox21b, None),
            'tox21c': (datasets2.load_Tox21c, None),
           }


if __name__ == '__main__':

    warnings.simplefilter('ignore')

    components = 2 ** np.arange(1, 6)

    for name in settings.keys():
        metric = "euclidean"
        results = {}
        func, given_domain = settings[name]
        x_all, y_all = func()
        print(f'{name}, {metric}\n')
        for c in components:
            z_all, _ = fit_umap(x_all, y_all, metric=metric, n_components=c)
            # remove NA
            z_all_no_NA, y_all_no_NA = [], []
            for i in range(len(z_all)):
                not_na_idx = ~np.isnan(z_all[i]).any(axis=1)
                z_all_no_NA.append(z_all[i][not_na_idx, :].copy())
                y_all_no_NA.append(y_all[i][not_na_idx].copy())
            if given_domain is not None:
                z_all_no_NA = [z_all_no_NA[i].copy() for i in given_domain]
                y_all_no_NA = [y_all_no_NA[i].copy() for i in given_domain]
            results[c] = (z_all_no_NA, y_all_no_NA)
            pd.to_pickle(results, f'./data/data_{name}.pkl')
        
            
    # # mnist dense
    # name = "mnist_dense"
    # results = {}
    # func, given_domain = settings['mnist']
    # x_all, y_all = func()
    # z_all, _ = fit_umap(x_all, y_all, metric="euclidean", n_components=4)
    # z_all_no_NA, y_all_no_NA = [], []
    # for i in range(len(z_all)):
    #     not_na_idx = ~np.isnan(z_all[i]).any(axis=1)
    #     z_all_no_NA.append(z_all[i][not_na_idx, :].copy())
    #     y_all_no_NA.append(y_all[i][not_na_idx].copy())
    # results[4] = (z_all_no_NA, y_all_no_NA)
    # pd.to_pickle(results, f'./data/data_{name}.pkl')
