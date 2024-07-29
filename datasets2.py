import json
import gzip
import warnings
import numpy as np
import pandas as pd
import plotly.express as px
from PIL import Image
from pathlib import Path
from scipy import ndimage
from rdkit import RDLogger
from rdkit.Chem import Descriptors, PandasTools, AllChem
from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from torchvision.datasets import MNIST
import torchvision.transforms as tforms
import torch
from torch.utils.data import DataLoader

#data_dir = Path('/home/')
data_dir = Path('/home/e12813/ISM/data/')


# +
def load_RotMNIST_for_generate(start=0, end=30, num_inter_domain=1, num_sample=60000, source_only=False):
    
    def add_noise(x):
        """ [0, 1] -> [0, 255] -> add noise -> [0, 1] """
        noise = x.new().resize_as_(x).uniform_()
        x = x * 255 + noise
        x = x / 256
        return x

    # set angles
    angles = np.linspace(start, end, num_inter_domain+2)
    angles = np.append(angles, end)

    # load mnist
    trans = lambda angle: tforms.Compose([tforms.Resize(28), tforms.ToTensor(), 
                                          tforms.RandomRotation(degrees=(angle,angle)), add_noise])
    mnist_dir = data_dir if data_dir.exists() else Path('./')
    x_all, y_all = [], []
    for seed, angle in enumerate(angles):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            dataset = MNIST(mnist_dir, train=True, download=True, transform=trans(angle))
        # torch.manual_seed(seed)
        torch.manual_seed(1234)
        train_loader = DataLoader(dataset, batch_size=num_sample, shuffle=True, drop_last=True)
        x, y = next(iter(train_loader))
        x_all.append(x.numpy())
        y_all.append(y.numpy())
        if source_only:
            break
    return x_all, y_all


def load_RotatedMNIST2(start=0, end=60, num_inter_domain=27, num_domain_samples=2000):
    """
    @param
    start, end: int, rotate angles
    num_inter_domain: int, how many intermediate domains needed
    num_inter_samples: set the same sample size in all domains (source, inter, target, eval)
    """
    global data_dir
    np.random.seed(1234)
    # load MNIST
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        # rotated mnist does not need test data
        dataset = MNIST(data_dir, train=True, download=True)
    x = np.array(dataset.data).astype(np.float32) / 255
    y = np.array(dataset.targets)
    # set angles
    angles = np.linspace(start, end, num_inter_domain+2)
    angles = np.append(angles, end)
    # set sample size and index
    index = np.arange(x.shape[0])
    np.random.shuffle(index)
    each_domain_samples = np.full(shape=(num_inter_domain+3), fill_value=num_domain_samples)  # source + inter + target +eval
    split_index = np.split(index, np.cumsum(each_domain_samples))
    # rotate
    x_all, y_all = list(), list()
    for idx, angle in zip(split_index, angles):
        #rotated_x = np.array([ndimage.rotate(i, np.random.normal(loc=angle, scale=5), reshape=False) for i in x[idx]])
        rotated_x = np.array([ndimage.rotate(i, angle, reshape=False) for i in x[idx]])
        x_all.append(rotated_x.reshape(-1, 1, 28, 28))
        y_all.append(y[idx])
    return x_all, y_all


def make_split_data(df: pd.DataFrame, target: str, num_inter_domain: int, num_domain_samples: dict):
    split_index = np.split(np.arange(df.shape[0]), np.cumsum(list(num_domain_samples.values())))
    x_all, y_all = list(), list()
    for idx, key in zip(split_index, num_domain_samples.keys()):
        x = df.drop(target, axis=1).loc[idx].values
        y = df.loc[idx, target].values
        if key == 'inter':
            x_all += np.vsplit(x, num_inter_domain)
            y_all += np.hsplit(y, num_inter_domain)
        else:
            x_all.append(x)
            y_all.append(y)
    return x_all, y_all


def shuffle_target_and_eval(x_all: list, y_all: list):
    tx, ty = x_all[-2].copy(), y_all[-2].copy()
    ex, ey = x_all[-1].copy(), y_all[-1].copy()
    marge_x = np.vstack([tx, ex])
    marge_y = np.hstack([ty, ey])
    idx = np.arange(marge_x.shape[0])
    np.random.seed(1234)
    np.random.shuffle(idx)
    t_idx, e_idx = idx[:tx.shape[0]], idx[tx.shape[0]:]
    x_all[-2], y_all[-2] = marge_x[t_idx], marge_y[t_idx]
    x_all[-1], y_all[-1] = marge_x[e_idx], marge_y[e_idx]
    return x_all, y_all


def read_path(sex: int):
    """ for load_Portraits function """
    p = 'portraits/F' if sex == 1 else 'portraits/M'
    p = Path(data_dir) / p
    p_list = list(p.glob("*.png"))
    data_frame = pd.DataFrame({'img_path': p_list})
    data_frame['sex'] = sex
    return data_frame


def convert_portraits(p: Path):
    """ for load_Portraits function """
    # read, gray scale, resize
    img = Image.open(p).convert('L').resize((32,32), Image.ANTIALIAS)
    img = np.array(img, dtype=np.float32) / 255
    return img


def load_Portraits(num_inter_domain=6, num_domain_samples='default', use_domain_index=[0, 3, 7, 8], return_df=False):
    """
    @param
    num_inter_domain: inter domain data will be vsplit by this param
    num_domain_samles: number of samples in each domain.

    @memo
    image shape will be change, (N, height, width) -> (N, 1, height, width)
    https://www.dropbox.com/s/ubjjoo0b2wz4vgz/faces_aligned_small_mirrored_co_aligned_cropped_cleaned.tar.gz?dl=0
    """
    global data_dir
    if num_domain_samples == 'default':
        num_domain_samples = {'source': 2000, 'inter': 12000, 'target': 2000, 'eval': 2000}

    # prepare portraits image, female as 1, male as 0
    df = pd.concat([read_path(1), read_path(0)]).reset_index(drop=True)
    df['year'] = df['img_path'].apply(lambda p: p.stem.split('_')[0]).astype(int)
    if return_df:
        df['decade'] = df['year'].apply(lambda y: int(str(y)[:3]+'0'))
        return df
    df = df.sort_values(by='year').reset_index(drop=True).drop('year', axis=1)

    # split to each domain
    x_all, y_all = make_split_data(df, 'sex', num_inter_domain, num_domain_samples)
    x_all, y_all = shuffle_target_and_eval(x_all, y_all)
    x_all = [x_all[i].copy() for i in use_domain_index]
    y_all = [y_all[i].copy() for i in use_domain_index]
    for i, domain in enumerate(x_all):
        domain = np.array([convert_portraits(x) for x in domain.flatten()])
        x_all[i] = domain.reshape(-1, 1, 32, 32)
    return x_all, y_all


def load_Portraits2(descending=True, inter:list=[1960], n_eval_sample=1000):
    # prepare portraits image, female as 1, male as 0
    df = pd.concat([read_path(1), read_path(0)]).reset_index(drop=True)
    df['year'] = df['img_path'].apply(lambda p: p.stem.split('_')[0]).astype(int)
    df['decade'] = df['year'].apply(lambda y: int(str(y)[:3]+'0'))
    
    # set domain
    if len(inter) == 0:
        years = [1930, 1940, 1950, 1960, 1970, 1980, 1990, 2000]
    else:
        years = [1930] + inter + [2000]
        
    if descending:
        years = years[::-1]
        
    # read image
    x_all, y_all = [], []
    for y in years:
        subset = df.query('decade==@y').copy()
        x = subset['img_path'].apply(convert_portraits).tolist()
        x = np.array(x)
        x = x.reshape(x.shape[0], 1, 32, 32)
        y = subset['sex'].values
        x_all.append(x)
        y_all.append(y)
    
    # make eval data
    x_target, y_target = x_all[-1].copy(), y_all[-1].copy()
    candidate = np.arange(y_target.shape[0])
    np.random.seed(1234)
    np.random.shuffle(candidate)
    idx_target = candidate[n_eval_sample:]
    idx_eval = candidate[:n_eval_sample]
    x_all[-1], y_all[-1] = x_target[idx_target], y_target[idx_target]
    x_all.append(x_target[idx_eval])
    y_all.append(y_target[idx_eval])
    
    return x_all, y_all


def make_imbalance_data(x_all, y_all):
    n_domain = len(x_all)
    for i in range(n_domain):
        labels, counts = np.unique(y_all[i], return_counts=True)
        n_removes = np.arange(1, counts.min()*0.2, 5, dtype=int)
        np.random.seed(i)
        np.random.shuffle(n_removes)
        idx = []
        for j, l in enumerate(labels):
            candidate = np.argwhere(y_all[i] == l).flatten()
            label_idx = np.random.choice(candidate, size=candidate.size-n_removes[j], replace=False)
            idx.append(label_idx)
        idx = np.hstack(idx)
        x_all[i] = x_all[i][idx,:]
        y_all[i] = y_all[i][idx]
    return x_all, y_all


def make_gradual_gaussian(n_dims, n_sample=2000):
    # make label
    y = np.zeros(n_sample)
    np.random.seed(1234)
    idx = np.random.choice(np.arange(n_sample), size=n_sample//2, replace=False)
    y[idx] = 1
    # make feature
    x_all, y_all = [], []
    mu = torch.zeros(n_dims)
    for s in [0.5, 0.8, 0.99, 0.99]: # source, inter, target, eval
        sigma = torch.full(size=(n_dims, n_dims), fill_value=s)
        sigma = sigma.fill_diagonal_(1)
        prior = torch.distributions.MultivariateNormal(mu, sigma)
        x = prior.sample(torch.Size([n_sample]))
        x_all.append(x.numpy())
        y_all.append(y)
    return x_all, y_all


def make_gradual_gaussian2(n_sample=2000, imbalanced=False):
    n_dims = 2
    
    # mu_z -> class 0: (3.0, 0.0)^\top, class 1: (-3.0, 0.0)^\top
    means0 = torch.tensor([(3.0, 1.0),
                           (6.0, 3.0),
                           (8.0, 3.0),
                           (3.0, 3.0),
                           (3.0, 5.0)])

    means1 = torch.tensor([(-3.0, 1.0),
                           (-6.0, 3.0),
                           (-8.0, 3.0),
                           (-3.0, 3.0),
                           (-3.0, 5.0)])

    sigma = torch.eye(n_dims)

    x_all, y_all = [], []
    for mu1, mu2 in zip(means0, means1):
        prior1 = torch.distributions.MultivariateNormal(mu1, sigma)
        prior2 = torch.distributions.MultivariateNormal(mu2, sigma)
        x = torch.vstack([prior1.sample(torch.Size([n_sample//2])),
                          prior2.sample(torch.Size([n_sample//2]))])
        y = torch.hstack([torch.zeros(n_sample//2),
                          torch.ones(n_sample//2)])
        x_all.append(x.numpy())
        y_all.append(y.numpy())
    # add eval data
    x_all.append(x.numpy())
    y_all.append(y.numpy())
    
    if imbalanced:
        return make_imbalance_data(x_all, y_all)
    else:
        return x_all, y_all


def make_gradual_block(steps=4, n_class=5, n_sample=2000, scaled=False, imbalanced=False):
    """
    @param
    steps: int, contral the density of sequence
    n_class: int, the numer of class
    n_samples: int, the number of samples of each domain
    """
    # make original blocks
    n_sample = n_sample // n_class
    mu_k = [(m,m) for m in np.linspace(-5, 5, n_class)]
    x = np.vstack([np.random.multivariate_normal(m, np.eye(2)*0.1, n_sample) for m in mu_k])
    y = np.hstack([np.full(n_sample, n_class-i) for i in range(n_class)]) - 1
    
    # make gradual blocks
    reflect = np.array([[-1,  0], [0, 1]])
    target = np.dot(x, reflect)
    
    shift_matrix = []
    for c in range(n_class):
        idx = np.where(y==c)[0]
        total_shift_by_class = target[idx,0].min() - x[idx,0].min()
        shift_matrix.append(np.linspace(0, total_shift_by_class, steps)[1:-1])
    shift_matrix = np.vstack(shift_matrix)
    
    x_all, y_all = [x], [y]
    for i in range(steps-2):
        shift = shift_matrix[:,i]
        new_x = x.copy()
        for c, s in enumerate(shift):
            idx = np.where(y==c)[0]
            new_x[idx,0] = new_x[idx,0] + s
        x_all.append(new_x)
        y_all.append(y.copy())

    # add target and eval data
    x_all += [target, target]
    y_all += [y, y]
    
    if scaled:
        X = np.vstack(x_all)
        X = StandardScaler().fit_transform(X)
        x_all = np.split(X, np.cumsum([x.shape[0] for x in x_all]))[:-1]
       
    if imbalanced:
        return make_imbalance_data(x_all, y_all)
    else:
        return x_all, y_all


def make_gradual_data(steps=3, n_samples=2000, start=0, end=90, scaled=False, imbalanced=False):
    """
    @param
    steps: int, how gradual is it
    n_samples: int, how many samples, each domains
    start: int, param of shift
    end: int, param of shift
    """
    x, y = make_moons(n_samples=n_samples, random_state=8, noise=0.05)
    shifts = np.linspace(start, end, steps)
    x_all, y_all = list(), list()
    for shift in shifts:
        x_all.append(_convert_moon(x, shift))
        y_all.append(y)
        # for eval data
        if shift == shifts[-1]:
            x_all.append(_convert_moon(x, shift))
            y_all.append(y)

    if scaled:
        x_all = [StandardScaler().fit_transform(x) for x in x_all]

    if imbalanced:
        return make_imbalance_data(x_all, y_all)
    else:
        return x_all, y_all


def _convert_moon(x: np.ndarray, shift: int) -> np.ndarray:
    x_copy = x.copy()
    rad = np.deg2rad(shift)
    rot_matrix = np.array([[np.cos(rad), np.sin(rad)],
                           [-np.sin(rad), np.cos(rad)]])
    rot_x = x_copy @ rot_matrix
    return rot_x.astype(np.float32)


def load_Tox21a():
    return load_Tox21(domain='NHOH', eval_size=500, seed=1234)

def load_Tox21b():
    return load_Tox21(domain='RingCount', eval_size=500, seed=1234)

def load_Tox21c():
    return load_Tox21(domain='NumHDonors', eval_size=500, seed=1234)


def load_Tox21(domain: str, eval_size:int=500, seed:int=1234):
    """
    @param
    domain: str, the indicator which divide the domain, NHOH/RingCount/NumHDonors
    eval_size: target domain spilit to target and eval dataset
    seed: random seed for train_test_split

    @memo
    We count the number of substituents of the compound and consider the number of substituents as a domain.
    NHOHCount 0 -> source, 1 -> inter, 2 -> target and eval
    """
    df = pd.read_csv(data_dir / 'tox21.csv.gz')
    # We consider compounds as toxic that the compound shows a positive reaction for any of the tests.
    df['ToxSum'] = df.iloc[:, :12].sum(axis=1, skipna=True)
    df['y'] = df['ToxSum'].apply(lambda s: 1 if s >= 1 else 0)
    
    # add Mol object
    RDLogger.DisableLog('rdApp.*')
    PandasTools.AddMoleculeColumnToFrame(df, smilesCol='smiles')
    df['NHOH'] = df['ROMol'].apply(Descriptors.NHOHCount)
    df['RingCount'] = df['ROMol'].apply(Descriptors.RingCount)
    df['NumHDonors'] = df['ROMol'].apply(Descriptors.NumHDonors)
    
    # get RDKit descriptors
    x = []
    for name, func in Descriptors.descList:
        if name not in ['NHOH', 'RingCount', 'NumHDonors']:
            x.append(df['ROMol'].apply(func).values)
    x = np.vstack(x).T
    no_na_column = ~np.isnan(x).any(axis=0)
    x = x[:, no_na_column]
    x = x.astype(np.float32)
    no_inf_column = ~np.isinf(x).any(axis=0)
    x = x[:, no_inf_column]
    x = StandardScaler().fit_transform(x)

    y = df['y'].values
    x_all, y_all = [], []
    for i in [0, 1, 2]:  # source, inter, target
        if domain == 'NHOH':
            idx = df.query('NHOH==@i').index
        elif domain == 'RingCount':
            idx = df.query('RingCount==@i').index
        elif domain == 'NumHDonors':
            idx = df.query('NumHDonors==@i').index
        x_all.append(x[idx])
        y_all.append(y[idx])
    # target domain split to target and eval
    x_target, x_eval, y_target, y_eval = train_test_split(x_all[-1], y_all[-1], test_size=eval_size,
                                                          stratify=y_all[-1], random_state=seed)
    _, _ = x_all.pop(), y_all.pop()
    x_all += [x_target, x_eval]
    y_all += [y_target, y_eval]
    return x_all, y_all


def load_RxRx1(eval_size: int=3000, seed: int=1234):
    """
    @param
    eval_size: target domain spilit to target and eval dataset
    seed: random seed for train_test_split

    @memo
    We estimate the cell type by using the information from images.
    number of experiment 1 -> source, 2 -> inter, 3 -> target and eval
    """
    rxrx1_dir = data_dir / 'rxrx1_v1.0'
    meta_df = pd.read_csv(rxrx1_dir / 'metadata.csv')
    meta_df['cell_type_id'] = meta_df['cell_type'].astype('category').cat.codes
    meta_df['num_experiment'] = meta_df['experiment'].apply(lambda s: int(s.split('-')[1]))
    # add path of images
    meta_df['img_path'] = rxrx1_dir / 'images' / (meta_df['experiment'] + "/Plate" + meta_df['plate'].astype(str) \
                           + "/" + meta_df['well'] + "_s" + meta_df['site'].astype(str) + ".png")
    x_all, y_all = [], []
    for nx in [1, 2, 3]:  # source, inter, target
        x = []
        idx = meta_df.query('num_experiment==@nx').index.values
        for i in idx:
            # The size of original image is 256 * 256
            img = Image.open(meta_df.loc[i, 'img_path']).resize((32,32), Image.ANTIALIAS)
            img = np.array(img, dtype=np.float32) / 255
            x.append(img.flatten())
        x = np.array(x)
        y = meta_df.loc[idx, 'cell_type_id'].values
        x_all.append(x)
        y_all.append(y)
    # target domain split to target and eval
    x_target, x_eval, y_target, y_eval = train_test_split(x_all[-1], y_all[-1], test_size=eval_size,
                                                           stratify=y_all[-1], random_state=seed)
    _, _ = x_all.pop(), y_all.pop()
    x_all += [x_target, x_eval]
    y_all += [y_target, y_eval]
    return x_all, y_all


def load_shift15m(sample_size: int=5000, seed: int=1234):
    """
    @param
    sample_size: sampling size of each year
    seed: random seed for train_test_split

    @memo
    2010&2011 -> source, 2015 -> inter, 2020 -> target
    """
    shift_dir = data_dir / 'shift15m/data'
    item_catalog = pd.read_csv(shift_dir/'item_catalog.txt', header=None, sep=" ",
                               names=["item_id", "category", "subcategory", "year"])
    item_catalog['category_id'] = item_catalog['category'].astype('category').cat.codes
    item_catalog['year'] = item_catalog['year'].replace(2010, 2011)  # merge 2010 and 2011
    # get indices of each domain
    idx_all = []
    # years = sorted(item_catalog['year'].unique())
    for qyear in [2011, 2015, 2020]:
        subset = item_catalog.query('year==@qyear').copy()
        idx = subset['item_id'].index.values
        y = subset['category'].values
        sample_idx, _ = train_test_split(idx, train_size=sample_size, stratify=y, random_state=seed)
        idx_all.append(sample_idx)
        # for eval data
        if qyear == 2020:
            sample_idx, _ = train_test_split(idx, train_size=sample_size, stratify=y, random_state=seed*2)
            idx_all.append(sample_idx)
    # load data
    x_all, y_all = [], []
    for i in idx_all:
        x = []
        for j in item_catalog.loc[i]['item_id'].tolist():
            path = (shift_dir / 'features') / f'{j}.json.gz'
            with gzip.open(path, "r") as f:
                feature = np.array(json.load(f), dtype=np.float32)
                x.append(feature)
        x_all.append(np.array(x))
        y_all.append(item_catalog.loc[i]['category_id'].values)
    return x_all, y_all


# -

if __name__ == '__main__':
    x_all, y_all = make_gradual_block(imbalanced=True)
    obj = {2: (x_all, y_all)}
    pd.to_pickle(obj, './data/data_block.pkl')

    x_all, y_all = make_gradual_data(imbalanced=True)
    obj = {2: (x_all, y_all)}
    pd.to_pickle(obj, './data/data_moon.pkl')

    x_all, y_all = make_gradual_data(scaled=True, imbalanced=True)
    obj = {2: (x_all, y_all)}
    pd.to_pickle(obj, './data/data_scaled_moon.pkl')
    
    x_all, y_all = make_gradual_gaussian2(imbalanced=True)
    obj = {2: (x_all, y_all)}
    pd.to_pickle(obj, './data/data_gaussian.pkl')
