import sys
import argparse
import random
import numpy as np
import pandas as pd
import plotly.express as px
from tqdm import tqdm
from pathlib import Path
from sklearn.model_selection import train_test_split
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import util
import datasets2


# + tags=[]
class VAE(nn.Module):
    def __init__(self, in_shape:tuple, latent_dim:int=128):
        super(VAE, self).__init__()
        in_channels, H, W = in_shape
        o_pad = 0 if H ==28 else 1
        self.latent_dim = latent_dim
        # self.loss_func = nn.BCELoss(size_average=False) 
        self.loss_func = nn.MSELoss(reduction='sum')
        # encoder
        self.encoder = nn.Sequential(nn.Conv2d(in_channels, 16, kernel_size=5, stride=2, padding=2), nn.ReLU(),
                                     nn.BatchNorm2d(16),
                                     nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=2), nn.ReLU(),
                                     nn.BatchNorm2d(32),
                                     nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=0), nn.ReLU(),
                                     nn.BatchNorm2d(64))
        self.flat = nn.Flatten()
        # reparametrizaion
        self.fc_mean = nn.Linear(2*2*64, latent_dim)
        self.fc_var = nn.Linear(2*2*64, latent_dim)
        # decoder
        self.d_lin = nn.Sequential(nn.Linear(latent_dim, 2*2*64), nn.ReLU(),)
        self.unflat = nn.Unflatten(dim=1, unflattened_size=(64, 2, 2))
        self.decoder = nn.Sequential(nn.ConvTranspose2d(64, 32, kernel_size=5, stride=2, output_padding=o_pad),
                                     nn.BatchNorm2d(32),
                                     nn.ReLU(),
                                     nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
                                     nn.BatchNorm2d(16),
                                     nn.ReLU(),
                                     nn.ConvTranspose2d(16, in_channels, kernel_size=3, stride=2, padding=1, output_padding=1))

    def encode(self, x):
        z = self.encoder(x)
        z = self.flat(z)
        return z

    def decode(self, z):
        x_hat = self.d_lin(z)
        x_hat = self.unflat(x_hat)
        x_hat = self.decoder(x_hat)
        x_hat = torch.sigmoid(x_hat)
        return x_hat

    def reparametrizaion(self, mean, log_var):
        std = torch.exp(0.5 * log_var)
        epsilon = torch.randn_like(std)
        return mean + epsilon*std

    def forward(self, x):
        x_out = self.encode(x)
        mean = self.fc_mean(x_out)
        log_var = self.fc_var(x_out)
        KL = -0.5 * torch.sum(1 + log_var - mean**2 - torch.exp(log_var))
        z = self.reparametrizaion(mean, log_var)
        x_hat = self.decode(z)
        reconstruction = self.loss_func(x, x_hat)
        lower_bound = reconstruction + KL
        lower_bound /= x.shape[0]
        return x_hat, z, lower_bound


def train_vae(x, z_dim, epochs=1001, batch_size=200, seed=1234):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if isinstance(x, list): x = np.vstack(x)
    dataset = util.preprocess_input(x)
    train_loader = DataLoader(dataset.tensors[0], batch_size=batch_size, shuffle=True, drop_last=True)
    
    model = VAE(x.shape[1:], z_dim)
    model = util.torch_to(model)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    
    loss_history = np.zeros(epochs)
    for e in tqdm(range(epochs)):
        for x_sample in train_loader:
            x_sample = util.torch_to(x_sample)
            x_decode, z, loss = model(x_sample)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_history[e] += loss.item()
    return model, loss_history


def encode_data(model, x):
    data = util.preprocess_input(x)
    model, data = util.torch_to(model, data.tensors[0])
    with torch.no_grad():
        x_hat, z, loss = model(data)
    return z.detach().cpu().numpy()


def decode_data(model, z):
    data = util.preprocess_input(z)
    model, data = util.torch_to(model, data.tensors[0])
    with torch.no_grad():
        x_hat = model.decode(data)
    x_hat = x_hat.detach().cpu().numpy() 
    N, C, H, W = x_hat.shape
    if C == 1:
        return x_hat.reshape(N, H, W)
    else:
        return x_hat.reshape(N, H, W, C)
    
    
def set_parser():
    parser = argparse.ArgumentParser(description='VAE')
    parser.add_argument("--n_inter", type=int, choices=[1, 27])
    parser.add_argument("--n_component", type=int, choices=2**np.arange(1, 6))
    parser.add_argument("--make_summary", type=eval, default=False)
    return parser


# -

if __name__ == '__main__':
    epochs = 3001
    batch_size = 1500
    n_sample = 60000
    n_sub_sample = 2000
    given_domain = [0, 14, 28, 29]
    args = set_parser().parse_args()
    
    if args.make_summary:
        for file_name in ['mnist_vae_sparse', 'mnist_vae']:
            for file_type in ['data', 'lh']:
                path = list(Path('./data').glob(f'{file_type}_{file_name}_[0-9]*.pkl'))
                data_dict = {}
                for p in path:
                    data_dict.update(pd.read_pickle(p))
                pd.to_pickle(data_dict, f'./data/{file_type}_{file_name}.pkl')
                [p.unlink() for p in path]
        sys.exit()

    file_name = 'mnist_vae' if args.n_inter==27 else 'mnist_vae_sparse'

    x_all, y_all = datasets2.load_RotMNIST_for_generate(start=0, end=60, num_inter_domain=args.n_inter, num_sample=n_sample)
    sample_idx, _ = train_test_split(np.arange(n_sample), train_size=n_sub_sample, stratify=y_all[0], random_state=1234)
    if args.n_inter == 27:
        y_all = [y_all[i].copy() for i in given_domain]
    y_all = [y[sample_idx].copy() for y in y_all]

    vae, lh = train_vae(x_all, args.n_component, epochs, batch_size, 1234)
    z_all = [encode_data(vae, x) for x in x_all]

    if args.n_inter == 27:
        z_all = [z_all[i].copy() for i in given_domain]
    z_all = [z[sample_idx,:].copy() for z in z_all]
    
    results, loss_history = {}, {}
    results[args.n_component] = (z_all, y_all)
    loss_history[args.n_component] = lh

    pd.to_pickle(results, f'./data/data_{file_name}_{args.n_component}.pkl')
    pd.to_pickle(loss_history, f'./data/lh_{file_name}_{args.n_component}.pkl')
    torch.save(vae.state_dict(), f'./data/state_{file_name}_{args.n_component}D.tar')
