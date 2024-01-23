import torch
from torch import nn, optim


# +
class FlowMLP(nn.Module):
    def __init__(self, input_dim:int, n_hidden:int):
        super(FlowMLP, self).__init__()
        self.fc = nn.Sequential(
                                nn.Linear(input_dim, n_hidden), nn.ReLU(),
                                nn.Linear(n_hidden, n_hidden), nn.ReLU(),
                                nn.Linear(n_hidden, n_hidden), nn.ReLU(),
                                nn.Linear(n_hidden, n_hidden), nn.ReLU(),
                                nn.Linear(n_hidden, n_hidden), nn.ReLU(),
                                nn.Dropout(p=0.2),
                               )
        # s and t have a individual weight
        self.fc_s = nn.Linear(n_hidden, input_dim)
        self.fc_t = nn.Linear(n_hidden, input_dim)

    def forward(self, x):
        out = self.fc(x)
        # to avoid a divergence of loss
        s = torch.tanh(self.fc_s(out))
        t = self.fc_t(out)
        return s, t


class RealNVP(nn.Module):
    def __init__(self, n_flows:int, data_dim:int, n_hidden:int):
        super(RealNVP, self).__init__()
        self.n_flows = n_flows
        self.n_hidden = n_hidden
        self.network = torch.nn.ModuleList()
        # for coupling
        assert data_dim % 2 == 0
        self.n_half = data_dim // 2
        for i in range(n_flows):
            self.network.append(FlowMLP(self.n_half, self.n_hidden))

    def forward(self, x):
        log_det_jacobian = 0
        self.inter_repr = [x.detach().numpy()]
        for i in range(self.n_flows):
            x_a = x[:, :self.n_half]
            x_b = x[:, self.n_half:]
            s, t = self.network[i](x_a)
            x_b = torch.exp(s) * x_b + t
            x = torch.cat([x_b, x_a], dim=1)
            log_det_jacobian += s
            self.inter_repr.append(x.detach().numpy())
        return x, log_det_jacobian

    def inverse(self, z):
        self.inter_repr = [z.detach().numpy()]
        for i in reversed(range(self.n_flows)):
            z_a = z[:, self.n_half:]
            z_b = z[:, :self.n_half]
            s, t = self.network[i](z_a)
            z_b = (z_b - t) / torch.exp(s)
            z = torch.cat([z_a, z_b], dim=1)
            self.inter_repr.append(z.detach().numpy())
        return z


# -

if __name__ == '__main__':
    import random
    import numpy as np
    import pandas as pd
    from tqdm import tqdm
    from torch.utils.data import DataLoader
    import util
    import ffjord as C
    from Distributions import knnDistribution, Gaussian
    
    # Following the settings of reference CNF model
    file_name = 'moon_seed_1'
    args = pd.read_pickle(f'./result/args_{file_name}.pkl')
    args.file_name = 'realnvp_' + args.file_name + "_" +str(args.seed)
    
    # load data
    x_all, y_all = pd.read_pickle(f'./data/data_{args.dataset}.pkl')[args.n_dim]
    x_eval, y_eval = x_all.pop(), y_all.pop()  # remove evaluation data

    # unsupervised or semi-supervised
    y_all = util.mask_labels(y_all, args.label_ratio, args.seed)
    
    # source and intermediate data is used for training of RealNVP
    prior = knnDistribution(x_all[0], args.log_prob_param, args.seed)
    # prior = Gaussian(args.n_dim, args.seed)
    dataset = util.preprocess_input(x_all[1], y_all[1])
    train_loader = DataLoader(dataset, batch_size=x_all[1].shape[0], shuffle=True, drop_last=True)
    
    # save RealNVP args
    pd.to_pickle(args, f'./result/args_{args.file_name}.pkl')
    
    # start training
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    args.dims = [int(i) for i in args.dims.split('-')]
    dnf = RealNVP(n_flows=len(args.dims), data_dim=args.n_dim, n_hidden=args.dims[0])
    optimizer = optim.Adam(dnf.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    loss_history = np.zeros(shape=(1, args.epochs))
    min_loss = np.full(shape=1, fill_value=np.inf)
    
    for e in tqdm(range(args.epochs)):
        C.update_lr(e, optimizer, args.lr_change)
        for x, y in train_loader:
            z, delta_logp = dnf(x)
            loss = prior.calc_loss(z, -delta_logp, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_history[0,e] += loss.item()
        pd.to_pickle(loss_history, f'./result/lh_{args.file_name}.pkl')
        # update min_loss and save model
        if np.all(min_loss > loss_history[:,e]):
            min_loss = loss_history[:,e].copy()
            torch.save(dnf.state_dict(), f'./result/state_{args.file_name}.tar')
