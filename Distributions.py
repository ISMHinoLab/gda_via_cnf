import torch
import numpy as np
from scipy.special import psi, gamma
from sklearn.mixture import GaussianMixture
from pynndescent import NNDescent

# +
class GaussianMixtureDA():
    """
    @memo
    using for base distribution
    @param
    n_dims: the number of input dimensions
    n_labels: the number of labels
    seed: set the seed of random number
    mean_r: hyper parameter of set_fix_gaussian function
    """
    def __init__(self, n_dims:int, n_labels:int, seed:int=1234, mean_r:float=None):
        self.n_dims = n_dims
        self.n_labels = n_labels
        self.seed = seed
        self.weights = torch.nn.functional.softmax(torch.ones(n_labels), dim=0)
        if mean_r is None:
            self.gaussians = [self.set_random_gaussian(k) for k in range(n_labels)]
        else:
            self.gaussians = self.set_fix_gaussian(mean_r)

    def set_random_gaussian(self, k:int):
        new_seed = int(str(self.seed) + str(k))
        torch.manual_seed(new_seed)
        mu = torch.randn(self.n_dims)
        sigma = torch.eye(self.n_dims)
        return torch.distributions.MultivariateNormal(mu, sigma)
    
    def get_means(self, mean_r:int):
        phis = np.linspace(0, 2*np.pi, self.n_labels+1)[:-1]
        means = []
        for d in range(1, self.n_dims+1):
            mu = np.cos(phis) * (np.sin(phis) ** (d-1)) if d < self.n_dims else np.sin(phis) ** (d-1)
            means.append(mu)
        means = np.array(means).T * mean_r
        means = torch.from_numpy(means).float()
        return means
    
    def set_fix_gaussian(self, mean_r:int):
        mu_k = self.get_means(mean_r)
        sigma = torch.eye(self.n_dims)
        gaussians = [torch.distributions.MultivariateNormal(k, sigma) for k in mu_k]
        return gaussians
    
    def log_prob(self, z:torch.Tensor):
        """
        @return
        component_log_prob: Tensor, mixture_log_prob: Tensor
        """
        component_log_prob = [g.log_prob(z.to('cpu')) for g in self.gaussians]
        component_log_prob = torch.stack(component_log_prob, dim=1) # shape -> [n_sample,  n_labels]
        # The first term in the right side of Eq.X
        mixture_log_prob = torch.logsumexp(component_log_prob + torch.log(self.weights), dim=1) 
        return component_log_prob.to(z), mixture_log_prob.to(z)
    
    def predict(self, z:torch.Tensor, return_numpy:bool=True):
        """
        @return
        predict: numpy array, probability: numpy array
        """
        component_log_prob, mixture_log_prob = self.log_prob(z)
        prob = []
        for c, weight in enumerate(self.weights):
            lp = component_log_prob[:,c] + torch.log(self.weights[c]) - mixture_log_prob
            prob.append(torch.exp(lp))
        prob = torch.stack(prob, dim=1) # shape -> [n_sample,  n_labels], prob.sum(dim=1) -> all 1
        predict = prob.argmax(dim=1) # shape -> [n_sample, 1]
        if return_numpy:
            return predict.detach().cpu().numpy(), prob.detach().cpu().numpy()
        else:
            return predict, prob
        
    def calc_loss(self, z:torch.tensor, delta_logp:torch.tensor, y:torch.tensor=None, conv:bool=False):
        """
        @retrun
        loss: Tensor
        """
        return self._calc_loss(z, delta_logp, y, conv)

    def _calc_loss(self, z, delta_logp, y=None, conv=False):
        delta_logp = delta_logp.view(-1)
        component_log_prob, mixture_log_prob = self.log_prob(z)
        if y is None:
            # unspuervised
            loss = mixture_log_prob - delta_logp
        else:
            loss = torch.zeros_like(mixture_log_prob)
            # semi-supervised
            mask = (y == -1)
            loss[mask] += mixture_log_prob[mask]
            # supervised
            for c in range(self.n_labels):
                mask = (y == c)
                loss[mask] += component_log_prob[mask,c]
            loss -= delta_logp
        # post processing, tabluar or image data
        if conv:
            return - ((torch.sum(loss) / z.nelement()) - np.log(256)) / np.log(2)
        else:
            return - torch.mean(loss)
        
    def sample(self, total_size:int):
        """
        @retrun
        z: Tensor, component_log_prob: Tensor
        """
        size = (total_size * self.weights).to(torch.int)
        torch.manual_seed(self.seed)
        z = torch.cat([g.sample(torch.Size([s])) for g, s in zip(self.gaussians, size)])
        component_log_prob, _ = self.log_prob(z)
        component_log_prob = component_log_prob.max(dim=1).values
        return z, component_log_prob.reshape(-1,1) # log_prob is used as an input of CNF
        
        
class Gaussian(GaussianMixtureDA):
    """
    @memo
    using for base distribution
    """
    def __init__(self, n_dims:int, seed:int=1234):
        self.n_dims = n_dims
        self.seed = seed
        self.gaussian = torch.distributions.MultivariateNormal(torch.zeros(self.n_dims), torch.eye(self.n_dims))
 
    def log_prob(self, z:torch.Tensor):
        """
        @return
        None, log_prob: Tensor
        """
        lp = self.gaussian.log_prob(z.to('cpu'))
        return None, lp.to(z)
    
    def predict(self, *unused_kwargs):
        raise NotImplementedError
    
    def sample(self, total_size:int):
        """
        @retrun
        z: Tensor, log_prob: Tensor
        """
        torch.manual_seed(self.seed)
        z = self.gaussian.sample(torch.Size([total_size]))
        _, lp = self.log_prob(z)
        return z, lp.reshape(-1, 1)
    
    def calc_loss(self, z:torch.tensor, delta_logp:torch.tensor, y:torch.tensor=None, conv:bool=False):
        """
        @retrun
        loss: Tensor
        """
        # can't input y
        return self._calc_loss(z, delta_logp, None, conv)



class GaussianMixtureEM(GaussianMixtureDA):
    """
    @memo
    using for arbitary target distribution
    @param
    x: realizations of X^{(j)}
    n_components: the number of mixture components
    seed: set the seed of random number
    """
    def __init__(self, x:np.ndarray, n_components:int, seed:int=1234):
        if x.ndim == 4: x = x.reshape(x.shape[0], -1)  # image data
        self.n_dims = x.shape[1]
        self.n_components = n_components
        self.seed = seed
        self.gaussians, self.weights = self.fit_gaussian_mixture(x) 
        
    def fit_gaussian_mixture(self, x):
        gm = GaussianMixture(n_components=self.n_components, random_state=self.seed)
        gm.fit(x)
        means, sigmas = torch.from_numpy(gm.means_).float(), torch.from_numpy(gm.covariances_).float() 
        gaussians = []
        for mu, sigma in zip(means, sigmas):        
            gaussians.append(torch.distributions.MultivariateNormal(mu, sigma))
        return gaussians, torch.from_numpy(gm.weights_).float()
    
    def predict(self, *unused_kwargs):
        raise NotImplementedError
        
    def sample(self, *unused_kwargs):
        raise NotImplementedError
        
    def calc_loss(self, z:torch.tensor, delta_logp:torch.tensor, y:torch.tensor=None, conv:bool=False):
        """
        @retrun
        loss: Tensor
        """
        # can't input y
        return self._calc_loss(z, delta_logp, None, conv)
    

class knnDistribution(GaussianMixtureDA):
    """
    @memo
    using for arbitary target distribution
    @param
    x: realizations of X^{(j)}
    n_neighborsint: the number of neighbors
    seed: set the seed of random number
    """
    def __init__(self, x:np.ndarray, n_neighbors:int=30, seed:int=1234):
        if x.ndim == 4: x = x.reshape(x.shape[0], -1)  # image data
        self.n_samples, self.n_dims = x.shape
        self.n_neighbors = n_neighbors
        self.seed = seed
        self.const = self.set_constant()
        self.graph = self.fit_knn_graph(x)
        self.x_tensor = torch.tensor(x, dtype=torch.float32)
        self.pdist = torch.nn.PairwiseDistance(p=2)
        
    def set_constant(self):
        cd = (np.pi ** (self.n_dims/2)) / (gamma(1+(self.n_dims/2)))
        return -psi(self.n_samples) + psi(self.n_neighbors) - np.log(cd)
    
    def fit_knn_graph(self, x):
        graph = NNDescent(x, metric='euclidean', n_neighbors=300, random_state=self.seed, n_jobs=4)
        graph.prepare()
        return graph
    
    def predict(self, *unused_kwargs):
        raise NotImplementedError
        
    def sample(self, *unused_kwargs):
        raise NotImplementedError

    def log_prob(self, z:torch.Tensor):
        m = z.shape[0]
        z_numpy = z.detach().cpu().numpy()
        knn_index, _ = self.graph.query(z_numpy, k=self.n_neighbors)
        knn_dist = self.pdist(z, self.x_tensor.to(z)[knn_index[:,-1]])
        lp = self.const - (self.n_dims/m) * torch.log(knn_dist).sum()    
        return None, lp.to(z)
    
    def calc_loss(self, z:torch.tensor, delta_logp:torch.tensor, y:torch.tensor=None, conv:bool=False):
        """
        @retrun
        loss: Tensor
        """
        # can't input y
        return self._calc_loss(z, delta_logp, None, conv)
