import argparse

NONLINEARITIES = ["tanh", "relu", "softplus", "elu", "swish"]
SOLVERS = ["dopri5", "bdf", "rk4", "midpoint", 'adams', 'explicit_adams']
LAYERS = ["ignore", "concat", "concat_v2", "squash", "concatsquash", "concatcoord", "hyper", "blend"]
DATASETS = ['moon', 'scaled_moon', 'block', 'gaussian', 'mnist','mnist_dense', 'mnist_vae', 'portraits', 'tox21a', 'tox21b', 'tox21c', 'rxrx1', 'shift15m']

parser = argparse.ArgumentParser(description='Gradual Domain Adaptation via Normalizing Flows')

parser.add_argument("--layer_type", type=str, default="concatsquash", choices=LAYERS)
parser.add_argument('--dims', type=str, default='64-64-64')
parser.add_argument("--num_blocks", type=int, default=1, help='Number of stacked CNFs. Not Use')
parser.add_argument('--time_length', type=float, default=1)
parser.add_argument('--train_T', type=eval, default=True)
parser.add_argument("--divergence_fn", type=str, default="approximate", choices=["brute_force", "approximate"])
parser.add_argument("--nonlinearity", type=str, default="tanh", choices=NONLINEARITIES)
parser.add_argument('--batch_norm', type=eval, default=False, choices=[True, False])
parser.add_argument('--bn_lag', type=float, default=0)
parser.add_argument('--residual', type=eval, default=False, choices=[True, False])
parser.add_argument('--rademacher', type=eval, default=True, choices=[True, False])

parser.add_argument('--solver', type=str, default='dopri5', choices=SOLVERS)
parser.add_argument('--atol', type=float, default=1e-4)
parser.add_argument('--rtol', type=float, default=1e-4)
parser.add_argument("--step_size", type=float, default=None, help="Optional fixed step size.")
parser.add_argument('--test_solver', type=str, default=None, choices=SOLVERS + [None])
parser.add_argument('--test_atol', type=float, default=1e-5)
parser.add_argument('--test_rtol', type=float, default=1e-5)

parser.add_argument('--poly_coef', type=float, default=None, help="Coefficient of polynomial regression loss")
parser.add_argument('--poly_num_sample', type=int, default=2, help="Number of samples of t for polynomial regression loss")
parser.add_argument('--poly_order', type=int, default=1, help="Order of polynomial regression loss")
parser.add_argument('--adjoint', type=eval, default=True, choices=[True, False], help="Using adjoint methods")

# my args
parser.add_argument("--dataset", type=str, default="moon", choices=DATASETS)
parser.add_argument("--n_dim", type=int, default=None, help="the number of dimension of dataset")
parser.add_argument('--batch_size', type=int, default=None)
parser.add_argument('--epochs', type=int, default=3001)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--lr_change", type=int, default=None)
parser.add_argument("--weight_decay", type=float, default=1e-5)
parser.add_argument("--file_name", type=str, required=True)
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--mean_r", type=float, default=None)
parser.add_argument("--no_inter", type=eval, default=False)
parser.add_argument("--source_only", type=eval, default=False)
parser.add_argument("--ignore_last_bn", type=eval, default=False)
parser.add_argument("--label_ratio", type=str, default="100-0-0", help="ratio of labeled samples of each domain")
parser.add_argument("--base_distribution", type=str, default='gmm', choices=['normal', 'gmm'])
parser.add_argument("--log_prob_method", type=str, default='knn', choices=['gmm', 'knn'])
# parser.add_argument("--log_prob_param", type=int, default=15, help="n_neighbors or n_components")
parser.add_argument("--log_prob_param", nargs="*", default=[15,15], type=int, help="n_neighbors or n_components")
parser.add_argument("--source_coef", type=float, default=1.0, help="hyper parameter of the training of labeled data")
parser.add_argument("--inter_index", type=int, default=14)
