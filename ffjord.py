import six
import math
import lib.odenvp as odenvp
import lib.multiscale_parallel as multiscale_parallel
import lib.layers.wrappers.cnf_regularization as reg_lib
import lib.spectral_norm as spectral_norm
import lib.layers as layers
from lib.layers.odefunc import divergence_bf, divergence_approx
import torch
from torch import nn, optim
import numpy as np
from sklearn.metrics import accuracy_score
import util
from Distributions import GaussianMixtureDA


# +
def set_cnf_options(args, model):

    def _set(module):
        if isinstance(module, layers.CNF):
            # Set training settings
            module.solver = args.solver
            module.atol = args.atol
            module.rtol = args.rtol
            if args.step_size is not None:
                module.solver_options['step_size'] = args.step_size

            # If using fixed-grid adams, restrict order to not be too high.
            if args.solver in ['fixed_adams', 'explicit_adams']:
                module.solver_options['max_order'] = 4

            # Set the test settings
            module.test_solver = args.test_solver if args.test_solver else args.solver
            module.test_atol = args.test_atol if args.test_atol else args.atol
            module.test_rtol = args.test_rtol if args.test_rtol else args.rtol

        if isinstance(module, layers.ODEfunc):
            module.rademacher = args.rademacher
            module.residual = args.residual

    model.apply(_set)


def override_divergence_fn(model, divergence_fn):

    def _set(module):
        if isinstance(module, layers.ODEfunc):
            if divergence_fn == "brute_force":
                module.divergence_fn = divergence_bf
            elif divergence_fn == "approximate":
                module.divergence_fn = divergence_approx

    model.apply(_set)


def count_nfe(model):

    class AccNumEvals(object):

        def __init__(self):
            self.num_evals = 0

        def __call__(self, module):
            if isinstance(module, layers.ODEfunc):
                self.num_evals += module.num_evals()

    accumulator = AccNumEvals()
    model.apply(accumulator)
    return accumulator.num_evals


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_total_time(model):

    class Accumulator(object):

        def __init__(self):
            self.total_time = 0

        def __call__(self, module):
            if isinstance(module, layers.CNF):
                self.total_time = self.total_time + module.sqrt_end_time * module.sqrt_end_time

    accumulator = Accumulator()
    model.apply(accumulator)
    return accumulator.total_time


def add_spectral_norm(model, logger=None):
    """Applies spectral norm to all modules within the scope of a CNF."""

    def apply_spectral_norm(module):
        if 'weight' in module._parameters:
            if logger: logger.info("Adding spectral norm to {}".format(module))
            spectral_norm.inplace_spectral_norm(module, 'weight')

    def find_cnf(module):
        if isinstance(module, layers.CNF):
            module.apply(apply_spectral_norm)
        else:
            for child in module.children():
                find_cnf(child)

    find_cnf(model)


def spectral_norm_power_iteration(model, n_power_iterations=1):

    def recursive_power_iteration(module):
        if hasattr(module, spectral_norm.POWER_ITERATION_FN):
            getattr(module, spectral_norm.POWER_ITERATION_FN)(n_power_iterations)

    model.apply(recursive_power_iteration)


REGULARIZATION_FNS = {
    "l1int": reg_lib.l1_regularzation_fn,
    "l2int": reg_lib.l2_regularzation_fn,
    "dl2int": reg_lib.directional_l2_regularization_fn,
    "JFrobint": reg_lib.jacobian_frobenius_regularization_fn,
    "JdiagFrobint": reg_lib.jacobian_diag_frobenius_regularization_fn,
    "JoffdiagFrobint": reg_lib.jacobian_offdiag_frobenius_regularization_fn,
}

INV_REGULARIZATION_FNS = {v: k for k, v in six.iteritems(REGULARIZATION_FNS)}


def append_regularization_to_log(log_message, regularization_fns, reg_states):
    for i, reg_fn in enumerate(regularization_fns):
        log_message = log_message + " | " + INV_REGULARIZATION_FNS[reg_fn] + ": {:.8f}".format(reg_states[i].item())
    return log_message


def create_regularization_fns(args):
    regularization_fns = []
    regularization_coeffs = []

    for arg_key, reg_fn in six.iteritems(REGULARIZATION_FNS):
        if getattr(args, arg_key) is not None:
            regularization_fns.append(reg_fn)
            regularization_coeffs.append(eval("args." + arg_key))

    regularization_fns = tuple(regularization_fns)
    regularization_coeffs = tuple(regularization_coeffs)
    return regularization_fns, regularization_coeffs


def get_regularization(model, regularization_coeffs):
    if len(regularization_coeffs) == 0:
        return None

    acc_reg_states = tuple([0.] * len(regularization_coeffs))
    for module in model.modules():
        if isinstance(module, layers.CNF):
            acc_reg_states = tuple(acc + reg for acc, reg in zip(acc_reg_states, module.get_regularization_states()))
    return acc_reg_states


def build_model_tabular(args, dims, regularization_fns=None):

    hidden_dims = tuple(map(int, args.dims.split("-")))

    def build_cnf():
        diffeq = layers.ODEnet(
            hidden_dims=hidden_dims,
            input_shape=(dims,),
            strides=None,
            conv=False,
            layer_type=args.layer_type,
            nonlinearity=args.nonlinearity,
        )
        odefunc = layers.ODEfunc(
            diffeq=diffeq,
            divergence_fn=args.divergence_fn,
            residual=args.residual,
            rademacher=args.rademacher,
        )
        cnf = layers.CNF(
            odefunc=odefunc,
            T=args.time_length,
            train_T=args.train_T,
            solver=args.solver,
            atol = args.atol,
            rtol = args.rtol,
            test_atol = args.test_atol,
            test_rtol = args.test_rtol,
            poly_num_sample=args.poly_num_sample,
            poly_order=args.poly_order,
            adjoint=args.adjoint,
        )
        return cnf

    chain = [build_cnf() for _ in range(args.num_blocks)]
    if args.batch_norm:
        bn_layers = [layers.MovingBatchNorm1d(dims, bn_lag=args.bn_lag) for _ in range(args.num_blocks)]
        bn_chain = [layers.MovingBatchNorm1d(dims, bn_lag=args.bn_lag)]
        for a, b in zip(chain, bn_layers):
            bn_chain.append(a)
            bn_chain.append(b)
        chain = bn_chain[:-1] if args.ignore_last_bn else bn_chain
    model = layers.SequentialFlow(chain)

    set_cnf_options(args, model)

    return model



# +
def compute_bits_per_dim(x, model):
    zero = torch.zeros(x.shape[0], 1).to(x)
    lec = None if (args.poly_coef is None or not model.training) else torch.tensor(0.0).to(x)

    # Don't use data parallelize if batch size is small.
    # if x.shape[0] < 200:
    #     model = model.module

    z, delta_logp, lec = model(x, zero, lec)  # run model forward

    logpz = standard_normal_logprob(z).view(z.shape[0], -1).sum(1, keepdim=True)  # logp(z)
    logpx = logpz - delta_logp

    logpx_per_dim = torch.sum(logpx) / x.nelement()  # averaged over batches
    bits_per_dim = -(logpx_per_dim - np.log(256)) / np.log(2)
    lec = lec / (x[0].nelement() * np.log(2)) if lec else None

    return bits_per_dim, lec


def create_model(args, data_shape, regularization_fns):
    hidden_dims = tuple(map(int, args.dims.split("-")))
    strides = tuple(map(int, args.strides.split("-")))

    if args.multiscale:
        model = odenvp.ODENVP(
            (args.batch_size, *data_shape),
            n_blocks=args.num_blocks,
            intermediate_dims=hidden_dims,
            nonlinearity=args.nonlinearity,
            alpha=args.alpha,
            cnf_kwargs={"T": args.time_length,
                "train_T": args.train_T,
                "atol":args.atol,
                "rtol":args.rtol,
                "test_atol":args.test_atol,
                "test_rtol":args.test_rtol,
                "adjoint": args.adjoint,
                "poly_num_sample": args.poly_num_sample,
                "poly_order": args.poly_order},
        )
    elif args.parallel:
        model = multiscale_parallel.MultiscaleParallelCNF(
            (args.batch_size, *data_shape),
            n_blocks=args.num_blocks,
            intermediate_dims=hidden_dims,
            alpha=args.alpha,
            time_length=args.time_length,
        )
    else:
        raise NotImplementedError
    return model


# -

def update_integration_times(cnf:nn.Module, time:int):
    if "SequentialFlow" in str(type(cnf)):
        for chain in cnf.chain:
            if 'sqrt_end_time' in list(chain.named_parameters())[0]:
                chain.sqrt_end_time = nn.Parameter(torch.sqrt(torch.tensor(time)))
    elif "ODENVP" in str(type(cnf)):
        for StackedCNFLayers in cnf.transforms:
            for chain in StackedCNFLayers.chain:
                parameters = list(chain.named_parameters())
                if len(parameters) > 0:
                    if 'sqrt_end_time' in parameters[0]:
                        chain.sqrt_end_time = nn.Parameter(torch.sqrt(torch.tensor(time)))
    # cnf.sqrt_end_time = nn.Parameter(torch.sqrt(torch.tensor(time)))
    return cnf


def update_lr(epoch, optimizer, lr_change=None):
    if lr_change is None:
        return
    else:
        if epoch > lr_change:
            for param_group in optimizer.param_groups:
                param_group["lr"] = 1e-4
        else:
            pass


def forward_with_integration_times(cnf, x, logpx=None, lec=None, integration_times=None, reverse=False):
    n_chain = len(cnf.chain)
    return_last = (integration_times.shape[0] == 2)

    if reverse:
        inds = range(n_chain-1, -1, -1)
    else:
        inds = range(n_chain)
    
    x_all, logpx_all = [x], [logpx]
    for i in inds:
        name = str(type(cnf.chain[i]))
        if 'CNF' in  name:
            x, logpx, lec = cnf.chain[i](x, logpz=logpx, lec=lec, integration_times=integration_times, reverse=reverse)
            for _x, _logpx in zip(x[1:], logpx[1:]):
                x_all.append(_x)
                logpx_all.append(_logpx)
            x, logpx = x[-1], logpx[-1]
        else:
            x, logpx, lec = cnf.chain[i](x, logpx=logpx, lec=lec, reverse=reverse)
            
    if return_last:
        return x_all[-1], logpx_all[-1], lec
    else:
        return x_all, logpx_all, lec


def visualize_trajectory_forward(cnf, x, t0:int, time_step:float=1.0):
    result = [x]
    logpx = torch.zeros(x.shape[0], 1).to(x)
    while t0 > -1:
        time = torch.arange(t0, t0+1+time_step, time_step).to(x)
        print(time)
        with torch.no_grad():
            z, logpz, _ = forward_with_integration_times(cnf, x, logpx, integration_times=time)
        x, logpx = (z[-1], logpz[-1]) if isinstance(z, list) else (z, logpz) 
        result += z[1:] if isinstance(z, list) else [z]
        t0 -= 1
    return result


def visualize_trajectory_backward(cnf, z, logpz, t1:int, time_step:float=1.0):
    result = [z]
    t0 = 0
    while t0 < t1:
        time = torch.arange(t0, t0+1+time_step, time_step).to(z)
        print(time)
        with torch.no_grad():
            x, logpx, _ = forward_with_integration_times(cnf, z, logpz, integration_times=time, reverse=True)
        z, logpz = (x[-1], logpx[-1]) if isinstance(x, list) else (x, logpx) 
        result += x[1:] if isinstance(x, list) else [x]
        t0 += 1
    return result


def predict_target(cnf, prior, x_eval:np.ndarray, y_eval:np.ndarray, t0:int):
    dataset = util.preprocess_input(x_eval, y_eval)
    x, y = dataset.tensors
    x, y = util.torch_to(x, y)
    cnf.eval()
    z = visualize_trajectory_forward(cnf, x, t0)[-1]
    pred, prob = prior.predict(z)
    acc = accuracy_score(y_eval, pred)
    cnf.train()
    return prob, pred, acc


def generate_target(cnf, prior, n_sample:int, t1:int, time_step:float=1):
    z, logpz = prior.sample(n_sample)
    z, logpz = util.torch_to(z, logpz)
    pred, _ = prior.predict(z) if prior.__class__ == GaussianMixtureDA else (None, None)
    cnf.eval()
    x_hat = visualize_trajectory_backward(cnf, z, logpz, t1, time_step)
    cnf.train()
    return x_hat, pred
