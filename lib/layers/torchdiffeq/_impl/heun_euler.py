# Based on https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/integrate
import torch
from .misc import (
    _scaled_dot_product, _convert_to_tensor, _is_finite, _select_initial_step, _handle_unused_kwargs, _is_iterable,
    _optimal_step_size, _compute_error_ratio
)
from .solvers import AdaptiveStepsizeODESolver
from .rk_common import _RungeKuttaState, _ButcherTableau, _runge_kutta_step

_TABLEAU = _ButcherTableau(
    alpha=[1.],
    beta=[
        [1.],
    ],
    c_sol=[1/2, 1/2],
    c_error=[
        -1/2,
        1/2
    ],
)

_TABLEAU = _ButcherTableau(
    alpha=[1/2, 1.],
    beta=[
        [1/2],
        [1/256, 255/256],
    ],
    c_sol=[1/256, 255/256],
    c_error=[
        1/256 - 1/512,
        0,
         - 1/512,
    ],
)



def _interp_evaluate(t0, y0, t1, y1, t):
    """Fit an interpolating polynomial to the results of a Runge-Kutta step."""
    t0 = t0.type_as(y0[0])
    t1 = t1.type_as(y0[0])
    t = t.type_as(y0[0])
    dt = t1-t0
    return tuple(y0_*((t1-t)/dt) + y1_*((t-t0)/dt) for y0_, y1_ in zip(y0, y1) )


#def _abs_square(x):
#    return torch.mul(x, x)
#
#
#def _ta_append(list_of_tensors, value):
#    """Append a value to the end of a list of PyTorch tensors."""
#    list_of_tensors.append(value)
#    return list_of_tensors


class HeunEuler(AdaptiveStepsizeODESolver):

    def __init__(
        self, func, y0, rtol, atol, first_step=None, safety=0.9, ifactor=10.0, dfactor=0.2, max_num_steps=2**31 - 1,
        **unused_kwargs
    ):
        _handle_unused_kwargs(self, unused_kwargs)
        del unused_kwargs

        self.func = func
        self.y0 = y0
        self.rtol = rtol if _is_iterable(rtol) else [rtol] * len(y0)
        self.atol = atol if _is_iterable(atol) else [atol] * len(y0)
        self.next_step = first_step
        self.safety = _convert_to_tensor(safety, dtype=torch.float64, device=y0[0].device)
        self.ifactor = _convert_to_tensor(ifactor, dtype=torch.float64, device=y0[0].device)
        self.dfactor = _convert_to_tensor(dfactor, dtype=torch.float64, device=y0[0].device)
        self.max_num_steps = _convert_to_tensor(max_num_steps, dtype=torch.int32, device=y0[0].device)

    def before_integrate(self, t):
        f0 = self.func(t[0].type_as(self.y0[0]), self.y0)
        if self.next_step is None:
            first_step = _select_initial_step(self.func, t[0], self.y0, 1, self.rtol[0], self.atol[0], f0=f0).to(t)
        else:
            first_step = self.next_step
        self.rk_state = _RungeKuttaState(self.y0, f0, t[0], t[0], first_step, None)

    def advance(self, next_t):
        """Interpolate through the next time point, integrating as necessary."""
        n_steps = 0
        while next_t > self.rk_state.t1:
            assert n_steps < self.max_num_steps, 'max_num_steps exceeded ({}>={})'.format(n_steps, self.max_num_steps)
            #print(self.rk_state.dt)
            self.rk_state = self._adaptive_dopri5_step(self.rk_state)
            n_steps += 1
        self.next_step = self.rk_state.dt
        return _interp_evaluate(self.rk_state.t0, self.rk_state.interp_coeff, self.rk_state.t1, self.rk_state.y1, next_t)

    def _adaptive_dopri5_step(self, rk_state):
        """Take an adaptive Runge-Kutta step to integrate the ODE."""
        y0, f0, _, t0, dt, interp_coeff = rk_state
        #print(dt)
        ########################################################
        #                      Assertions                      #
        ########################################################
        assert t0 + dt > t0, 'underflow in dt {}'.format(dt.item())
        for y0_ in y0:
            assert _is_finite(torch.abs(y0_)), 'non-finite values in state `y`: {}'.format(y0_)
        y1, f1, y1_error, k = _runge_kutta_step(self.func, y0, f0, t0, dt, tableau=_TABLEAU)

        ########################################################
        #                     Error Ratio                      #
        ########################################################
        mean_sq_error_ratio = _compute_error_ratio(y1_error, atol=self.atol, rtol=self.rtol, y0=y0, y1=y1)
        accept_step = (torch.tensor(mean_sq_error_ratio) <= 1).all()

        ########################################################
        #                   Update RK State                    #
        ########################################################
        y_next = y1 if accept_step else y0
        f_next = f1 if accept_step else f0
        t_next = t0 + dt if accept_step else t0
        interp_coeff = y0 if accept_step else interp_coeff
        dt_next = _optimal_step_size(
            dt, mean_sq_error_ratio, safety=self.safety, ifactor=self.ifactor, dfactor=self.dfactor, order=2
        )
        rk_state = _RungeKuttaState(y_next, f_next, t0, t_next, dt_next, interp_coeff)
        return rk_state
