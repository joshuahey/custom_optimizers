import math
import torch
from torch.optim.optimizer import Optimizer


# ============================================================
# Helper utilities
# ============================================================
def _params_list(param_groups):
    """
    Flatten all trainable parameters from the optimizer param groups
    into a single list.
    """
    ps = []
    for g in param_groups:
        for p in g["params"]:
            if p is not None and p.requires_grad:
                ps.append(p)
    return ps


@torch.no_grad()
def _get_params(ps):
    """
    Return a detached clone of the current parameter values.
    """
    return [p.detach().clone() for p in ps]


@torch.no_grad()
def _set_params(ps, theta):
    """
    Overwrite the model parameters with the tensors in theta.
    """
    for p, t in zip(ps, theta):
        p.copy_(t)


def _norm_list(xs):
    """
    Euclidean norm of a list of tensors.
    """
    s = 0.0
    for x in xs:
        s += float(x.detach().pow(2).sum().cpu())
    return math.sqrt(s)


@torch.no_grad()
def _zero_grads(ps):
    """
    Zero existing gradients on the parameter list.
    """
    for p in ps:
        if p.grad is not None:
            p.grad.zero_()


def _closure_grads(ps, closure):
    """
    Evaluate the closure, collect the gradients, and return them as
    a list of detached tensors.
    """
    _zero_grads(ps)
    with torch.enable_grad():
        loss = closure()

    grads = []
    for p in ps:
        if p.grad is None:
            grads.append(torch.zeros_like(p))
        else:
            grads.append(p.grad.detach().clone())
    return grads


def _gbar_avf(ps, closure, theta_k, theta_next, m=1):
    """
    Approximate the AVF discrete gradient

        gbar = ∫_0^1 ∇L((1-s) theta_k + s theta_next) ds

    by midpoint quadrature with m equally spaced samples.
    """
    gsum = [torch.zeros_like(p) for p in ps]

    for i in range(m):
        s = (i + 0.5) / m
        theta_s = [(1.0 - s) * tk + s * tn for tk, tn in zip(theta_k, theta_next)]
        _set_params(ps, theta_s)
        grads = _closure_grads(ps, closure)
        for j in range(len(ps)):
            gsum[j].add_(grads[j])

    return [g / float(m) for g in gsum]

# ============================================================
# DG implementation
# ============================================================
class DG(Optimizer):
    """
    Discrete-gradient optimizer using the AVF discrete gradient.
    """

    def __init__(self, params, h=5e-2, gamma=0.9, fp_iters=2, avf_samples=1, fp_tol=1e-2):
        super().__init__(
            params,
            dict(h=h, gamma=gamma, fp_iters=fp_iters, avf_samples=avf_samples, fp_tol=fp_tol),
        )

    @torch.no_grad()
    def step(self, closure=None):
        if closure is None:
            raise RuntimeError("DG requires closure.")

        ps = _params_list(self.param_groups)
        theta_k = _get_params(ps)

        omega_k = []
        for p in ps:
            st = self.state[p]
            if "omega" not in st:
                st["omega"] = torch.zeros_like(p)
            omega_k.append(st["omega"].detach().clone())

        g0 = self.param_groups[0]
        h, gamma = g0["h"], g0["gamma"]
        fp_iters, m, fp_tol = g0["fp_iters"], g0["avf_samples"], g0["fp_tol"]

        acoef = (2.0 - h * gamma) / (2.0 + h * gamma)
        bcoef = (2.0 * h) / (2.0 + h * gamma)

    
        theta_next = [tk + (2.0 * h / (2.0 + h * gamma)) * ok for tk, ok in zip(theta_k, omega_k)]
        omega_next = None

        for _ in range(fp_iters):
            gbar = _gbar_avf(ps, closure, theta_k, theta_next, m=m)

            omega_next = [acoef * ok - bcoef * gb for ok, gb in zip(omega_k, gbar)]

            theta_new = [
                tk + 0.5 * h * (ok + on)
                for tk, ok, on in zip(theta_k, omega_k, omega_next)
            ]

            if _norm_list([tnn - tn for tnn, tn in zip(theta_new, theta_next)]) <= fp_tol:
                theta_next = theta_new
                break

            theta_next = theta_new

        if omega_next is None:
            raise RuntimeError("DG: fp_iters must be at least 1.")

        _set_params(ps, theta_next)

        for p, on in zip(ps, omega_next):
            self.state[p]["omega"].copy_(on)

        with torch.enable_grad():
            loss = closure()

        return loss
# ============================================================
# DG-Delta implementation
# ============================================================
class DG_Delta(Optimizer):
    """
    Discrete-gradient-delta optimizer using the AVF discrete gradient.

    The update is implemented in the equivalent form
        omega_{k+1} = ((2 - h gamma)/(2 + h gamma)) omega_k
                      - (2h/(2 + h gamma)) gbar
        theta_{k+1} = theta_k + (h/2)(omega_k + omega_{k+1}) - h delta gbar
    """

    def __init__(self, params, h=5e-2, gamma=0.5, delta=1e-4, fp_iters=2, avf_samples=1, fp_tol=1e-2):
        super().__init__(
            params,
            dict(
                h=h,
                gamma=gamma,
                delta=delta,
                fp_iters=fp_iters,
                avf_samples=avf_samples,
                fp_tol=fp_tol,
            ),
        )

    @torch.no_grad()
    def step(self, closure=None):
        if closure is None:
            raise RuntimeError("DG_Delta requires closure.")

        ps = _params_list(self.param_groups)
        theta_k = _get_params(ps)

        omega_k = []
        for p in ps:
            st = self.state[p]
            if "omega" not in st:
                st["omega"] = torch.zeros_like(p)
            omega_k.append(st["omega"].detach().clone())

        g0 = self.param_groups[0]
        h, gamma, delta = g0["h"], g0["gamma"], g0["delta"]
        fp_iters, m, fp_tol = g0["fp_iters"], g0["avf_samples"], g0["fp_tol"]

        acoef = (2.0 - h * gamma) / (2.0 + h * gamma)
        bcoef = (2.0 * h) / (2.0 + h * gamma)


        theta_next = [tk + (2.0 * h / (2.0 + h * gamma)) * ok for tk, ok in zip(theta_k, omega_k)]
        omega_next = None

        for _ in range(fp_iters):
            gbar = _gbar_avf(ps, closure, theta_k, theta_next, m=m)

            omega_next = [acoef * ok - bcoef * gb for ok, gb in zip(omega_k, gbar)]

            theta_new = [
                tk + 0.5 * h * (ok + on) - h * delta * gb
                for tk, ok, on, gb in zip(theta_k, omega_k, omega_next, gbar)
            ]

            if _norm_list([tnn - tn for tnn, tn in zip(theta_new, theta_next)]) <= fp_tol:
                theta_next = theta_new
                break

            theta_next = theta_new

        if omega_next is None:
            raise RuntimeError("DG_Delta: fp_iters must be at least 1.")

        _set_params(ps, theta_next)

        for p, on in zip(ps, omega_next):
            self.state[p]["omega"].copy_(on)

        with torch.enable_grad():
            loss = closure()

        return loss
