import torch
from torch.optim.optimizer import Optimizer


# ============================================================
# Helpers for SIDG / SIDG-Delta with inverse-H L-BFGS approximation
# ============================================================

def _dot(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Euclidean inner product between two tensors/vectors.
    We flatten first so this works regardless of original shape.
    """
    return torch.dot(x.reshape(-1), y.reshape(-1))


def _numel_params(params) -> int:
    """
    Total number of scalar parameters in a list of tensors.
    Used to size the flattened velocity vector omega.
    """
    return sum(p.numel() for p in params)


def _params_list(param_groups):
    """
    Collect all trainable parameters from the optimizer parameter groups.
    """
    ps = []
    for g in param_groups:
        for p in g["params"]:
            if p.requires_grad:
                ps.append(p)
    if len(ps) == 0:
        raise RuntimeError("No trainable parameters.")
    return ps


@torch.no_grad()
def _get_flat_params(params) -> torch.Tensor:
    """
    Flatten all model parameters into a single vector theta.
    """
    return torch.cat([p.detach().reshape(-1) for p in params], dim=0)


@torch.no_grad()
def _set_flat_params(params, x: torch.Tensor) -> None:
    """
    Write a flattened parameter vector x back into the model tensors.
    """
    off = 0
    for p in params:
        n = p.numel()
        p.copy_(x[off:off + n].view_as(p))
        off += n


@torch.no_grad()
def _get_flat_grads(params) -> torch.Tensor:
    """
    Flatten all current parameter gradients into a single vector.
    If a parameter has no gradient, we insert zeros of the correct size.
    """
    grads = []
    for p in params:
        if p.grad is None:
            grads.append(torch.zeros_like(p).reshape(-1))
        else:
            grads.append(p.grad.detach().reshape(-1).clone())
    return torch.cat(grads, dim=0)


def _cg_solve(A_mv, b: torch.Tensor, x0: torch.Tensor = None,
              tol: float = 1e-6, max_iter: int = 25) -> torch.Tensor:
    """
    Conjugate-gradient solver for a linear system A x = b, where A is given
    implicitly through the callable A_mv(v) = A v.

    This is used because SIDG only needs matrix-vector products with the
    semi-implicit operator; we do not form the full matrix explicitly.
    """
    x = torch.zeros_like(b) if x0 is None else x0.clone()

    r = b - A_mv(x)
    p = r.clone()
    rs_old = _dot(r, r)

    if torch.sqrt(rs_old).item() <= tol:
        return x

    for _ in range(max_iter):
        Ap = A_mv(p)
        denom = _dot(p, Ap).clamp_min(1e-30)  
        alpha = rs_old / denom

        x = x + alpha * p
        r = r - alpha * Ap
        rs_new = _dot(r, r)

        if torch.sqrt(rs_new).item() <= tol:
            break

        beta = rs_new / rs_old.clamp_min(1e-30)
        p = r + beta * p
        rs_old = rs_new

    return x


def lbfgs_apply_H(v: torch.Tensor,
                  s_hist,
                  y_hist,
                  init_scale: float = 1.0,
                  eps: float = 1e-10) -> torch.Tensor:
    """
    Apply the limited-memory BFGS inverse-Hessian approximation H_k to a vector v
    using the standard two-loop recursion.

    Parameters
    ----------
    v : torch.Tensor
        Vector to which the inverse-Hessian approximation is applied.
    s_hist : list[torch.Tensor]
        History of s_k = theta_{k+1} - theta_k.
    y_hist : list[torch.Tensor]
        History of y_k = grad_{k+1} - grad_k.
    init_scale : float
        Initial scalar multiple of the identity used when no curvature
        information is available or when the most recent pair is unreliable.
    eps : float
        Curvature threshold. Pairs with s^T y <= eps are ignored.

    Notes
    -----
    - If there is no history, we return init_scale * v, i.e. H_k ≈ init_scale * I.
    - The curvature threshold is a practical safeguard; it does not by itself
      guarantee the uniform spectral bounds assumed in theory.
    """
    if len(s_hist) == 0:
        return init_scale * v

    q = v.clone()
    alpha_list = []
    rho_list = []

    # First loop: newest pair to oldest pair
    for s, y in zip(reversed(s_hist), reversed(y_hist)):
        sy = _dot(s, y)

        # Skip pairs with insufficient positive curvature
        if float(sy) <= eps:
            alpha_list.append(torch.tensor(0.0, device=v.device, dtype=v.dtype))
            rho_list.append(torch.tensor(0.0, device=v.device, dtype=v.dtype))
            continue

        rho = 1.0 / sy
        alpha = rho * _dot(s, q)
        q = q - alpha * y

        alpha_list.append(alpha)
        rho_list.append(rho)

    # Standard scalar initialization H0 = gamma I
    s_last = s_hist[-1]
    y_last = y_hist[-1]
    yy = _dot(y_last, y_last)
    sy = _dot(s_last, y_last)

    if float(sy) > eps and float(yy) > eps:
        gamma0 = sy / yy
    else:
        gamma0 = torch.as_tensor(init_scale, device=v.device, dtype=v.dtype)

    r = gamma0 * q

    # Second loop: oldest pair to newest pair
    for i, (s, y) in enumerate(zip(s_hist, y_hist)):
        alpha = alpha_list[len(s_hist) - 1 - i]
        rho = rho_list[len(s_hist) - 1 - i]

        if float(rho) == 0.0:
            continue

        beta = rho * _dot(y, r)
        r = r + s * (alpha - beta)

    return r


# ============================================================
# Shared base class for SIDG variants
# ============================================================

class _SIDGBase(Optimizer):
    """
    Shared utilities for SIDG methods using an inverse-Hessian L-BFGS operator.

    State stored across iterations:
    - omega_flat : flattened velocity variable
    - s_hist     : L-BFGS history for parameter differences
    - y_hist     : L-BFGS history for gradient differences
    """

    @torch.no_grad()
    def _get_state_buffers(self, ps):
        n = _numel_params(ps)
        if "omega_flat" not in self._global_state:
            self._global_state["omega_flat"] = torch.zeros(
                n, device=ps[0].device, dtype=ps[0].dtype
            )
            self._global_state["s_hist"] = []
            self._global_state["y_hist"] = []
        return (
            self._global_state["omega_flat"],
            self._global_state["s_hist"],
            self._global_state["y_hist"],
        )

    @torch.no_grad()
    def _update_lbfgs_history(self,
                              theta_k: torch.Tensor,
                              theta_next: torch.Tensor,
                              grad_k: torch.Tensor,
                              grad_new: torch.Tensor,
                              s_hist,
                              y_hist,
                              history_size: int,
                              curvature_eps: float) -> None:
        """
        Update the limited-memory curvature pairs (s_k, y_k) if the curvature
        condition s_k^T y_k > curvature_eps is satisfied.
        """
        s = theta_next - theta_k
        y = grad_new - grad_k

        if float(_dot(s, y)) > curvature_eps:
            s_hist.append(s.detach().clone())
            y_hist.append(y.detach().clone())

            if len(s_hist) > history_size:
                s_hist.pop(0)
                y_hist.pop(0)


# ============================================================
# SIDG 
# ============================================================

class SIDG(_SIDGBase):
    """
    Semi-Implicit Discrete Gradient (SIDG) method using an inverse-Hessian
    approximation H_k through an operator L-BFGS representation.
    """

    def __init__(self,
                 params,
                 h: float = 1e-1,
                 gamma: float = 0.9,
                 history_size: int = 10,
                 init_H_scale: float = 1.0,
                 cg_tol: float = 1e-6,
                 cg_max_iter: int = 25,
                 curvature_eps: float = 1e-8):
        defaults = dict(
            h=h,
            gamma=gamma,
            history_size=history_size,
            init_H_scale=init_H_scale,
            cg_tol=cg_tol,
            cg_max_iter=cg_max_iter,
            curvature_eps=curvature_eps,
        )
        super().__init__(params, defaults)
        self._global_state = {}

    @torch.no_grad()
    def step(self, closure=None):
        if closure is None:
            raise RuntimeError("SIDG_InvH requires a closure.")

        ps = _params_list(self.param_groups)
        g0 = self.param_groups[0]

        h = g0["h"]
        gamma = g0["gamma"]
        history_size = g0["history_size"]
        init_H_scale = g0["init_H_scale"]
        cg_tol = g0["cg_tol"]
        cg_max_iter = g0["cg_max_iter"]
        curvature_eps = g0["curvature_eps"]

        # Load optimizer state
        omega_buf, s_hist, y_hist = self._get_state_buffers(ps)

        # Current flattened state
        omega_k = omega_buf.detach().clone()
        theta_k = _get_flat_params(ps)

        # Compute gradient at current iterate theta_k
        with torch.enable_grad():
            _ = closure()
        grad_k = _get_flat_grads(ps)

        # Coefficients in the SIDG linear system
        c0 = 1.0 + 0.5 * h * gamma
        c1 = 1.0 - 0.5 * h * gamma
        beta = 0.25 * h * h

        # Inverse-Hessian operator H_k v
        def Hv(v):
            return lbfgs_apply_H(
                v,
                s_hist,
                y_hist,
                init_scale=init_H_scale,
                eps=curvature_eps,
            )

        # Left-hand side operator:
        # A(v) = ((1 + h*gamma/2) H_k + (h^2/4) I) v
        def Atilde(v):
            return c0 * Hv(v) + beta * v

        # Right-hand side of the semi-implicit velocity equation
        rhs = c1 * Hv(omega_k) - beta * omega_k - h * Hv(grad_k)

        # Solve for omega_{k+1} using conjugate gradient
        omega_next = _cg_solve(
            Atilde, rhs, x0=omega_k, tol=cg_tol, max_iter=cg_max_iter
        )

        # Midpoint-type parameter update
        theta_next = theta_k + 0.5 * h * (omega_k + omega_next)
        _set_flat_params(ps, theta_next)

        # Recompute gradient at theta_{k+1} for the next L-BFGS update
        with torch.enable_grad():
            loss_new = closure()
        grad_new = _get_flat_grads(ps)

        # Update limited-memory curvature pairs
        self._update_lbfgs_history(
            theta_k=theta_k,
            theta_next=theta_next,
            grad_k=grad_k,
            grad_new=grad_new,
            s_hist=s_hist,
            y_hist=y_hist,
            history_size=history_size,
            curvature_eps=curvature_eps,
        )

        # Store new velocity
        self._global_state["omega_flat"] = omega_next.detach().clone()

        return loss_new


# ============================================================
# SIDG-Delta 
# ============================================================

class SIDG_Delta(_SIDGBase):
    """
    Semi-Implicit Discrete Gradient with additional damping delta.
    """

    def __init__(self,
                 params,
                 h: float = 1e-1,
                 gamma: float = 0.3,
                 delta: float = 1e-3,
                 history_size: int = 10,
                 init_H_scale: float = 1.0,
                 cg_tol: float = 1e-6,
                 cg_max_iter: int = 25,
                 curvature_eps: float = 1e-8):
        defaults = dict(
            h=h,
            gamma=gamma,
            delta=delta,
            history_size=history_size,
            init_H_scale=init_H_scale,
            cg_tol=cg_tol,
            cg_max_iter=cg_max_iter,
            curvature_eps=curvature_eps,
        )
        super().__init__(params, defaults)
        self._global_state = {}

    @torch.no_grad()
    def step(self, closure=None):
        if closure is None:
            raise RuntimeError("SIDG_Delta_InvH requires a closure.")

        ps = _params_list(self.param_groups)
        g0 = self.param_groups[0]

        h = g0["h"]
        gamma = g0["gamma"]
        delta = g0["delta"]
        history_size = g0["history_size"]
        init_H_scale = g0["init_H_scale"]
        cg_tol = g0["cg_tol"]
        cg_max_iter = g0["cg_max_iter"]
        curvature_eps = g0["curvature_eps"]

        # Load optimizer state
        omega_buf, s_hist, y_hist = self._get_state_buffers(ps)

        # Current flattened state
        omega_k = omega_buf.detach().clone()
        theta_k = _get_flat_params(ps)

        # Compute gradient at current iterate theta_k
        with torch.enable_grad():
            _ = closure()
        grad_k = _get_flat_grads(ps)

        # Short-hand coefficients
        a = 0.5 * h * gamma
        d = 0.5 * h * delta
        c = 0.25 * h * h

        # Inverse-Hessian operator H_k v
        def Hv(v):
            return lbfgs_apply_H(
                v,
                s_hist,
                y_hist,
                init_scale=init_H_scale,
                eps=curvature_eps,
            )

        # Coefficients for the omega-system
        aI = 1.0 + a
        aB = (1.0 + a) * d + c
        rI = 1.0 - a
        rB = (1.0 - a) * d - c

        # Left-hand side operator for omega_{k+1}
        def Aomega(v):
            return aI * Hv(v) + aB * v

        # Right-hand side for omega_{k+1}
        rhs = rI * Hv(omega_k) + rB * omega_k - h * Hv(grad_k)

        # Solve for omega_{k+1}
        omega_next = _cg_solve(
            Aomega, rhs, x0=omega_k, tol=cg_tol, max_iter=cg_max_iter
        )

        # Linear system for the approximate discrete gradient gbar:
        # (H_k + (h*delta/2) I) gbar
        #     = H_k grad_k + (h/4)(omega_k + omega_next)
        def Mg(v):
            return Hv(v) + d * v

        rhs_g = Hv(grad_k) + 0.25 * h * (omega_k + omega_next)

        # Warm start from grad_k for the gbar solve
        gbar = _cg_solve(
            Mg, rhs_g, x0=grad_k, tol=cg_tol, max_iter=cg_max_iter
        )

        # Parameter update with additional delta damping
        theta_next = theta_k + 0.5 * h * (omega_k + omega_next) - h * delta * gbar
        _set_flat_params(ps, theta_next)

        # Recompute gradient at theta_{k+1} for the next L-BFGS update
        with torch.enable_grad():
            loss_new = closure()
        grad_new = _get_flat_grads(ps)

        # Update limited-memory curvature pairs
        self._update_lbfgs_history(
            theta_k=theta_k,
            theta_next=theta_next,
            grad_k=grad_k,
            grad_new=grad_new,
            s_hist=s_hist,
            y_hist=y_hist,
            history_size=history_size,
            curvature_eps=curvature_eps,
        )

        # Store new velocity
        self._global_state["omega_flat"] = omega_next.detach().clone()

        return loss_new
