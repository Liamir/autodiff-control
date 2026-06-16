"""Direct collocation optimal control solver using CasADi + IPOPT."""
import casadi as ca
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Callable
from .models import CasadiODE


@dataclass
class CollocationConfig:
    """Configuration for direct collocation optimal control."""

    # Time grid
    N: int = 100                    # Number of finite elements
    T: float = 50.0                 # Total time horizon
    d: int = 3                      # Collocation degree (Radau points)

    # Cost weights
    Q: np.ndarray = None            # State tracking weight matrix (n_tracked x n_tracked)
    R: np.ndarray = None            # Control effort weight matrix (n_controls x n_controls)
    target_state: np.ndarray = None # Target state for tracking
    tracked_indices: list = None    # Which states to track (None = all)
    u_ref: np.ndarray = None        # Reference control (null action). Penalizes |u - u_ref|.

    # Bounds
    u_min: np.ndarray = None        # Lower bound per control
    u_max: np.ndarray = None        # Upper bound per control
    x_min: np.ndarray = None        # State lower bounds
    x_max: np.ndarray = None        # State upper bounds

    # Initial state
    x0: np.ndarray = None

    # External inputs
    external_input_fn: Callable = None  # f(t) -> np.ndarray

    # IPOPT options
    ipopt_opts: dict = field(default_factory=dict)


def _collocation_matrices(d: int):
    """Compute collocation matrices for Legendre-Gauss-Radau points.

    Args:
        d: Number of collocation points (polynomial degree).

    Returns:
        C: Differentiation matrix (d+1 x d+1).
           C[j, i] = L'_i(tau_j) for j=1..d (collocation points), i=0..d
        D: Quadrature weights (d+1,)
        B: Continuity coefficients - Lagrange polynomials evaluated at tau=1 (d+1,)
        tau: Collocation points including 0, shape (d+1,)
    """
    # Collocation points: 0 + Radau points in (0, 1]
    tau_root = [0] + list(ca.collocation_points(d, 'radau'))
    tau = np.array(tau_root)

    # Build Lagrange polynomials
    # L_i(t) = prod_{j!=i} (t - tau_j) / (tau_i - tau_j)
    C = np.zeros((d + 1, d + 1))
    D = np.zeros(d + 1)
    B = np.zeros(d + 1)

    for i in range(d + 1):
        # Construct Lagrange polynomial L_i
        poly = np.polynomial.polynomial.Polynomial([1.0])
        for j in range(d + 1):
            if j != i:
                # Multiply by (t - tau_j) / (tau_i - tau_j)
                factor = np.polynomial.polynomial.Polynomial(
                    [-tau[j], 1.0]
                ) / (tau[i] - tau[j])
                poly = poly * factor

        # Derivative of L_i evaluated at collocation points
        dpoly = poly.deriv()
        for j in range(d + 1):
            C[j, i] = dpoly(tau[j])

        # Quadrature weight: integral of L_i over [0, 1]
        poly_int = poly.integ()
        D[i] = poly_int(1.0) - poly_int(0.0)

        # Continuity: L_i(1)
        B[i] = poly(1.0)

    return C, D, B, tau


def solve_optimal_control(model: CasadiODE, config: CollocationConfig) -> dict:
    """Solve open-loop optimal control via direct collocation with IPOPT.

    Args:
        model: CasADi ODE model.
        config: Collocation configuration.

    Returns:
        Dictionary with:
            t: Element boundary times (N+1,)
            x: States at element boundaries (N+1, n_states)
            u: Controls per element (N, n_controls)
            t_col: All collocation point times (N*d,)
            x_col: States at collocation points (N*d, n_states)
            cost: Optimal cost
            stats: IPOPT solver statistics
    """
    N = config.N
    T = config.T
    d = config.d
    nx = model.n_states
    nu = model.n_controls
    h = T / N  # Element width

    # Collocation matrices
    C, D, B, tau = _collocation_matrices(d)

    # Determine tracked indices
    tracked = config.tracked_indices if config.tracked_indices is not None else list(range(nx))
    n_tracked = len(tracked)

    # Default cost matrices
    Q = config.Q if config.Q is not None else np.eye(n_tracked)
    R = config.R if config.R is not None else np.zeros((nu, nu))
    x_target = config.target_state if config.target_state is not None else np.zeros(nx)
    u_ref = config.u_ref if config.u_ref is not None else np.zeros(nu)

    # Decision variables and constraints
    w = []      # Decision variables
    w0 = []     # Initial guess
    lbw = []    # Lower bounds
    ubw = []    # Upper bounds
    g = []      # Constraints
    lbg = []    # Constraint lower bounds
    ubg = []    # Constraint upper bounds
    J = 0       # Objective

    # State bounds (default: unbounded)
    x_lb = config.x_min if config.x_min is not None else -np.inf * np.ones(nx)
    x_ub = config.x_max if config.x_max is not None else np.inf * np.ones(nx)

    # Control bounds (default: unbounded)
    u_lb = config.u_min if config.u_min is not None else -np.inf * np.ones(nu)
    u_ub = config.u_max if config.u_max is not None else np.inf * np.ones(nu)

    # Default control initial guess: midpoint of bounds (or 1.0 if unbounded)
    u_init = np.where(
        np.isfinite(u_lb) & np.isfinite(u_ub),
        (u_lb + u_ub) / 2,
        np.ones(nu)
    )

    # Initial state at t=0
    X0 = ca.MX.sym('X0', nx)
    w.append(X0)
    w0.append(config.x0)
    lbw.append(config.x0)  # Fixed initial condition
    ubw.append(config.x0)

    Xk = X0  # State at start of current element

    for k in range(N):
        t_k = k * h  # Time at start of element

        # Control for this element (piecewise constant)
        Uk = ca.MX.sym(f'U_{k}', nu)
        w.append(Uk)
        w0.append(u_init)
        lbw.append(u_lb)
        ubw.append(u_ub)

        # External input at element midpoint
        if config.external_input_fn is not None:
            p_k = config.external_input_fn(t_k + h / 2)
            if not isinstance(p_k, np.ndarray):
                p_k = np.array([p_k])
        else:
            p_k = None

        # State at collocation points
        Xc = []
        for j in range(1, d + 1):
            Xkj = ca.MX.sym(f'X_{k}_{j}', nx)
            w.append(Xkj)
            w0.append(config.x0)  # Initial guess: initial state
            lbw.append(x_lb)
            ubw.append(x_ub)
            Xc.append(Xkj)

        # Collocation equations
        for j in range(1, d + 1):
            # State derivative approximation from polynomial
            xp = C[j, 0] * Xk
            for r in range(d):
                xp += C[j, r + 1] * Xc[r]

            # ODE right-hand side at collocation point
            if p_k is not None:
                fj = model.dynamics(Xc[j - 1], Uk, p_k)
            else:
                fj = model.dynamics(Xc[j - 1], Uk)

            # Collocation constraint: polynomial derivative = h * dynamics
            g.append(h * fj - xp)
            lbg.append(np.zeros(nx))
            ubg.append(np.zeros(nx))

        # Stage cost at collocation points (quadrature)
        for j in range(1, d + 1):
            # Tracking error
            e = Xc[j - 1][tracked] - x_target[tracked]
            state_cost = ca.mtimes([e.T, Q, e])

            # Control effort
            du = Uk - u_ref
            control_cost = ca.mtimes([du.T, R, du])

            # Weighted by quadrature weight and element size
            J += D[j] * h * (state_cost + control_cost)

        # Continuity: state at end of element
        Xk_end = B[0] * Xk
        for r in range(d):
            Xk_end += B[r + 1] * Xc[r]

        # New element boundary state
        Xk_next = ca.MX.sym(f'X_{k + 1}', nx)
        w.append(Xk_next)
        w0.append(config.x0)
        lbw.append(x_lb)
        ubw.append(x_ub)

        # Continuity constraint
        g.append(Xk_next - Xk_end)
        lbg.append(np.zeros(nx))
        ubg.append(np.zeros(nx))

        Xk = Xk_next

    # Build NLP
    nlp = {
        'x': ca.vertcat(*w),
        'f': J,
        'g': ca.vertcat(*g),
    }

    # IPOPT options
    opts = {
        'ipopt.max_iter': 3000,
        'ipopt.tol': 1e-6,
        'ipopt.print_level': 5,
        'ipopt.linear_solver': 'mumps',
        'ipopt.mu_strategy': 'adaptive',
        'print_time': False,
    }
    for key, val in config.ipopt_opts.items():
        if '.' not in key:
            opts[f'ipopt.{key}'] = val
        else:
            opts[key] = val

    solver = ca.nlpsol('solver', 'ipopt', nlp, opts)

    # Solve
    sol = solver(
        x0=np.concatenate(w0),
        lbx=np.concatenate(lbw),
        ubx=np.concatenate(ubw),
        lbg=np.concatenate(lbg),
        ubg=np.concatenate(ubg),
    )

    # Extract solution
    w_opt = sol['x'].full().flatten()
    stats = solver.stats()

    # Parse decision variables
    t_grid = np.linspace(0, T, N + 1)
    x_opt = np.zeros((N + 1, nx))
    u_opt = np.zeros((N, nu))
    x_col = np.zeros((N * d, nx))
    t_col = np.zeros(N * d)

    idx = 0
    # Initial state
    x_opt[0] = w_opt[idx:idx + nx]
    idx += nx

    for k in range(N):
        # Control
        u_opt[k] = w_opt[idx:idx + nu]
        idx += nu

        # Collocation states
        for j in range(d):
            x_col[k * d + j] = w_opt[idx:idx + nx]
            t_col[k * d + j] = t_grid[k] + tau[j + 1] * h
            idx += nx

        # Next element boundary state
        x_opt[k + 1] = w_opt[idx:idx + nx]
        idx += nx

    return {
        't': t_grid,
        'x': x_opt,
        'u': u_opt,
        't_col': t_col,
        'x_col': x_col,
        'cost': float(sol['f']),
        'stats': {
            'success': stats['success'],
            'return_status': stats['return_status'],
            'iter_count': stats['iter_count'],
            't_wall_total': stats.get('t_wall_total', None),
        },
    }
