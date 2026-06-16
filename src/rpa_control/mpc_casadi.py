"""CasADi-based Nonlinear MPC using multiple shooting + IPOPT.

Uses CVODES (SUNDIALS) for ODE integration inside the NLP — handles stiff
systems natively. The NLP is built once symbolically; at each MPC step we
just update the current state and re-solve with warm-starting.

Compared to mpc.py (scipy + finite differences):
  - Exact gradients via CasADi automatic differentiation
  - CVODES handles stiff systems (no RK4 stability limit)
  - IPOPT second-order convergence vs SLSQP first-order
  - Compiled NLP graph solved repeatedly → fast warm-starts
"""
import casadi as ca
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Tuple

from .collocation.models import CasadiODE


@dataclass
class CasadiMPCConfig:
    """Configuration for CasADi MPC."""

    prediction_horizon: int = 20    # N: number of control intervals
    dt: float = 1.0                 # Control interval length

    # Cost weights
    Q: np.ndarray = None            # State tracking weights (diagonal), shape (n_tracked,)
    Ru: float = 0.01                # Control magnitude weight
    R_deltau: float = 0.1           # Control rate-of-change weight
    u_ref: float = 1.0              # Reference control (no-treatment default)
    tracked_indices: list = None    # Which states to penalise (None = all)
    target_state: np.ndarray = None # Target state (default: zeros)

    # Control bounds
    u_min: float = 0.01
    u_max: float = 10.0

    # Collocation integrator options (used for NLP planning)
    collocation_elements: int = 10   # finite elements per control interval
    collocation_order: int = 3       # polynomial order (Radau points)

    # IPOPT options
    ipopt_max_iter: int = 100
    ipopt_tol: float = 1e-4
    ipopt_print_level: int = 0
    ipopt_opts: dict = field(default_factory=dict)

    # Objective type
    # 'tracking': minimize sum of stage costs (Q-weighted tracking error)
    # 'min_M':    minimize the minimum M achieved anywhere in the horizon,
    #             using a smooth log-sum-exp approximation:
    #             softmin_beta(M_1..M_N) = -1/beta * log(sum exp(-beta * M_k))
    #             → min_k M_k as beta → inf.  beta=20 is a good default.
    objective_type: str = 'tracking'
    min_M_state_index: int = 3   # which state index is M (for 'min_M' objective)
    min_M_beta: float = 20.0     # sharpness of the soft-min approximation

    # Warm-starting
    warm_start: bool = True


class CasadiMPCController:
    """Receding-horizon MPC controller using CasADi + IPOPT.

    The NLP is built once in __init__. Each call to step() solves it with
    the current state as a parameter, warm-started from the previous solution.
    """

    def __init__(self, model: CasadiODE, config: CasadiMPCConfig):
        self.model = model
        self.config = config
        self.nx = model.n_states
        self.nu = model.n_controls
        N = config.prediction_horizon

        tracked = config.tracked_indices if config.tracked_indices is not None \
            else list(range(self.nx))
        n_tracked = len(tracked)
        Q_diag = config.Q if config.Q is not None else np.ones(n_tracked)
        x_target = config.target_state if config.target_state is not None \
            else np.zeros(self.nx)

        # ── Build CVODES integrator ──────────────────────────────────────────
        x_sym = ca.MX.sym('x', self.nx)
        u_sym = ca.MX.sym('u', self.nu)
        xdot = model.dynamics(x_sym, u_sym)

        ode_def = {'x': x_sym, 'p': u_sym, 'ode': xdot}
        integ_opts = {
            'tf': config.dt,
            'number_of_finite_elements': config.collocation_elements,
            'interpolation_order': config.collocation_order,
        }
        F = ca.integrator('F', 'collocation', ode_def, integ_opts)

        # ── Build NLP symbolically ───────────────────────────────────────────
        U = ca.MX.sym('U', N * self.nu)
        X0 = ca.MX.sym('X0', self.nx)
        u_prev_sym = ca.MX.sym('u_prev', self.nu)
        p_sym = ca.vertcat(X0, u_prev_sym)

        # Propagate states and collect control penalties
        x_k = X0
        x_seq = []          # x_1 ... x_N (states after each interval)
        control_penalty = ca.MX(0)

        for k in range(N):
            u_k = U[k * self.nu:(k + 1) * self.nu]
            x_next = F(x0=x_k, p=u_k)['xf']
            x_seq.append(x_next)

            du_abs = u_k - config.u_ref
            control_penalty += config.Ru * ca.dot(du_abs, du_abs)

            u_km1 = U[(k - 1) * self.nu:k * self.nu] if k > 0 else u_prev_sym
            control_penalty += config.R_deltau * ca.dot(u_k - u_km1, u_k - u_km1)

            x_k = x_next

        if config.objective_type == 'min_M':
            # Smooth minimum via log-sum-exp:
            #   softmin_beta(M_1..M_N) = -1/beta * log( sum_k exp(-beta * M_k) )
            # As beta → inf this converges to min_k M_k.
            idx_M = config.min_M_state_index
            beta = config.min_M_beta
            M_vals = ca.vertcat(*[x_seq[k][idx_M] for k in range(N)])
            softmin = (-1.0 / beta) * ca.log(ca.sum1(ca.exp(-beta * M_vals)))
            J = softmin + control_penalty

            nlp = {'x': U, 'f': J, 'p': p_sym}
            self.lbx = np.full(N * self.nu, config.u_min)
            self.ubx = np.full(N * self.nu, config.u_max)
            self.lbg = np.array([])
            self.ubg = np.array([])
            self._has_constraints = False
            self._n_dec = N * self.nu

        else:  # 'tracking'
            J = control_penalty
            for k in range(N):
                e = x_seq[k][tracked] - x_target[tracked]
                J += ca.dot(Q_diag * e, e)

            nlp = {'x': U, 'f': J, 'p': p_sym}
            self.lbx = np.full(N * self.nu, config.u_min)
            self.ubx = np.full(N * self.nu, config.u_max)
            self.lbg = np.array([])
            self.ubg = np.array([])
            self._has_constraints = False
            self._n_dec = N * self.nu

        ipopt_opts = {
            'ipopt.max_iter': config.ipopt_max_iter,
            'ipopt.tol': config.ipopt_tol,
            'ipopt.print_level': config.ipopt_print_level,
            'ipopt.linear_solver': 'mumps',
            'print_time': False,
        }
        ipopt_opts.update(config.ipopt_opts)

        self.solver = ca.nlpsol('mpc', 'ipopt', nlp, ipopt_opts)

        # Warm-start guess
        self.u_guess = np.full(self._n_dec, config.u_ref)
        if config.objective_type == 'min_M':
            self.u_guess[-1] = 1.0   # initial guess for z: current M
        self.u_prev = np.full(self.nu, config.u_ref)
        self.N = N

    def step(self, x_current: np.ndarray) -> Tuple[np.ndarray, dict]:
        """Solve MPC for the current state and return the first control.

        Args:
            x_current: Current state (nx,)

        Returns:
            u_apply: Control to apply (nu,)
            info: Solver statistics
        """
        p_val = np.concatenate([x_current, self.u_prev])

        solve_kwargs = dict(x0=self.u_guess, p=p_val, lbx=self.lbx, ubx=self.ubx)
        if self._has_constraints:
            solve_kwargs['lbg'] = self.lbg
            solve_kwargs['ubg'] = self.ubg

        sol = self.solver(**solve_kwargs)

        dec_opt = sol['x'].full().flatten()
        stats = self.solver.stats()

        u_opt = dec_opt[:self.N * self.nu]
        u_apply = u_opt[:self.nu]
        self.u_prev = u_apply.copy()

        if self.config.warm_start:
            shifted = np.roll(u_opt.reshape(self.N, self.nu), -1, axis=0)
            shifted[-1] = u_opt[-self.nu:]
            self.u_guess[:self.N * self.nu] = shifted.flatten()
            if self.config.objective_type == 'min_M':
                self.u_guess[-1] = dec_opt[-1]  # warm-start z

        info = {
            'success': stats['success'],
            'return_status': stats['return_status'],
            'iter_count': stats['iter_count'],
            'cost': float(sol['f']),
        }
        return u_apply, info


def simulate_mpc_casadi(
    model: CasadiODE,
    mpc: CasadiMPCController,
    x0: np.ndarray,
    time_horizon: float,
    dt: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Simulate closed-loop system with CasADi MPC.

    Uses the same CVODES integrator as the MPC planner (perfect model).

    Returns:
        times:    (n_steps+1,)
        states:   (n_steps+1, nx)
        controls: (n_steps, nu)
    """
    # Build a single-step integrator for simulation
    nx = model.n_states
    nu = model.n_controls
    x_sym = ca.MX.sym('x', nx)
    u_sym = ca.MX.sym('u', nu)
    xdot = model.dynamics(x_sym, u_sym)
    ode_def = {'x': x_sym, 'p': u_sym, 'ode': xdot}
    F_sim = ca.integrator('F_sim', 'cvodes', ode_def, {
        'tf': dt,
        'abstol': 1e-8,
        'reltol': 1e-6,
        'max_num_steps': 5000,
    })

    n_steps = int(time_horizon / dt)
    times = [0.0]
    states = [x0.copy()]
    controls = []

    x_k = x0.copy()

    for step in range(n_steps):
        u_k, info = mpc.step(x_k)
        controls.append(u_k.copy())

        if step < 3:
            print(f"  Step {step}: M={x_k[3]:.4f}, u={u_k[0]:.4f}, "
                  f"cost={info['cost']:.4f}, iters={info['iter_count']}, "
                  f"status={info['return_status']}")

        x_next = F_sim(x0=x_k, p=u_k)['xf'].full().flatten()
        x_k = x_next
        times.append(times[-1] + dt)
        states.append(x_k.copy())

    return (
        np.array(times),
        np.array(states),
        np.array(controls),
    )
