"""Nonlinear Model Predictive Control (NMPC) for ODE systems.

Uses scipy.optimize with RK4 integration based on the SINDY-MPC paper approach.

Supports generic control interface: ODEs that accept control via the `control` parameter
in their forward() method. This works with both additive and nonlinear control coupling.

Supports two cost function modes:
1. Reference tracking (default): Minimize tracking error with Q, Ru, R_deltau weights
2. Custom stage cost: User-defined cost function stage_cost_fn(x_next, u, x_curr, k)

For legacy additive control (PopulationDynamics), see mpc_additive.py.
"""
import torch
import numpy as np
import inspect
from typing import Optional, Tuple
from scipy.optimize import minimize


class MPCConfig:
    """Configuration for Model Predictive Control."""

    def __init__(
        self,
        prediction_horizon: int = 10,
        dt: float = 0.1,
        Q: Optional[torch.Tensor] = None,
        Ru: Optional[float] = None,
        R_deltau: Optional[float] = None,
        u_min: float = -20.0,
        u_max: float = 20.0,
        state_min: Optional[torch.Tensor] = None,
        state_max: Optional[torch.Tensor] = None,
        ftol: float = 1e-3,
        disp: bool = False,
        cost_type: str = 'quadratic',
        tracked_state_indices: Optional[list[int]] = None,
        stage_cost_fn: Optional[callable] = None,
        solver: str = 'slsqp',
        warm_start: bool = True,
        integration_substeps: int = 1,
        integration_method: str = 'rk4',
    ):
        """Initialize MPC configuration.

        Args:
            prediction_horizon: Number of steps to predict ahead
            dt: Time step (control grid spacing)
            Q: State tracking weight (diagonal of Q matrix for quadratic, or vector for L1)
            Ru: Control magnitude weight
            R_deltau: Control rate-of-change weight
            u_min: Minimum control input
            u_max: Maximum control input
            state_min: Minimum state values (optional)
            state_max: Maximum state values (optional)
            ftol: Optimization tolerance
            disp: Display optimization progress
            cost_type: Type of cost function ('quadratic' or 'l1')
            tracked_state_indices: Indices of states to track (None = track all states)
            stage_cost_fn: Optional custom stage cost function with signature:
                           stage_cost_fn(x_next, u, x_curr=None, k=None) -> float
                           If None, uses reference tracking cost (Q, Ru, R_deltau)
            solver: Optimization solver ('slsqp' or 'ipopt')
            warm_start: Use previous solution to initialize next optimization (default: True)
            integration_substeps: Number of RK4 sub-steps per control interval (default: 1).
                                  Higher values improve integration accuracy without adding
                                  decision variables. Use when dt is too coarse for stable
                                  integration but the control grid is fine.
            integration_method: Integration method for the inner ODE ('rk4' or 'scipy').
                                 Use 'scipy' for stiff systems — it uses LSODA via
                                 scipy.integrate.odeint, which handles stiff ODEs robustly.
        """
        self.prediction_horizon = prediction_horizon
        self.dt = dt
        self.Q = Q if Q is not None else torch.tensor([1.0, 1.0])
        self.Ru = Ru if Ru is not None else 0.5
        self.R_deltau = R_deltau if R_deltau is not None else 0.5
        self.u_min = u_min
        self.u_max = u_max
        self.state_min = state_min
        self.state_max = state_max
        self.ftol = ftol
        self.disp = disp
        self.cost_type = cost_type
        self.tracked_state_indices = tracked_state_indices
        self.stage_cost_fn = stage_cost_fn
        self.solver = solver.lower()
        self.warm_start = warm_start
        self.integration_substeps = integration_substeps
        self.integration_method = integration_method


class MPCController:
    """Nonlinear Model Predictive Control using scipy.optimize.

    Uses generic control interface: passes control to ODE via `control` parameter.
    Works with both additive and nonlinear control coupling.

    Example with custom cost function:
    >>> def my_stage_cost(x_next, u, x_curr=None, k=None):
    ...     '''Custom cost: maximize |B| while penalizing large control'''
    ...     B = x_next[1]  # Second state variable
    ...     reward = torch.abs(B)  # Maximize |B|
    ...     cost = -reward + 0.1 * (u**2)  # Minimize cost = maximize reward
    ...     return cost
    >>>
    >>> config = MPCConfig(
    ...     prediction_horizon=20,
    ...     dt=0.1,
    ...     stage_cost_fn=my_stage_cost,  # Use custom cost
    ...     Ru=0.0,  # Not used (already in stage_cost_fn)
    ...     R_deltau=0.1,  # Control smoothness (always applied)
    ...     u_min=0.2,
    ...     u_max=1.0,
    ... )
    >>> mpc = MPCController(ode, config, reference_state=None)  # No reference needed
    """

    def __init__(
        self,
        ode: torch.nn.Module,
        config: MPCConfig,
        reference_state: torch.Tensor = None,
        n_controls: int = 1,
    ):
        """Initialize MPC controller.

        Args:
            ode: ODE system that accepts `control` parameter in forward()
            config: MPC configuration
            reference_state: Target state to track (required if not using custom stage_cost_fn)
            n_controls: Number of control inputs (default: 1)
        """
        self.ode = ode
        self.config = config
        self.n_controls = n_controls

        # Custom stage cost function (if provided)
        self.stage_cost_fn = config.stage_cost_fn

        # Reference state (required only for default reference tracking)
        if self.stage_cost_fn is None and reference_state is None:
            raise ValueError("reference_state is required when not using custom stage_cost_fn")

        if reference_state is not None:
            self.reference_state = reference_state.detach().numpy()
            self.n_states = len(reference_state)
        else:
            self.reference_state = None
            # Will infer n_states from first control step

        # Verify ODE supports control parameter
        forward_sig = inspect.signature(ode.forward)
        if 'control' not in forward_sig.parameters:
            raise ValueError(
                f"ODE {type(ode).__name__} does not accept 'control' parameter. "
                "For additive control ODEs, use mpc_additive.py instead."
            )

        # Convert Q to weights (only needed if using default cost)
        if self.stage_cost_fn is None:
            Q_weights = config.Q.numpy() if isinstance(config.Q, torch.Tensor) else np.array(config.Q)
            if config.cost_type == 'quadratic':
                self.Q = np.diag(Q_weights)  # Diagonal matrix for quadratic cost
            else:  # L1 cost
                self.Q = Q_weights  # Vector of weights for L1 cost

            self.Ru = config.Ru
            self.R_deltau = config.R_deltau
            self.cost_type = config.cost_type
            self.tracked_state_indices = config.tracked_state_indices
        else:
            # Not needed for custom cost
            self.Q = None
            self.Ru = config.Ru  # Still might be used for control regularization
            self.R_deltau = config.R_deltau  # Still might be used for smoothness
            self.cost_type = None
            self.tracked_state_indices = None

        # Previous control for warm-starting (ones for multi-control since
        # multipliers default to 1.0; zero for legacy single-control)
        self.u_prev = np.ones(n_controls) if n_controls > 1 else 0.0
        self.u_guess = None

    def _ode_dynamics(self, x: np.ndarray, u) -> np.ndarray:
        """Evaluate ODE dynamics: dx/dt = f(x, u)

        Uses generic control interface: passes control to ODE.forward().

        Args:
            x: State vector
            u: Control input (scalar or array)

        Returns:
            State derivative
        """
        x_torch = torch.tensor(x, dtype=torch.float32)

        # Convert control to tensor
        if np.isscalar(u):
            u_torch = torch.tensor([u], dtype=torch.float32)
        else:
            u_torch = torch.tensor(np.asarray(u), dtype=torch.float32)

        with torch.no_grad():
            # Get dynamics with control
            dx = self.ode(
                torch.tensor(0.0),
                x_torch,
                control=u_torch
            ).numpy()

        return dx

    def _rk4_step(self, x: np.ndarray, u, dt: float) -> np.ndarray:
        """Integrate one control interval, dispatching to RK4 or scipy."""
        if self.config.integration_method == 'scipy':
            return self._scipy_step(x, u, dt)
        n_sub = self.config.integration_substeps
        sub_dt = dt / n_sub
        for _ in range(n_sub):
            k1 = self._ode_dynamics(x, u)
            k2 = self._ode_dynamics(x + 0.5 * sub_dt * k1, u)
            k3 = self._ode_dynamics(x + 0.5 * sub_dt * k2, u)
            k4 = self._ode_dynamics(x + sub_dt * k3, u)
            x = x + (sub_dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        return x

    def _scipy_step(self, x: np.ndarray, u, dt: float) -> np.ndarray:
        """Integrate one control interval with scipy.integrate.odeint (LSODA).

        LSODA automatically switches between stiff and non-stiff methods,
        making it suitable for systems with fast and slow timescales.
        mxstep is set high to handle systems with very different timescales
        (e.g., stiffness ratio ~1000).
        """
        from scipy.integrate import odeint

        def rhs(y, t):
            return self._ode_dynamics(y, u)

        sol = odeint(rhs, x, [0.0, dt], rtol=1e-6, atol=1e-8, mxstep=5000)
        return sol[-1]

    def _mpc_objective(self, u_seq: np.ndarray, x0: np.ndarray) -> float:
        """MPC cost function.

        Args:
            u_seq: Control sequence (flattened, length N * n_controls)
            x0: Initial state

        Returns:
            Total cost
        """
        N = self.config.prediction_horizon
        nc = self.n_controls
        dt = self.config.dt

        u_seq = u_seq.reshape(N, nc)

        cost = 0.0
        x_curr = x0.copy()

        for k in range(N):
            u_k = u_seq[k] if nc > 1 else u_seq[k, 0]

            # Predict next state
            x_next = self._rk4_step(x_curr, u_k, dt)

            # ===== Custom stage cost function =====
            if self.stage_cost_fn is not None:
                # Use custom cost function
                # Convert to torch tensors for the cost function
                x_next_torch = torch.tensor(x_next, dtype=torch.float32)
                x_curr_torch = torch.tensor(x_curr, dtype=torch.float32)
                u_k_torch = torch.tensor(u_k, dtype=torch.float32)

                # Call custom stage cost
                # Try to call with different signatures for flexibility
                try:
                    stage_cost = self.stage_cost_fn(x_next_torch, u_k_torch, x_curr=x_curr_torch, k=k)
                except TypeError:
                    # Try simpler signature
                    try:
                        stage_cost = self.stage_cost_fn(x_next_torch, u_k_torch)
                    except TypeError:
                        # Try state-only signature (for control-free costs)
                        stage_cost = self.stage_cost_fn(x_next_torch)

                # Convert to scalar
                if torch.is_tensor(stage_cost):
                    stage_cost = stage_cost.item()

            # ===== Default reference tracking cost =====
            else:
                # State tracking cost
                error = x_next - self.reference_state

                if self.cost_type == 'l1':
                    # L1 cost: sum of absolute values
                    if self.tracked_state_indices is not None:
                        # Only track specific states
                        tracked_error = error[self.tracked_state_indices]
                        weighted_error = self.Q * np.abs(tracked_error)
                        state_cost = np.sum(weighted_error)
                    else:
                        # Track all states
                        weighted_error = self.Q * np.abs(error)
                        state_cost = np.sum(weighted_error)
                else:  # quadratic cost
                    # Quadratic cost: error^T @ Q @ error
                    state_cost = error.T @ self.Q @ error

                # Control magnitude cost
                u_k_arr = np.atleast_1d(u_k)
                if self.cost_type == 'l1':
                    control_magnitude_cost = self.Ru * np.sum(np.abs(u_k_arr))
                else:
                    control_magnitude_cost = self.Ru * np.sum(u_k_arr**2)

                stage_cost = state_cost + control_magnitude_cost

            # Control rate-of-change cost (always applied for smoothness)
            if k == 0:
                du = u_seq[k] - self.u_prev
            else:
                du = u_seq[k] - u_seq[k-1]
            control_change_cost = self.R_deltau * np.sum(du**2)

            cost += stage_cost + control_change_cost

            x_curr = x_next

        return cost

    def step(self, x_current: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """Compute MPC control action (receding horizon).

        Args:
            x_current: Current state

        Returns:
            Control input to apply (shape: (n_controls,))
            Info dictionary
        """
        N = self.config.prediction_horizon
        nc = self.n_controls
        x0_np = x_current.detach().numpy()

        # Initial guess (N * nc decision variables)
        if self.u_guess is None or not self.config.warm_start:
            self.u_guess = np.ones(N * nc)

        # Bounds for each decision variable
        bounds = [(self.config.u_min, self.config.u_max) for _ in range(N * nc)]

        # Optimize
        if self.config.solver == 'ipopt':
            from cyipopt import minimize_ipopt
            res = minimize_ipopt(
                self._mpc_objective,
                self.u_guess,
                args=(x0_np,),
                bounds=bounds,
                options={
                    'tol': self.config.ftol,
                    'maxiter': 100,
                    'print_level': 5 if self.config.disp else 0,
                    'hessian_approximation': 'limited-memory',
                }
            )
        else:
            res = minimize(
                self._mpc_objective,
                self.u_guess,
                args=(x0_np,),
                method='SLSQP',
                bounds=bounds,
                options={
                    'ftol': self.config.ftol,
                    'disp': self.config.disp,
                    'maxiter': 100,
                    'eps': 1e-4,  # Larger step for gradient estimation to survive float32 conversion
                }
            )

        u_optimal = res.x.reshape(N, nc)
        u_control = u_optimal[0]  # First timestep control (shape: (nc,))

        # Update for next iteration (warm start: shift sequence by one timestep)
        self.u_prev = u_control.copy()
        u_shifted = np.roll(u_optimal, -1, axis=0)
        u_shifted[-1] = u_optimal[-1]
        self.u_guess = u_shifted.flatten()

        info = {
            'success': res.success,
            'cost': float(res.fun),
            'message': res.message,
            'nit': res.nit,
        }

        return torch.tensor(u_control, dtype=torch.float32), info


def simulate_mpc(
    ode: torch.nn.Module,
    mpc_controller: MPCController,
    x0: torch.Tensor,
    time_horizon: float,
    dt: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Simulate closed-loop system with MPC control.

    Args:
        ode: ODE system
        mpc_controller: MPC controller
        x0: Initial state
        time_horizon: Total simulation time
        dt: Time step

    Returns:
        times: Time points
        states: State trajectory
        controls: Control inputs
    """
    n_steps = int(time_horizon / dt)
    times = [0.0]
    states = [x0.detach().numpy()]
    controls = []

    x_current = x0.clone()

    for step in range(n_steps):
        # Compute MPC control
        u_mpc, info = mpc_controller.step(x_current)
        controls.append(u_mpc.numpy())

        if step < 3:
            print(f"  Step {step}: state={x_current.tolist()}, u={u_mpc.tolist()}, "
                  f"cost={info['cost']:.6f}, nit={info['nit']}, success={info['success']}")

        # Apply control and simulate one step
        x_current_np = x_current.detach().numpy()
        if mpc_controller.n_controls == 1:
            u_apply = float(u_mpc[0])
        else:
            u_apply = u_mpc.numpy()
        x_next_np = mpc_controller._rk4_step(x_current_np, u_apply, dt)
        x_current = torch.tensor(x_next_np, dtype=torch.float32)

        times.append(times[-1] + dt)
        states.append(x_next_np)

    return (
        torch.tensor(times, dtype=torch.float32),
        torch.tensor(np.array(states), dtype=torch.float32),
        torch.tensor(np.array(controls), dtype=torch.float32)
    )
