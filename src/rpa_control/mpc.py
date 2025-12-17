"""Model Predictive Control (MPC) implementation for nonlinear ODEs.

Uses CasADi for efficient nonlinear optimization with automatic differentiation.
Reference: https://web.casadi.org/
"""
import torch
import numpy as np
import casadi as ca
from typing import Optional, Tuple
from torchdiffeq import odeint


class MPCConfig:
    """Configuration for Model Predictive Control."""

    def __init__(
        self,
        prediction_horizon: int = 10,
        dt: float = 0.1,
        # Cost function weights
        Q: Optional[torch.Tensor] = None,  # State tracking weight
        R: Optional[torch.Tensor] = None,  # Control rate-of-change weight
        Ru: Optional[torch.Tensor] = None,  # Control magnitude weight
        # Control constraints
        u_min: float = -20.0,
        u_max: float = 20.0,
        # State constraints (optional)
        state_min: Optional[torch.Tensor] = None,
        state_max: Optional[torch.Tensor] = None,
        # Optimization settings
        max_iter: int = 100,
        tol: float = 1e-6,
    ):
        """Initialize MPC configuration.

        Args:
            prediction_horizon: Number of steps to predict ahead (N)
            dt: Time step for discrete-time prediction
            Q: State tracking weight matrix (n_states x n_states) or scalar
            R: Control rate-of-change weight matrix (n_controls x n_controls) or scalar
            Ru: Control magnitude weight matrix (n_controls x n_controls) or scalar
            u_min: Minimum control input
            u_max: Maximum control input
            state_min: Minimum state values (optional)
            state_max: Maximum state values (optional)
            max_iter: Maximum optimization iterations
            tol: Optimization tolerance
        """
        self.prediction_horizon = prediction_horizon
        self.dt = dt
        self.Q = Q
        self.R = R
        self.Ru = Ru
        self.u_min = u_min
        self.u_max = u_max
        self.state_min = state_min
        self.state_max = state_max
        self.max_iter = max_iter
        self.tol = tol


class MPCController:
    """Nonlinear Model Predictive Control using CasADi."""

    def __init__(
        self,
        ode: torch.nn.Module,
        config: MPCConfig,
        reference_state: torch.Tensor,
        control_indices: list[int],
    ):
        """Initialize MPC controller.

        Args:
            ode: ODE system (must be a torch.nn.Module with forward(t, x) method)
            config: MPC configuration
            reference_state: Target/reference state to track
            control_indices: Indices of state variables affected by control
        """
        self.ode = ode
        self.config = config
        self.reference_state = reference_state.clone().detach()
        self.control_indices = control_indices

        # Infer dimensions
        self.n_states = len(reference_state)
        self.n_controls = len(control_indices)

        # Initialize default weight matrices if not provided
        if config.Q is None:
            self.Q = torch.ones(self.n_states)
        elif config.Q.numel() == 1:
            self.Q = config.Q * torch.ones(self.n_states)
        else:
            self.Q = config.Q

        if config.R is None:
            self.R = 0.5 * torch.ones(self.n_controls)
        elif config.R.numel() == 1:
            self.R = config.R * torch.ones(self.n_controls)
        else:
            self.R = config.R

        if config.Ru is None:
            self.Ru = 0.5 * torch.ones(self.n_controls)
        elif config.Ru.numel() == 1:
            self.Ru = config.Ru * torch.ones(self.n_controls)
        else:
            self.Ru = config.Ru

        # Previous control for rate-of-change penalty
        self.u_prev = torch.zeros(self.n_controls)

    def _ode_numpy(self, x_np: np.ndarray) -> np.ndarray:
        """Evaluate ODE in numpy (for CasADi).

        Args:
            x_np: State vector (numpy)

        Returns:
            State derivative dx/dt (numpy)
        """
        x_torch = torch.tensor(x_np, dtype=torch.float32)
        with torch.no_grad():
            dx_torch = self.ode(torch.tensor(0.0), x_torch)
        return dx_torch.numpy()

    def optimize_control(
        self,
        x0: torch.Tensor,
        u_init: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, dict]:
        """Solve MPC optimization problem using CasADi.

        Args:
            x0: Current state
            u_init: Initial guess for control sequence (if None, use zero)

        Returns:
            Optimal control sequence (prediction_horizon, n_controls)
            Optimization info dictionary
        """
        N = self.config.prediction_horizon
        dt = self.config.dt

        # Create CasADi optimization problem
        opti = ca.Opti()

        # Decision variables
        X = opti.variable(self.n_states, N + 1)  # States
        U = opti.variable(self.n_controls, N)     # Controls

        # Convert torch tensors to numpy for CasADi
        x0_np = x0.numpy()
        ref_np = self.reference_state.numpy()
        Q_np = self.Q.numpy()
        R_np = self.R.numpy()
        Ru_np = self.Ru.numpy()
        u_prev_np = self.u_prev.numpy()

        # Objective function
        cost = 0

        for k in range(N):
            # State tracking cost
            state_error = X[:, k+1] - ref_np
            for i in range(self.n_states):
                cost += Q_np[i] * state_error[i]**2

            # Control rate-of-change cost
            if k == 0:
                du = U[:, k] - u_prev_np
            else:
                du = U[:, k] - U[:, k-1]
            for i in range(self.n_controls):
                cost += R_np[i] * du[i]**2

            # Control magnitude cost
            for i in range(self.n_controls):
                cost += Ru_np[i] * U[i, k]**2

        opti.minimize(cost)

        # Initial condition constraint
        opti.subject_to(X[:, 0] == x0_np)

        # Dynamics constraints (Euler integration with linearized dynamics)
        # Evaluate dynamics at initial state for linearization
        dx0 = self._ode_numpy(x0_np)

        for k in range(N):
            x_k = X[:, k]

            # Linearized dynamics: dx = dx0 (constant) + control
            # Build dx as list, then convert to vertcat
            dx_list = []
            for i in range(self.n_states):
                if i in self.control_indices:
                    # Find which control index this is
                    ctrl_idx = self.control_indices.index(i)
                    dx_list.append(dx0[i] + U[ctrl_idx, k])
                else:
                    dx_list.append(dx0[i])

            dx = ca.vertcat(*dx_list)
            x_next = x_k + dt * dx
            opti.subject_to(X[:, k+1] == x_next)

        # Control bounds
        for k in range(N):
            for i in range(self.n_controls):
                opti.subject_to(U[i, k] >= self.config.u_min)
                opti.subject_to(U[i, k] <= self.config.u_max)

        # State constraints (if provided)
        if self.config.state_min is not None:
            state_min_np = self.config.state_min.numpy()
            for k in range(1, N+1):
                for i in range(self.n_states):
                    opti.subject_to(X[i, k] >= state_min_np[i])

        if self.config.state_max is not None:
            state_max_np = self.config.state_max.numpy()
            for k in range(1, N+1):
                for i in range(self.n_states):
                    opti.subject_to(X[i, k] <= state_max_np[i])

        # Set initial guess
        if u_init is None:
            u_init = torch.zeros(N, self.n_controls)
        opti.set_initial(U, u_init.numpy().T)

        # Warm-start states with forward simulation
        x_guess = x0.clone()
        x_traj_guess = [x_guess.numpy()]
        for k in range(N):
            u_k = u_init[k]
            u_full = torch.zeros(self.n_states)
            for i, idx in enumerate(self.control_indices):
                u_full[idx] = u_k[i]
            with torch.no_grad():
                dx = self.ode(torch.tensor(0.0), x_guess) + u_full
                x_guess = x_guess + dt * dx
            x_traj_guess.append(x_guess.numpy())
        opti.set_initial(X, np.array(x_traj_guess).T)

        # Solver options
        opts = {
            'ipopt.print_level': 0,
            'print_time': 0,
            'ipopt.max_iter': self.config.max_iter,
            'ipopt.tol': self.config.tol,
            'ipopt.acceptable_tol': self.config.tol * 10,
        }
        opti.solver('ipopt', opts)

        # Solve
        try:
            sol = opti.solve()
            success = True
            U_opt = sol.value(U)
            cost_val = float(sol.value(cost))
            message = "Optimization succeeded"
        except RuntimeError as e:
            # If solver fails, try to get the best solution so far
            success = False
            try:
                U_opt = opti.debug.value(U)
                cost_val = float(opti.debug.value(cost))
            except:
                U_opt = u_init.numpy().T
                cost_val = float('inf')
            message = f"Optimization failed: {str(e)}"

        # Convert back to torch
        u_opt = torch.tensor(U_opt.T, dtype=torch.float32)

        info = {
            'success': success,
            'cost': cost_val,
            'message': message
        }

        return u_opt, info

    def step(
        self,
        x_current: torch.Tensor
    ) -> Tuple[torch.Tensor, dict]:
        """Compute MPC control action for current state (receding horizon).

        Args:
            x_current: Current state

        Returns:
            Control input to apply (n_controls,)
            Info dictionary with optimization details
        """
        # Solve optimization problem
        u_opt_sequence, info = self.optimize_control(x_current)

        # Extract first control action (receding horizon principle)
        u_mpc = u_opt_sequence[0]

        # Update previous control for next iteration
        self.u_prev = u_mpc.clone()

        # Store optimal sequence for warm-starting next iteration
        # Shift sequence by one and pad with last value
        self._u_opt_sequence = torch.cat([u_opt_sequence[1:], u_opt_sequence[-1:]])

        return u_mpc, info


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
        dt: Time step for simulation (can differ from MPC dt)

    Returns:
        times: Time points (n_steps+1,)
        states: State trajectory (n_steps+1, n_states)
        controls: Control inputs (n_steps, n_controls)
    """
    n_steps = int(time_horizon / dt)
    times = [0.0]
    states = [x0]
    controls = []

    x_current = x0.clone()

    for step in range(n_steps):
        # Compute MPC control action
        u_mpc, info = mpc_controller.step(x_current)
        controls.append(u_mpc)

        # Apply control and simulate one step (Euler integration)
        u_full = torch.zeros_like(x_current)
        # Handle both single and multiple controls
        u_mpc_flat = u_mpc.flatten()
        for i, idx in enumerate(mpc_controller.control_indices):
            u_full[idx] = u_mpc_flat[i]

        with torch.no_grad():
            dx = ode(torch.tensor(0.0), x_current) + u_full
            x_next = x_current + dt * dx

        times.append(times[-1] + dt)
        states.append(x_next)
        x_current = x_next

    return (
        torch.tensor(times),
        torch.stack(states),
        torch.stack(controls)
    )
