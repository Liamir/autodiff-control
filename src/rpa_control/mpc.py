"""Nonlinear Model Predictive Control (NMPC) for ODE systems.

Uses scipy.optimize with RK4 integration based on the SINDY-MPC paper approach.
"""
import torch
import numpy as np
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
    ):
        """Initialize MPC configuration.

        Args:
            prediction_horizon: Number of steps to predict ahead
            dt: Time step
            Q: State tracking weight (diagonal of Q matrix)
            Ru: Control magnitude weight
            R_deltau: Control rate-of-change weight
            u_min: Minimum control input
            u_max: Maximum control input
            state_min: Minimum state values (optional)
            state_max: Maximum state values (optional)
            ftol: Optimization tolerance
            disp: Display optimization progress
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


class MPCController:
    """Nonlinear Model Predictive Control using scipy.optimize."""

    def __init__(
        self,
        ode: torch.nn.Module,
        config: MPCConfig,
        reference_state: torch.Tensor,
        control_indices: list[int],
    ):
        """Initialize MPC controller.

        Args:
            ode: ODE system
            config: MPC configuration
            reference_state: Target state to track
            control_indices: Indices of state variables affected by control
        """
        self.ode = ode
        self.config = config
        self.reference_state = reference_state.detach().numpy()
        self.control_indices = control_indices
        self.n_states = len(reference_state)
        self.n_controls = len(control_indices)

        # Convert Q to diagonal matrix
        Q_diag = config.Q.numpy() if isinstance(config.Q, torch.Tensor) else np.array(config.Q)
        self.Q = np.diag(Q_diag)
        self.Ru = config.Ru
        self.R_deltau = config.R_deltau

        # Previous control for warm-starting
        self.u_prev = 0.0
        self.u_guess = None

    def _ode_dynamics(self, x: np.ndarray, u: float) -> np.ndarray:
        """Evaluate ODE dynamics: dx/dt = f(x) + control

        Args:
            x: State vector
            u: Control input (scalar)

        Returns:
            State derivative
        """
        x_torch = torch.tensor(x, dtype=torch.float32)

        with torch.no_grad():
            # Get base dynamics
            dx = self.ode(torch.tensor(0.0), x_torch).numpy()

            # Add control to appropriate state(s)
            for idx in self.control_indices:
                dx[idx] += u

        return dx

    def _rk4_step(self, x: np.ndarray, u: float, dt: float) -> np.ndarray:
        """RK4 integration step.

        Args:
            x: Current state
            u: Control input
            dt: Time step

        Returns:
            Next state
        """
        k1 = self._ode_dynamics(x, u)
        k2 = self._ode_dynamics(x + 0.5 * dt * k1, u)
        k3 = self._ode_dynamics(x + 0.5 * dt * k2, u)
        k4 = self._ode_dynamics(x + dt * k3, u)
        return x + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

    def _mpc_objective(self, u_seq: np.ndarray, x0: np.ndarray) -> float:
        """MPC cost function.

        Args:
            u_seq: Control sequence (flattened)
            x0: Initial state

        Returns:
            Total cost
        """
        N = self.config.prediction_horizon
        dt = self.config.dt

        u_seq = u_seq.reshape(N)

        cost = 0.0
        x_curr = x0.copy()

        for k in range(N):
            u_k = u_seq[k]

            # Predict next state
            x_next = self._rk4_step(x_curr, u_k, dt)

            # State tracking cost
            error = x_next - self.reference_state
            state_cost = error.T @ self.Q @ error

            # Control magnitude cost
            control_magnitude_cost = self.Ru * (u_k**2)

            # Control rate-of-change cost
            if k == 0:
                du = u_k - self.u_prev
            else:
                du = u_k - u_seq[k-1]
            control_change_cost = self.R_deltau * (du**2)

            cost += state_cost + control_magnitude_cost + control_change_cost

            x_curr = x_next

        return cost

    def step(self, x_current: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """Compute MPC control action (receding horizon).

        Args:
            x_current: Current state

        Returns:
            Control input to apply
            Info dictionary
        """
        N = self.config.prediction_horizon
        x0_np = x_current.detach().numpy()

        # Initial guess (warm-start from previous solution)
        if self.u_guess is None:
            self.u_guess = np.zeros(N)

        # Bounds
        bounds = [(self.config.u_min, self.config.u_max) for _ in range(N)]

        # Optimize
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

        u_optimal = res.x
        u_control = u_optimal[0]

        # Update for next iteration
        self.u_prev = u_control
        self.u_guess = np.roll(u_optimal, -1)
        self.u_guess[-1] = u_optimal[-1]

        info = {
            'success': res.success,
            'cost': float(res.fun),
            'message': res.message,
            'nit': res.nit,
        }

        return torch.tensor([u_control], dtype=torch.float32), info


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

        # Apply control and simulate one step
        x_current_np = x_current.detach().numpy()
        u_scalar = float(u_mpc[0])
        x_next_np = mpc_controller._rk4_step(x_current_np, u_scalar, dt)
        x_current = torch.tensor(x_next_np, dtype=torch.float32)

        times.append(times[-1] + dt)
        states.append(x_next_np)

    return (
        torch.tensor(times, dtype=torch.float32),
        torch.tensor(np.array(states), dtype=torch.float32),
        torch.tensor(np.array(controls), dtype=torch.float32)
    )
