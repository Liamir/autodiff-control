"""Configuration for IFFL2VarsControlled environment with MLP controller.

The IFFL2VarsControlled system has parameter-level control:
    dx/dt = (u[0] * alpha) * Input(t) - (u[1] * delta) * x
    dy/dt = (u[2] * beta) * Input(t) - (u[3] * gamma) * x * y

The input signal is fixed (not controllable). Instead, 4 control inputs
multiply the base parameters: alpha, delta, beta, gamma.

Goal: Maximize |dy/dt| to force perpetual transients, defeating RPA.
The IFFL adapts to constant parameter changes, so only time-varying
control can keep y away from steady state.

We use an MLP controller: [u_alpha, u_delta, u_beta, u_gamma] = MLP(x, y)
"""
import torch
from rpasim.ode.rpa.iffl2vars_controlled import IFFL2VarsControlled
from rpa_control.controllers import MLPController, ControlledODE


# Fixed parameters: [alpha, delta, beta, gamma]
FIXED_PARAMS = torch.tensor([1.0, 0.1, 10.0, 1.0])

# Calculate steady state for y (with all u[i]=1)
alpha, delta, beta, gamma = FIXED_PARAMS
Y_SS = (beta * delta) / (alpha * gamma)  # 0.1
print(f"Steady state for y: y_ss = {Y_SS:.4f}")


def input_signal(t):
    """Input signal: constant 1.0 for now."""
    if torch.is_tensor(t):
        return torch.tensor(1.0, dtype=t.dtype, device=t.device)
    else:
        return 1.0


def reward_fn(state, time=None):
    """Reward function: maximize |dy/dt| (force perpetual transients).

    We compute dy/dt analytically from the ODE equations:
        dy/dt = beta * Input - gamma * x * y

    With default (uncontrolled) parameters. This measures how far
    the system is from any steady state (where dy/dt = 0).

    Args:
        state: [x, y]
        time: Optional time (unused)

    Returns:
        Reward (to maximize)
    """
    x_val, y_val = state[0], state[1]
    # dy/dt with base parameters (no control modulation)
    dy_dt = beta * 1.0 - gamma * x_val * y_val
    return dy_dt ** 2


def mpc_stage_cost_fn(x_next, u, x_curr=None, k=None):
    """Custom MPC cost: maximize |dy/dt| between steps (force transients).

    Uses finite difference (y_next - y_curr) / dt as proxy for |dy/dt|.
    Control penalties use log-space for multiplicative controls.

    Args:
        x_next: Next state [x, y]
        u: Control input [u_alpha, u_delta, u_beta, u_gamma]
        x_curr: Current state (optional)
        k: Time step (optional)

    Returns:
        Cost (to minimize)
    """
    # Maximize |dy/dt| approximated by (y_next - y_curr)^2
    if x_curr is not None:
        dy = x_next[1] - x_curr[1]
        transient_reward = dy ** 2
    else:
        # Fallback: use analytical dy/dt at x_next
        transient_reward = (beta * 1.0 - gamma * x_next[0] * x_next[1]) ** 2

    # Control effort penalty in log-space (deviation from u=1 baseline)
    Ru = 0.01
    control_effort = Ru * torch.sum(torch.log(u) ** 2)

    # MPC minimizes cost
    cost = -transient_reward + control_effort
    return cost


def create_base_ode():
    """Create base IFFL2VarsControlled ODE (for MPC or uncontrolled simulation)."""
    return IFFL2VarsControlled(fixed_params=FIXED_PARAMS, input_signal=input_signal)


def create_ode(n_hidden=8, activation='tanh', controller_order=None, include_constant=None, **kwargs):
    """Create controlled IFFL2VarsControlled ODE instance with MLP controller.

    Args:
        n_hidden: Number of hidden layer neurons (default: 8)
        activation: Activation function - 'tanh', 'relu', or 'sigmoid' (default: 'tanh')
        controller_order: Ignored (MLP doesn't use polynomial basis)
        include_constant: Ignored (MLP doesn't use polynomial basis)
        **kwargs: Additional ignored arguments for compatibility
    """
    base_ode = create_base_ode()

    # Create MLP controller
    # 2 state vars (x, y), 4 control outputs (multipliers for alpha, delta, beta, gamma)
    controller = MLPController(
        n_state_vars=2,
        n_control_vars=4,
        n_hidden=n_hidden,
        activation=activation,
    )

    # Create controlled ODE
    # control_indices is required but IFFL2VarsControlled handles control internally
    controlled_ode = ControlledODE(
        base_ode=base_ode,
        controller=controller,
        control_indices=[0, 0, 1, 1],  # Placeholder - base ODE uses control internally
        control_bounds=(0.5, 2.0),  # Parameter multiplier range
    )

    return controlled_ode


# Environment configuration
ENV_CONFIG = {
    'name': 'iffl2vars_controlled',
    'experiment_name': 'iffl2vars_controlled',
    'has_controller': True,

    # ODE setup
    'create_base_ode': create_base_ode,
    'create_ode': create_ode,
    'reward_fn': reward_fn,
    'mpc_stage_cost_fn': mpc_stage_cost_fn,

    # Initial conditions
    'initial_state': torch.tensor([10.0, 0.1]),  # Start at steady state
    'initial_state_range': [(0.0, 15.0), (0.0, 1.0)],

    # Fixed evaluation initial conditions
    'eval_initial_states': [
        torch.tensor([2.0, 0.05]),   # Low x, low y
        torch.tensor([2.0, 0.8]),    # Low x, high y
        torch.tensor([10.0, 0.05]),  # High x, low y
        torch.tensor([10.0, 0.8]),   # High x, high y
        torch.tensor([5.0, 0.5]),    # Middle
    ],

    # Perturbation settings (for ODE parameters)
    'perturb_param_indices': [0, 1, 2, 3],

    # Display settings
    'param_names': None,  # MLP parameters
    'state_var_names': ['x', 'y'],
    'control_names': ['u_alpha', 'u_delta', 'u_beta', 'u_gamma'],

    # Description for display
    'description': f"""IFFL2VarsControlled with MLP Controller

System:
  dx/dt = (u_alpha * alpha) * Input(t) - (u_delta * delta) * x
  dy/dt = (u_beta * beta) * Input(t) - (u_gamma * gamma) * x * y

Controller: [u_alpha, u_delta, u_beta, u_gamma] = MLP(x, y)
  - Input: 2 neurons (x, y)
  - Hidden: configurable neurons (tanh activation)
  - Output: 4 neurons (parameter multipliers)

Control objective: Maximize |dy/dt| to force perpetual transients
  y_ss = (beta * delta) / (alpha * gamma) = {Y_SS:.4f} (with u=1)

Parameters (fixed):
  alpha = {FIXED_PARAMS[0]:.2f}
  delta = {FIXED_PARAMS[1]:.2f}
  beta  = {FIXED_PARAMS[2]:.2f}
  gamma = {FIXED_PARAMS[3]:.2f}

Input signal: constant 1.0

Note: Control modulates the 4 ODE parameters directly.""",

    # Default training settings
    'defaults': {
        'time_horizon': 15.0,
        'n_reward_steps': 200,
        'steady_state_fraction': 0.5,
        'learning_rate': 0.01,
        'n_iterations': 500,
        'log_interval': 20,
        'eval_interval': 60,
        'n_hidden': 8,
        'activation': 'tanh',
        'scale_aware_regularization': False,
        'state_limits': (-100.0, 100.0),
        'control_bounds': (0.5, 2.0),
        'seed': 1,
        'use_single_ic': True,
        'use_tbptt': True,
        'tbptt_truncation_steps': 50,
        'plot_trajectories': True,
        'plot_trajectory_interval': 5,
    },

    # MPC settings
    'mpc_defaults': {
        'prediction_horizon': 10,
        'dt': 1.0,
        'integration_substeps': 40,
        'Q': [1.0, 1.0],
        'Ru': 0.0,
        'R_deltau': 0.0,
        'u_min': 0.5,
        'u_max': 2.0,
        'cost_type': 'quadratic',
        'ftol': 1e-3,
        'n_controls': 4,
    }
}
