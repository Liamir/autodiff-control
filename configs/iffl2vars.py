"""Configuration for IFFL2Vars environment with MLP controller.

The IFFL2Vars system has simple equations with scalar control on input:
    dx/dt = alpha * (control * Input) - delta * x
    dy/dt = beta * (control * Input) - gamma * x * y

Goal: Destabilize the system by maximizing deviation from steady state.
Steady state: y_ss = (beta * delta) / (alpha * gamma)

With default params (alpha=1, delta=0.1, beta=1, gamma=1): y_ss = 0.1

We use an MLP controller: control = MLP(x, y)
"""
import torch
from rpasim.ode.rpa.iffl2vars import IFFL2Vars
from rpa_control.controllers import MLPController, ControlledODE


# Fixed parameters: [alpha, delta, beta, gamma]
FIXED_PARAMS = torch.tensor([1.0, 0.1, 1.0, 1.0])

# Calculate steady state for y
alpha, delta, beta, gamma = FIXED_PARAMS
Y_SS = (beta * delta) / (alpha * gamma)  # 0.1
DESIRED_Y_SS = 0.15
print(f"Steady state for y: y_ss = {Y_SS:.4f}")


def input_signal(t):
    """Input signal: constant 1.0 for now."""
    if torch.is_tensor(t):
        return torch.tensor(1.0, dtype=t.dtype, device=t.device)
    else:
        return 1.0


def reward_fn(state, time=None):
    """Reward function: maximize |y - y_ss| (destabilize).

    Args:
        state: [x, y]
        time: Optional time (unused)

    Returns:
        Reward (to maximize)
    """
    y = state[1]
    deviation = (y - Y_SS)**2
    return deviation


def mpc_stage_cost_fn(x_next, u, x_curr=None, k=None):
    """Custom MPC cost function: maximize |y - y_ss| (destabilize).

    Args:
        x_next: Next state [x, y]
        u: Control input (scalar)
        x_curr: Current state (optional)
        k: Time step (optional)

    Returns:
        Cost (to minimize)
    """
    # Reuse training reward function
    reward = reward_fn(x_next)

    # MPC minimizes cost, training maximizes reward
    # We want to maximize deviation, so minimize negative deviation
    cost = -reward

    return cost


def create_base_ode():
    """Create base IFFL2Vars ODE (for MPC or uncontrolled simulation)."""
    return IFFL2Vars(fixed_params=FIXED_PARAMS, input_signal=input_signal)


def create_ode(n_hidden=8, activation='tanh', controller_order=None, include_constant=None, **kwargs):
    """Create controlled IFFL2Vars ODE instance with MLP controller.

    Args:
        n_hidden: Number of hidden layer neurons (default: 8)
        activation: Activation function - 'tanh', 'relu', or 'sigmoid' (default: 'tanh')
        controller_order: Ignored (MLP doesn't use polynomial basis)
        include_constant: Ignored (MLP doesn't use polynomial basis)
        **kwargs: Additional ignored arguments for compatibility
    """
    # Create IFFL2Vars ODE
    iffl_ode = create_base_ode()

    # Create MLP controller
    # 2 state vars (x, y), 1 control output (scalar multiplier for Input)
    controller = MLPController(
        n_state_vars=2,
        n_control_vars=1,
        n_hidden=n_hidden,
        activation=activation,
    )

    # Create controlled ODE
    # control_indices is required but IFFL2Vars handles control internally
    controlled_ode = ControlledODE(
        base_ode=iffl_ode,
        controller=controller,
        control_indices=[0],  # Placeholder - IFFL2Vars uses control internally
        control_bounds=(0.5, 2.0)  # Control multiplier range
    )

    return controlled_ode


# Environment configuration
ENV_CONFIG = {
    'name': 'iffl2vars',
    'experiment_name': 'iffl2vars',
    'has_controller': True,  # This is a control problem

    # ODE setup
    'create_base_ode': create_base_ode,
    'create_ode': create_ode,
    'reward_fn': reward_fn,
    'mpc_stage_cost_fn': mpc_stage_cost_fn,  # Custom MPC cost (same objective as training)

    # Initial conditions
    'initial_state': torch.tensor([10.0, 0.1]),  # Start at steady state
    'initial_state_range': [(0.0, 15.0), (0.0, 1.0)],  # x: [0, 15], y: [0, 1]

    # Fixed evaluation initial conditions
    'eval_initial_states': [
        torch.tensor([2.0, 0.05]),   # Low x, low y (below y_ss)
        torch.tensor([2.0, 0.8]),    # Low x, high y (above y_ss)
        torch.tensor([10.0, 0.05]),  # High x, low y
        torch.tensor([10.0, 0.8]),   # High x, high y
        torch.tensor([5.0, 0.5]),    # Middle
    ],

    # Perturbation settings (for ODE parameters)
    'perturb_param_indices': [0, 1, 2, 3],  # Perturb all 4 parameters

    # Display settings
    'param_names': None,  # MLP parameters (will show architecture info)
    'state_var_names': ['x', 'y'],
    'control_names': ['u'],

    # Description for display
    'description': f"""IFFL2Vars with MLP Controller

System:
  dx/dt = alpha * (u * Input) - delta * x
  dy/dt = beta * (u * Input) - gamma * x * y

Controller: u = MLP(x, y)
  - Input: 2 neurons (x, y)
  - Hidden: configurable neurons (tanh activation)
  - Output: 1 neuron (u)

Control objective: Destabilize system by maximizing |y - y_ss|
  y_ss = (beta * delta) / (alpha * gamma) = {Y_SS:.4f}

Parameters (fixed):
  alpha = {FIXED_PARAMS[0]:.2f}
  delta = {FIXED_PARAMS[1]:.2f}
  beta  = {FIXED_PARAMS[2]:.2f}
  gamma = {FIXED_PARAMS[3]:.2f}

Input signal: constant 1.0

Note: Control modulates the input signal effect on both equations.""",

    # Default training settings
    'defaults': {
        'time_horizon': 20.0,
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
        'control_bounds': (0.5, 2.0),  # Control multiplier bounds
        'seed': 1,
        'use_single_ic': True,
        'use_tbptt': True,
        'tbptt_truncation_steps': 50,
        'plot_trajectories': True,
        'plot_trajectory_interval': 5,
    },

    # MPC settings
    'mpc_defaults': {
        'prediction_horizon': 20,
        'dt': 0.1,                # Time step size
        'Q': [1.0, 1.0],           # State tracking weights [x, y]
        'Ru': 0.0,                 # Control magnitude weight
        'R_deltau': 0.0,           # Control rate-of-change weight
        'u_min': 0.5,              # Minimum control input
        'u_max': 2.0,              # Maximum control input
        'cost_type': 'quadratic',
        'ftol': 1e-3,
        'n_controls': 1,           # Single control input
    }
}
