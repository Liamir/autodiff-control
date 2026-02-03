"""Configuration for ABControlled environment with MLP controller.

The ABControlled ODE has simplified equations with single control input:
    dA/dt = u*alpha*(1-B)
    dB/dt = A - B

We use an MLP controller: u = MLP(A, B)
"""
import torch
from rpasim.ode.rpa.ab import ABControlled
from rpa_control.controllers import MLPController, ControlledODE


def reward_fn(state, time=None):
    """Reward function: maximize |B| (resonance behavior).
    """
    B_norm = torch.abs(state[1])
    # -1 / e^B_norm
    return B_norm


def mpc_stage_cost_fn(x_next, u, x_curr=None, k=None):
    """Custom MPC cost function: same objective as training (maximize |B|).

    Args:
        x_next: Next state [A, B]
        u: Control input
        x_curr: Current state (optional)
        k: Time step (optional)

    Returns:
        Cost (to minimize)
    """
    # Reuse training reward function
    reward = reward_fn(x_next)

    # Add control penalty to avoid excessive control
    control_penalty = 0.01 * (u**2)

    # MPC minimizes cost, training maximizes reward
    cost = -reward + control_penalty

    return cost


# ODE parameter: alpha
FIXED_PARAMS = torch.tensor([50.0])


def create_base_ode():
    """Create base ABControlled ODE (for MPC or uncontrolled simulation)."""
    return ABControlled(fixed_params=FIXED_PARAMS)


def create_ode(n_hidden=8, activation='tanh', controller_order=None, include_constant=None, **kwargs):
    """Create controlled ABControlled ODE instance with MLP controller.

    Args:
        n_hidden: Number of hidden layer neurons (default: 8)
        activation: Activation function - 'tanh', 'relu', or 'sigmoid' (default: 'tanh')
        controller_order: Ignored (MLP doesn't use polynomial basis)
        include_constant: Ignored (MLP doesn't use polynomial basis)
        **kwargs: Additional ignored arguments for compatibility
    """
    # Create ABControlled ODE
    ab_ode = create_base_ode()

    # Create MLP controller
    # 2 state vars (A, B), 1 control output (u)
    controller = MLPController(
        n_state_vars=2,
        n_control_vars=1,
        n_hidden=n_hidden,
        activation=activation,
    )

    # Create controlled ODE
    # control_indices is required but ABControlled handles control internally
    controlled_ode = ControlledODE(
        base_ode=ab_ode,
        controller=controller,
        control_indices=[0],  # Placeholder - ABControlled uses control internally
        control_bounds=(0.2, 1.0)  # Clip control between 0 and 1
    )

    return controlled_ode


# Environment configuration
ENV_CONFIG = {
    'name': 'ab_controlled_mlp',
    'experiment_name': 'ab_controlled_mlp',
    'has_controller': True,  # This is a control problem

    # ODE setup
    'create_base_ode': create_base_ode,
    'create_ode': create_ode,
    'reward_fn': reward_fn,
    'mpc_stage_cost_fn': mpc_stage_cost_fn,  # Custom MPC cost (same objective as training)

    # Initial conditions
    'initial_state': torch.tensor([0.5, 2.0]),
    'initial_state_range': [(-2.0, 2.0), (-2.0, 2.0)],  # A: [0.2, 2.0], B: [0.5, 6.0]

    # Fixed evaluation initial conditions
    'eval_initial_states': [
        torch.tensor([0.2, 0.5]),   # Low A, Low B
        torch.tensor([0.2, 6.0]),   # Low A, High B
        torch.tensor([2.0, 0.5]),   # High A, Low B
        torch.tensor([2.0, 6.0]),   # High A, High B
        torch.tensor([1.0, 1.0]),   # Middle
    ],

    # Perturbation settings (for ODE parameters)
    'perturb_param_indices': [0],  # Perturb alpha parameter

    # Display settings
    'param_names': None,  # MLP parameters (will show architecture info)
    'state_var_names': ['A', 'B'],
    'control_names': ['u'],

    # Description for display
    'description': """ABControlled (Simplified) with MLP Controller

System:
  dA/dt = u*alpha*(1-B)
  dB/dt = A - B

Controller: u = MLP(A, B)
  - Input: 2 neurons (A, B)
  - Hidden: 16 neurons (relu activation)
  - Output: 1 neuron (u)

Control objective: maximize |B| (resonance behavior)

Parameters (fixed):
  alpha = 50.0

Note: With u=1, system is unstable. MLP learns control to amplify B.""",

    # Default training settings
    'defaults': {
        'time_horizon': 10.0,
        'n_reward_steps': 100,
        'steady_state_fraction': 0.5,
        'learning_rate': 0.01,  # May need tuning for MLP
        'n_iterations': 300,
        'log_interval': 20,
        'eval_interval': 60,
        'n_hidden': 8,  # MLP-specific
        'activation': 'tanh',  # MLP-specific
        'scale_aware_regularization': False,  # Not applicable for MLP
        'state_limits': (-2000.0, 2000.0),
        'control_bounds': (0.2, 1.0),
        'seed': 1,
        'use_single_ic': True,  # Start with single IC for stability
        'use_tbptt': True,  # Truncated BPTT: breaks gradient flow to prevent vanishing/exploding gradients
        'tbptt_truncation_steps': 35,  # Number of timesteps per chunk (50 = 2 chunks for 100 steps)
        'plot_trajectories': True,  # Enable trajectory plotting during training
        'plot_trajectory_interval': 5,  # Plot when eval hasn't improved for N evaluations
    },

    # MPC settings
    'mpc_defaults': {
        'prediction_horizon': 20,
        'dt': 0.1,
        'Q': [1.0, 1.0],           # State tracking weights [A, B]
        'Ru': 0.1,                  # Control magnitude weight
        'R_deltau': 0.1,           # Control rate-of-change weight
        'u_min': 0.2,              # Minimum control input
        'u_max': 1.0,              # Maximum control input
        'cost_type': 'quadratic',
        'ftol': 1e-3,
        'n_controls': 1,           # Single control input
    }
}
