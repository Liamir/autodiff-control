"""Configuration for Lorenz System (Chaotic) environment with static controller."""
import torch
from rpasim.ode.classic_control.lorenz import Lorenz
from rpa_control.controllers import StaticController, ControlledODE


def reward_fn(state, time=None):
    """Reward function: stabilize one of the fixed points.

    Fixed points: (±8.49, ±8.49, 27)
    We target the positive fixed point: (8.49, 8.49, 27)
    """
    fixed_point = torch.tensor([8.49, 8.49, 27.0])
    return -((state - fixed_point) ** 2).sum()


def create_base_ode():
    """Create base Lorenz ODE (for MPC or uncontrolled simulation)."""
    return Lorenz()


def create_ode(controller_order=2, include_constant=True):
    """Create controlled Lorenz ODE instance."""
    # Create Lorenz ODE
    lorenz_ode = create_base_ode()

    # Create static controller
    # 3 state vars (x1, x2, x3), 1 control output (affects x1 only)
    controller = StaticController(
        n_state_vars=3,
        n_control_vars=1,
        order=controller_order,
        include_constant=include_constant
    )

    # Create controlled ODE
    controlled_ode = ControlledODE(
        base_ode=lorenz_ode,
        controller=controller,
        control_indices=[0]  # Control affects x1 only
    )

    return controlled_ode


# Environment configuration
ENV_CONFIG = {
    'name': 'lorenz',
    'experiment_name': 'lorenz_static',
    'has_controller': True,  # This is a control problem

    # ODE setup
    'create_base_ode': create_base_ode,
    'create_ode': create_ode,
    'reward_fn': reward_fn,

    # Initial conditions
    'initial_state': torch.tensor([2.5, 2.0, 15.0]),
    'initial_state_range': [(2.0, 12.0), (2.0, 12.0), (15.0, 35.0)],  # Around fixed points

    # Fixed evaluation initial conditions (match training IC when use_single_ic=True)
    'eval_initial_states': [
        torch.tensor([2.5, 2.0, 15.0]),   # Training IC
    ],

    # Perturbation settings (for ODE parameters [sigma, beta, rho])
    'perturb_param_indices': [],  # No perturbations during evaluation

    # Display settings
    'param_names': None,  # Controller parameters (will be basis function names)
    'state_var_names': ['x1', 'x2', 'x3'],
    'control_names': ['u'],
    'target_vars': {0: 8.49, 1: 8.49, 2: 27.0},  # Target values for plotting: {state_idx: value}

    # Description for display
    'description': """Lorenz System (Chaotic) with Static Controller

System:
  dx1/dt = 10.00*(x2 - x1) + u
  dx2/dt = x1*(28.00 - x3) - x2
  dx3/dt = x1*x2 - 2.67*x3

Controller: u = θ · Φ(x1, x2, x3)

Control objective: stabilize fixed point
  x1 = 8.49
  x2 = 8.49
  x3 = 27.0

Parameters:
  ODE fixed params: sigma=10.00, beta=2.67, rho=28.00
  Controller params: θ (polynomial basis coefficients)""",

    # Default training settings
    'defaults': {
        'time_horizon': 3.0,
        'n_reward_steps': 3000,  # dt = 3.0 / 3000 = 0.001
        'steady_state_fraction': 0.5,
        'learning_rate': 0.01,
        'n_iterations': 500,
        'log_interval': 50,
        'eval_interval': 50,
        'controller_order': 2,
        'scale_aware_regularization': True,
        'state_limits': (-20.0, 50.0),  # Wide limits for chaotic system
        'gradient_clip_norm': 10.0,  # Clip gradients to prevent explosion
        'seed': 42,  # Random seed for reproducibility
        'use_single_ic': True,  # Train from single initial condition (ignore initial_state_range)
    },

    # MPC settings
    'mpc_defaults': {
        'prediction_horizon': 10,     # Number of steps to predict ahead
        'dt': 0.1,                     # MPC time step
        'Q': [1.0, 1.0, 1.0],          # State tracking weights [x1, x2, x3]
        'Ru': 0.001,                   # Control magnitude weight
        'R_deltau': 0.001,             # Control rate-of-change weight
        'u_min': -50.0,                # Minimum control input
        'u_max': 50.0,                 # Maximum control input
        'cost_type': 'quadratic',      # 'quadratic' or 'l1'
        'ftol': 1e-3,                  # Optimization tolerance
        'n_controls': 1,               # Number of control inputs
    }
}
