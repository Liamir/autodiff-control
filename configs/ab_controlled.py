"""Configuration for ABControlled environment with learned controller.

The ABControlled ODE has simplified equations with single control input:
    dA/dt = u*alpha*(1-B)
    dB/dt = A - B

We learn a polynomial controller: u = theta * Phi(A, B)
"""
import torch
from rpasim.ode.rpa.ab import ABControlled
from rpa_control.controllers import StaticController, ControlledODE


def reward_fn(state, time=None):
    """Reward function: stabilize at target (B=0.0).
    """
    # target is B's norm as large as possible
    B_norm = torch.abs(state[1])
    return -1.0 + B_norm  # larger is better


# ODE parameter: alpha
FIXED_PARAMS = torch.tensor([50.0])


def create_base_ode():
    """Create base ABControlled ODE (for MPC or uncontrolled simulation)."""
    return ABControlled(fixed_params=FIXED_PARAMS)


def create_ode(controller_order=2, include_constant=True):
    """Create controlled ABControlled ODE instance."""
    # Create ABControlled ODE
    ab_ode = create_base_ode()

    # Calculate basis size to create initial params
    from rpa_control.controllers.basis import get_basis_size
    n_basis = get_basis_size(n_vars=2, order=controller_order, include_constant=include_constant)

    # Initialize controller to output u=1.0 by default
    # The constant term "1" is the first basis function (index 0)
    # Setting params[0, 0] = 1.0 gives u = 1.0 + other_terms*0
    initial_params = torch.zeros(1, n_basis)
    if include_constant:
        initial_params[0, 0] = 1.0  # Constant term for single control output

    # Create static controller
    # 2 state vars (A, B), 1 control output (u)
    controller = StaticController(
        n_state_vars=2,
        n_control_vars=1,
        order=controller_order,
        include_constant=include_constant,
        initial_params=initial_params,
    )

    # Create controlled ODE
    # control_indices is required but ABControlled handles control internally
    controlled_ode = ControlledODE(
        base_ode=ab_ode,
        controller=controller,
        control_indices=[0],  # Placeholder - ABControlled uses control internally
        control_bounds=(0.0, 1.0)  # Clip control between 0 and 1
    )

    return controlled_ode


# Environment configuration
ENV_CONFIG = {
    'name': 'ab_controlled',
    'experiment_name': 'ab_controlled_static',
    'has_controller': True,  # This is a control problem

    # ODE setup
    'create_base_ode': create_base_ode,
    'create_ode': create_ode,
    'reward_fn': reward_fn,

    # Initial conditions
    'initial_state': torch.tensor([0.5, 2.0]),
    'initial_state_range': [(0.2, 2.0), (0.5, 6.0)],  # A: [0.2, 2.0], B: [0.5, 6.0]

    # Fixed evaluation initial conditions
    'eval_initial_states': [
        torch.tensor([0.2, 0.5]),   # Low A, Low B
        torch.tensor([0.2, 6.0]),   # Low A, High B
        torch.tensor([2.0, 0.5]),   # High A, Low B
        torch.tensor([2.0, 6.0]),   # High A, High B
        torch.tensor([1.0, 1.0]),   # Target (steady state)
    ],

    # Perturbation settings (for ODE parameters)
    'perturb_param_indices': [0],  # Perturb alpha parameter

    # Display settings
    'param_names': None,  # Controller parameters (will be basis function names)
    'state_var_names': ['A', 'B'],
    'control_names': ['u'],
    # 'target_vars': {1: -100.0},  # Target value: B=0.0

    # Description for display
    'description': """ABControlled (Simplified) with Static Controller

System:
  dA/dt = u*alpha*(1-B)
  dB/dt = A - B

Controller: u = theta * Phi(A, B)

Control objective: stabilize at
  B = 0.0

Parameters (fixed):
  alpha = 1.0

Note: With u=1, system is unstable. Need active control to stabilize at B=0.""",

    # Default training settings
    'defaults': {
        'time_horizon': 20.0,
        'n_reward_steps': 100,
        'steady_state_fraction': 0.5,
        'learning_rate': 0.1,
        'n_iterations': 500,
        'log_interval': 10,
        'eval_interval': 50,
        'controller_order': 2,
        'scale_aware_regularization': True,
        'state_limits': (-20.0, 20.0),
        'control_bounds': (0.0, 1.0),  # Clip control signal between 0 and 1
        'seed': 42,
        'use_single_ic': True,  # Start with single IC for stability
    },

    # MPC settings
    'mpc_defaults': {
        'prediction_horizon': 20,
        'dt': 0.1,
        'Q': [1.0, 1.0],           # State tracking weights [A, B]
        'Ru': 0.1,                  # Control magnitude weight
        'R_deltau': 0.1,           # Control rate-of-change weight
        'u_min': -5.0,             # Minimum control input
        'u_max': 5.0,              # Maximum control input
        'cost_type': 'quadratic',
        'ftol': 1e-3,
        'n_controls': 1,           # Single control input
    }
}
