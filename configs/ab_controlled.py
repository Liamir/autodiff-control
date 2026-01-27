"""Configuration for ABControlled environment with learned controller.

The ABControlled ODE has control inputs that modulate the alpha parameters:
    dA/dt = alpha1*u[0] + alpha2*u[1]*A + alpha3*u[2]*B
    dB/dt = beta1*A - beta2*B

We learn a polynomial controller: u = theta * Phi(A, B)
"""
import torch
from rpasim.ode.ab import ABControlled
from rpa_control.controllers import StaticController, ControlledODE


def reward_fn(state, time=None):
    """Reward function: stabilize at target (B=1.0).
    """
    target_b = 1.0
    return -(state[1] - target_b) ** 2


# ODE parameters: [alpha1, alpha2, alpha3, beta1, beta2]
# With u=[1,1,1], steady state: A = -alpha1/(alpha2 + 4*alpha3), B = 4*A
FIXED_PARAMS = torch.tensor([1.0, -0.5, -0.25, 4.0, 1.0])
# Steady state: A = -1/(-0.5 + 4*(-0.25)) = -1/(-1.5) = 0.667, B = 2.667


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

    # Initialize controller to output u=[1,1,1] by default
    # The constant term "1" is the first basis function (index 0)
    # Setting params[:, 0] = 1.0 gives u = [1, 1, 1] + other_terms*0
    initial_params = torch.zeros(3, n_basis)
    if include_constant:
        initial_params[:, 0] = 1.0  # Constant term for all 3 control outputs

    # Create static controller
    # 2 state vars (A, B), 3 control outputs (u0, u1, u2)
    controller = StaticController(
        n_state_vars=2,
        n_control_vars=3,
        order=controller_order,
        include_constant=include_constant,
        initial_params=initial_params,
    )

    # Create controlled ODE
    # control_indices is required but ABControlled handles control internally
    # (it's multiplicative on the alpha terms, not additive on derivatives)
    controlled_ode = ControlledODE(
        base_ode=ab_ode,
        controller=controller,
        control_indices=[0, 1, 2]  # Placeholder - ABControlled uses control internally
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
    'perturb_param_indices': [0, 1, 2, 3, 4],  # Perturb all 5 params

    # Display settings
    'param_names': None,  # Controller parameters (will be basis function names)
    'state_var_names': ['A', 'B'],
    'control_names': ['u0', 'u1', 'u2'],
    'target_vars': {1: 1.0},  # Target value: B=1.0

    # Description for display
    'description': """ABControlled (Integral Feedback) with Static Controller

System:
  dA/dt = alpha1*u[0] + alpha2*u[1]*A + alpha3*u[2]*B
  dB/dt = beta1*A - beta2*B

Controller: [u0, u1, u2] = theta * Phi(A, B)

Control objective: stabilize at
  B = 1.0

Parameters (fixed):
  alpha1=1.0, alpha2=-0.5, alpha3=-0.25
  beta1=4.0, beta2=1.0

Note: With u=[1,1,1], the natural steady state is (A=0.667, B=2.667).""",

    # Default training settings
    'defaults': {
        'time_horizon': 20.0,
        'n_reward_steps': 100,
        'steady_state_fraction': 0.5,
        'learning_rate': 0.1,
        'n_iterations': 500,
        'log_interval': 50,
        'eval_interval': 50,
        'controller_order': 2,
        'scale_aware_regularization': True,
        'state_limits': (-20.0, 20.0),
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
        'n_controls': 3,           # 3 control inputs
    }
}
