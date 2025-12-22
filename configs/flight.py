"""Configuration for Flight Control System (F-8 Crusader) environment with static controller."""
import torch
from rpasim.ode.classic_control.flight import FlightControl
from rpa_control.controllers import StaticController, ControlledODE


def reference_trajectory(t):
    """Reference trajectory for angle of attack x1(t).

    From paper Eq. 5.2:
    r(t) = 0.4*(-0.5/(1+exp(-t/0.1-0.8)) + 1/(1+exp(t/0.1-3)) - 0.4)

    Args:
        t: Time (scalar or tensor)

    Returns:
        Reference value for x1 (angle of attack)
    """
    term1 = -0.5 / (1 + torch.exp(-t / 0.1 - 0.8))
    term2 = 1.0 / (1 + torch.exp(t / 0.1 - 3))
    return 0.4 * (term1 + term2 - 0.4)


def reward_fn(state, time=None):
    """Reward function: track reference trajectory for x1.

    From paper Eq. 5.2:
    - Q = 25 (tracking weight for x1)
    - R = 0.05 (control effort - handled separately)

    Args:
        state: State tensor [x1, x2, x3] or batch of states
        time: Time value (optional, for time-varying reference)

    Returns:
        Reward (negative tracking error)
    """
    # Extract x1 (angle of attack)
    x1 = state[0] if state.dim() == 1 else state[:, 0]

    # Reference trajectory (if time provided, otherwise track zero)
    if time is not None:
        ref = reference_trajectory(time)
    else:
        ref = torch.tensor(0.0)

    # Tracking error with Q = 25
    Q = 25.0
    tracking_error = Q * (x1 - ref) ** 2

    return -tracking_error


def create_ode(controller_order=2, include_constant=True):
    """Create controlled flight ODE instance."""
    # Create flight ODE
    flight_ode = FlightControl()

    # Create static controller
    # 3 state vars (x1, x2, x3), 1 control output (tail deflection)
    controller = StaticController(
        n_state_vars=3,
        n_control_vars=1,
        order=controller_order,
        include_constant=include_constant
    )

    # Create controlled ODE
    # Control affects both x1 and x3 (via nonlinear terms in FlightControl)
    # But we pass it as a single control input that the ODE handles
    controlled_ode = ControlledODE(
        base_ode=flight_ode,
        controller=controller,
        control_indices=[0]  # FlightControl expects single control input
    )

    return controlled_ode


# Environment configuration
ENV_CONFIG = {
    'name': 'flight',
    'experiment_name': 'flight_static',

    # ODE setup
    'create_ode': create_ode,
    'reward_fn': reward_fn,

    # Initial conditions
    # NOTE: These initial states are uncertain - not specified in the paper.
    # Chosen based on:
    # - x1 constraint from paper: [-0.2, 0.4]
    # - Staying near equilibrium [0, 0, 0]
    # - Avoiding extreme values that might cause numerical issues
    # May need adjustment based on training results or domain knowledge.
    'initial_state': torch.tensor([0.1, 0.0, 0.0]),  # Slightly perturbed from zero
    'initial_state_range': [(-0.1, 0.3), (-0.2, 0.2), (-0.5, 0.5)],  # x1, x2, x3

    # Fixed evaluation initial conditions
    'eval_initial_states': [
        torch.tensor([0.0, 0.0, 0.0]),      # Equilibrium
        torch.tensor([0.2, 0.0, 0.0]),      # Perturbed x1
        torch.tensor([0.0, 0.2, 0.0]),      # Perturbed x2
        torch.tensor([0.0, 0.0, 0.3]),      # Perturbed x3
        torch.tensor([0.1, 0.1, 0.1]),      # All perturbed
    ],

    # Perturbation settings (for ODE parameters - FlightControl has 12 fixed params)
    'perturb_param_indices': None,  # Don't perturb for now (too many params)

    # Display settings
    'param_names': None,  # Controller parameters (will be basis function names)
    'state_var_names': ['x1', 'x2', 'x3'],
    'control_names': ['u'],
    'target_var_idx': 0,  # Track x1 (angle of attack)
    'target_value': 0.0,   # Target equilibrium (or reference trajectory)

    # Description for display
    'description': """Flight Control System (F-8 Crusader)

System:
  dx1/dt = -0.877*x1 + x3 - 0.088*x1*x3 + 0.47*x1^2 - 0.019*x2^2 - x1^2*x3 + 3.846*x1^3
           + delta_i * u terms (nonlinear control coupling)
  dx2/dt = x3
  dx3/dt = -4.208*x1 - 0.396*x3 - 0.47*x1^2 - 3.564*x1^3
           + epsilon_i * u terms (nonlinear control coupling)

Controller: u = θ · Φ(x1, x2, x3)

Control objective: track reference trajectory for x1 (angle of attack)
  Control limits: u ∈ [-0.3, 0.5] rad (tail deflection)
  State constraint: x1 ∈ [-0.2, 0.4] rad

Parameters:
  ODE fixed params: alpha_i, beta_i, gamma_i (12 total)
  Controller params: θ (polynomial basis coefficients)""",

    # Default training settings
    'defaults': {
        'time_horizon': 13.0,  # From paper
        'n_reward_steps': 100,
        'steady_state_fraction': 0.3,  # Shorter settling time for tracking
        'learning_rate': 0.05,
        'n_iterations': 500,
        'log_interval': 50,
        'eval_interval': 50,
        'controller_order': 2,
        'scale_aware_regularization': True,
        'state_limits': (-1.0, 1.0),  # Reasonable limits for normalized states
    }
}
