"""Configuration for HIV/AIDS Treatment environment with static controller."""
import torch
from rpasim.ode.classic_control.hiv import HIVTreatment
from rpa_control.controllers import StaticController, ControlledODE


def compute_steady_state():
    """Compute healthy steady state values from paper Eq. 6.2.

    Returns:
        x1_B, x3_B: Healthy steady state values for x1 and x3
    """
    # Parameters from HIVTreatment
    lambda_p = 1.0
    d = 0.1
    beta = 1.0
    a = 0.2
    p2 = 1.0
    c2 = 0.06
    b2 = 0.01
    q = 0.5
    h = 0.1

    # Compute x2^B from Eq. 6.2a
    term1 = c2 * (lambda_p - d * q) - b2 * beta
    term2_sq = term1**2 - 4 * beta * c2 * q * d * b2
    x2_B = (term1 - torch.sqrt(torch.tensor(term2_sq))) / (2 * beta * c2 * q)

    # Compute x1^B from Eq. 6.2a
    x1_B = lambda_p / (d + beta * x2_B)

    # Compute x5^B from Eq. 6.2b
    x5_B = (x2_B * c2 * (beta * q - a) + b2 * beta) / (c2 * p2 * x2_B)

    # Compute x3^B from Eq. 6.2b
    x3_B = h * x5_B / (c2 * q * x2_B)

    return x1_B, x3_B


def reward_fn(state, time=None):
    """Reward function from paper Eq. 6.3.

    J = ∫(x1 - x̂1) + (x3 - x̂3) + |u| dt

    We omit the |u| term as suggested.

    Args:
        state: State tensor [x1, x2, x3, x4, x5]
        time: Time value (optional)

    Returns:
        Reward (negative cost)
    """
    # Get steady state targets
    x1_B, x3_B = compute_steady_state()

    # Extract states
    x1 = state[0] if state.dim() == 1 else state[:, 0]  # healthy CD4+
    x3 = state[2] if state.dim() == 1 else state[:, 2]  # CTL precursor

    # Cost from paper Eq. 6.3 (without |u| term)
    # Using absolute values for proper tracking cost
    cost = torch.abs(x1 - x1_B) + torch.abs(x3 - x3_B)

    return -cost


def create_ode(controller_order=2, include_constant=True):
    """Create controlled HIV ODE instance."""
    # Create HIV ODE
    hiv_ode = HIVTreatment()

    # Create static controller
    # 5 state vars (x1-x5), 1 control output (HAART therapy level)
    controller = StaticController(
        n_state_vars=5,
        n_control_vars=1,
        order=controller_order,
        include_constant=include_constant
    )

    # Create controlled ODE
    # Control affects infection rate (modifies both x1 and x2 dynamics)
    controlled_ode = ControlledODE(
        base_ode=hiv_ode,
        controller=controller,
        control_indices=[0]  # HIVTreatment expects single control input
    )

    return controlled_ode


# Environment configuration
ENV_CONFIG = {
    'name': 'hiv',
    'experiment_name': 'hiv_static',
    'has_controller': True,  # This is a control problem

    # ODE setup
    'create_ode': create_ode,
    'reward_fn': reward_fn,

    # Initial conditions
    # From paper: x0 = (λ/d, 0.1, 0.1, 0.1, 0.1)
    # With λ = 1.0, d = 0.1: x0 = (10.0, 0.1, 0.1, 0.1, 0.1)
    'initial_state': torch.tensor([10.0, 0.1, 0.1, 0.1, 0.1]),
    'initial_state_range': None,  # Fixed initial condition

    # Fixed evaluation initial conditions
    'eval_initial_states': [
        torch.tensor([10.0, 0.1, 0.1, 0.1, 0.1]),  # Same as training (fixed IC)
    ],

    # Perturbation settings (13 fixed params total)
    'perturb_param_indices': None,  # Don't perturb for now (many parameters)

    # Display settings
    'param_names': None,  # Controller parameters (will be basis function names)
    'state_var_names': ['healthy_CD4', 'infected_CD4', 'CTL_precursor', 'CTL_indep', 'CTL_dep'],
    'control_names': ['u'],
    'target_vars': {0: 8.225466, 2: 1240.011475},  # Target x1 and x3 to healthy steady state

    # Description for display
    'description': """HIV/AIDS Treatment with HAART Therapy

System:
  dx1/dt = λ - d*x1 - β(1 - η*u)*x1*x2  (healthy CD4+ cells)
  dx2/dt = β(1 - η*u)*x1*x2 - a*x2 - p1*x4*x2 - p2*x5*x2  (infected CD4+ cells)
  dx3/dt = c2*x1*x2*x3 - c2*q*x2*x3 - b2*x3  (CTL precursors)
  dx4/dt = c1*x2*x4 - b1*x4  (helper-independent CTL)
  dx5/dt = c2*q*x2*x3 - h*x5  (helper-dependent CTL)

Controller: u = θ · Φ(x1, x2, x3, x4, x5)

Control objective: maintain healthy CD4+ cells and minimize infection
  Control limits: u ∈ [0, 1] (HAART therapy level)
  Therapy efficacy: η = 0.9799

Parameters:
  ODE fixed params: λ, d, β, a, p1, p2, c1, c2, b1, b2, q, h, η (13 total)
  Controller params: θ (polynomial basis coefficients)""",

    # Default training settings
    'defaults': {
        'time_horizon': 50.0,  # Training horizon (paper uses 50 weeks for full simulation)
        'n_reward_steps': 100,
        'steady_state_fraction': 0.3,
        'learning_rate': 0.001,
        'n_iterations': 100,
        'log_interval': 10,
        'eval_interval': 10,
        'controller_order': 1,  # Start with first-order
        'scale_aware_regularization': True,
        'state_limits': (0.0, 20.0),  # Reasonable limits for cell populations
        'seed': 42,  # Random seed for reproducibility
    }
}
