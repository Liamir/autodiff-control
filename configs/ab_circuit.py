"""Configuration for AB gene regulatory network environment."""
import torch
from rpasim.ode.ab import AB


def reward_fn(state, time=None):
    """Reward function: drive B (second variable) to 1.0."""
    target_b = 1.0
    return -(state[1] - target_b) ** 2


def create_ode(differentiable_params=None, fixed_params=None):
    """Create AB ODE instance."""
    if differentiable_params is None:
        differentiable_params = torch.tensor([1.0, -0.25, -0.25], requires_grad=True)
    if fixed_params is None:
        fixed_params = torch.tensor([1.0, 1.0])

    return AB(
        differentiable_params=differentiable_params,
        fixed_params=fixed_params,
    )


# Environment configuration
ENV_CONFIG = {
    'name': 'ab_circuit',
    'experiment_name': 'circuit_ab',

    # ODE setup
    'create_ode': create_ode,
    'reward_fn': reward_fn,

    # Initial conditions
    'initial_state': torch.tensor([1.0, 0.5]),
    'initial_state_range': None,  # No randomization for AB

    # Fixed evaluation initial conditions
    'eval_initial_states': [
        torch.tensor([0.5, 0.25]),   # Low A, Low B
        torch.tensor([0.5, 0.75]),   # Low A, High B
        torch.tensor([1.5, 0.25]),   # High A, Low B
        torch.tensor([1.5, 0.75]),   # High A, High B
        torch.tensor([1.0, 0.5]),    # Center
    ],

    # Perturbation settings
    'perturb_param_indices': [0, 1],  # Perturb beta1, beta2

    # Display settings
    'param_names': ['alpha1', 'alpha2', 'alpha3'],
    'state_var_names': ['A', 'B'],
    'target_var_idx': 1,  # B is the target
    'target_value': 1.0,

    # Description for display
    'description': """AB Gene Regulatory Network

Equations:
  dA/dt = alpha1*beta1/(1 + B^2) - A
  dB/dt = alpha2*A^2/(alpha3 + A^2)*beta2 - B

Control objective: drive B to 1.0

Parameters:
  Differentiable: alpha1, alpha2, alpha3
  Fixed: beta1, beta2""",

    # Default training settings (can be overridden)
    'defaults': {
        'time_horizon': 20.0,
        'n_reward_steps': 100,
        'steady_state_fraction': 0.5,
        'learning_rate': 0.01,
        'n_iterations': 1000,
        'log_interval': 100,
        'eval_interval': 100,
    }
}
