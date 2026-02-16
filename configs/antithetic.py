"""Configuration for Antithetic integral feedback motif with MLP controller.

The Antithetic system has parameter-level control:
    dZ1/dt = mu - (u[0] * eta) * Z1 * Z2
    dZ2/dt = theta * B - (u[0] * eta) * Z1 * Z2
    dB/dt  = Z1 - (u[1] * gamma) * B

Control inputs multiply eta (annihilation) and gamma (degradation).

Goal: Maximize |dB/dt| to force perpetual transients, defeating RPA.
The antithetic motif adapts to constant parameter changes (B_ss = mu/theta
independent of eta, gamma), so only time-varying control can keep B
away from steady state.

We use an MLP controller: [u_eta, u_gamma] = MLP(Z1, Z2, B)
"""
import torch
from rpasim.ode.rpa.antithetic import Antithetic
from rpa_control.controllers import MLPController, ControlledODE


# Fixed parameters: [mu, eta, theta, gamma]
FIXED_PARAMS = torch.tensor([1.0, 1.0, 1.0, 1.0])

# Calculate steady state for B (with all u[i]=1)
mu, eta, theta, gamma = FIXED_PARAMS
B_SS = mu / theta  # 1.0
# At SS: Z1_ss = gamma * B_ss, Z2_ss = mu / (eta * Z1_ss)
Z1_SS = gamma * B_SS
Z2_SS = mu / (eta * Z1_SS)
print(f"Steady state: B_ss = {B_SS:.4f}, Z1_ss = {Z1_SS:.4f}, Z2_ss = {Z2_SS:.4f}")


def reward_fn(state, time=None):
    """Reward function: maximize |dB/dt| (force perpetual transients).

    We compute dB/dt analytically from the ODE equation:
        dB/dt = Z1 - gamma * B

    With default (uncontrolled) parameters. This measures how far
    the system is from any steady state (where dB/dt = 0).

    Args:
        state: [Z1, Z2, B]
        time: Optional time (unused)

    Returns:
        Reward (to maximize)
    """
    Z1_val, _, B_val = state[0], state[1], state[2]
    dB_dt = Z1_val - gamma * B_val
    return dB_dt ** 2


def mpc_stage_cost_fn(x_next, u, x_curr=None, k=None):
    """Custom MPC cost: maximize |dB/dt| between steps (force transients).

    Uses finite difference (B_next - B_curr) as proxy for |dB/dt|.
    Control penalties use log-space for multiplicative controls.

    Args:
        x_next: Next state [Z1, Z2, B]
        u: Control input [u_eta, u_gamma]
        x_curr: Current state (optional)
        k: Time step (optional)

    Returns:
        Cost (to minimize)
    """
    # Maximize |dB/dt| approximated by (B_next - B_curr)^2
    # if x_curr is not None:
    #     dB = x_next[2] - x_curr[2]
    #     transient_reward = dB ** 2
    # else:
    #     # Fallback: use analytical dB/dt at x_next
    #     transient_reward = (x_next[0] - gamma * x_next[2]) ** 2
    transient_reward = 1000 * (x_next[2] - B_SS) ** 2

    # Control effort penalty in log-space (deviation from u=1 baseline)
    Ru = 0.01
    control_effort = Ru * torch.sum(torch.log(u) ** 2)

    # MPC minimizes cost
    cost = -transient_reward + control_effort
    return cost


def create_base_ode():
    """Create base Antithetic ODE (for MPC or uncontrolled simulation)."""
    return Antithetic(fixed_params=FIXED_PARAMS)


def create_ode(n_hidden=8, activation='tanh', controller_order=None, include_constant=None, **kwargs):
    """Create controlled Antithetic ODE instance with MLP controller.

    Args:
        n_hidden: Number of hidden layer neurons (default: 8)
        activation: Activation function - 'tanh', 'relu', or 'sigmoid' (default: 'tanh')
        controller_order: Ignored (MLP doesn't use polynomial basis)
        include_constant: Ignored (MLP doesn't use polynomial basis)
        **kwargs: Additional ignored arguments for compatibility
    """
    base_ode = create_base_ode()

    # Create MLP controller
    # 3 state vars (Z1, Z2, B), 2 control outputs (multipliers for eta, gamma)
    controller = MLPController(
        n_state_vars=3,
        n_control_vars=2,
        n_hidden=n_hidden,
        activation=activation,
    )

    # Create controlled ODE
    # control_indices is required but Antithetic handles control internally
    controlled_ode = ControlledODE(
        base_ode=base_ode,
        controller=controller,
        control_indices=[0, 2],  # Placeholder - base ODE uses control internally
        control_bounds=(0.5, 2.0),  # Parameter multiplier range
    )

    return controlled_ode


# Environment configuration
ENV_CONFIG = {
    'name': 'antithetic',
    'experiment_name': 'antithetic',
    'has_controller': True,

    # ODE setup
    'create_base_ode': create_base_ode,
    'create_ode': create_ode,
    'reward_fn': reward_fn,
    'mpc_stage_cost_fn': mpc_stage_cost_fn,

    # Initial conditions (at steady state)
    'initial_state': torch.tensor([Z1_SS.item(), Z2_SS.item(), B_SS.item()]),
    'initial_state_range': [(0.1, 3.0), (0.1, 3.0), (0.1, 3.0)],

    # Fixed evaluation initial conditions
    'eval_initial_states': [
        torch.tensor([0.5, 0.5, 0.5]),   # Below SS
        torch.tensor([2.0, 2.0, 2.0]),   # Above SS
        torch.tensor([0.5, 2.0, 1.0]),   # Mixed
        torch.tensor([2.0, 0.5, 1.0]),   # Mixed (opposite)
        torch.tensor([1.0, 1.0, 1.0]),   # At SS
    ],

    # Perturbation settings (for ODE parameters)
    'perturb_param_indices': [0, 1, 2, 3],

    # Display settings
    'param_names': None,  # MLP parameters
    'state_var_names': ['Z1', 'Z2', 'B'],
    'control_names': ['u_eta', 'u_gamma'],

    # Description for display
    'description': f"""Antithetic Integral Feedback with MLP Controller

System:
  dZ1/dt = mu - (u_eta * eta) * Z1 * Z2
  dZ2/dt = theta * B - (u_eta * eta) * Z1 * Z2
  dB/dt  = Z1 - (u_gamma * gamma) * B

Controller: [u_eta, u_gamma] = MLP(Z1, Z2, B)
  - Input: 3 neurons (Z1, Z2, B)
  - Hidden: configurable neurons (tanh activation)
  - Output: 2 neurons (parameter multipliers)

Control objective: Maximize |dB/dt| to force perpetual transients
  B_ss = mu / theta = {B_SS:.4f} (with u=1, independent of eta, gamma)

Parameters (fixed):
  mu    = {FIXED_PARAMS[0]:.2f}
  eta   = {FIXED_PARAMS[1]:.2f}
  theta = {FIXED_PARAMS[2]:.2f}
  gamma = {FIXED_PARAMS[3]:.2f}

Note: Control modulates eta and gamma directly.""",

    # Default training settings
    'defaults': {
        'time_horizon': 1.0,
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
        'dt': 0.1,
        'integration_substeps': 1,
        'Q': [1.0, 1.0, 1.0],
        'Ru': 0.0,
        'R_deltau': 0.0,
        'u_min': 0.5,
        'u_max': 2.0,
        'cost_type': 'quadratic',
        'ftol': 1e-3,
        'n_controls': 2,
    }
}
