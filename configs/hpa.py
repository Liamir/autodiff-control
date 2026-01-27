"""Configuration for HPA (Hypothalamic-Pituitary-Adrenal) axis environment.

The HPA model has 5 state variables and 9 control inputs:
    States: [CRH, ACTH, Cortisol, Pituitary, Adrenal]
    Controls: [I1, I2, I3, C1, C2, C3, A1, A2, A3]

Control inputs modulate hormone synthesis, receptor binding, and degradation.

Scenario:
    - Days 0-50: Baseline stress (u=1)
    - Days 50-150: Chronic stress (u=2)
    - Days 150+: Treatment phase (controller active, u=2 continues)

Goal: Restore Cortisol to 1.0 during treatment phase.
"""
import torch
from rpasim.ode.rpa.hpa import HPA
from rpa_control.controllers import StaticController, ControlledODE


# Stressor protocol parameters
BASELINE_END = 50.0      # End of baseline period (days)
STRESS_START = 50.0      # Start of chronic stress (days)
TREATMENT_START = 150.0  # Start of treatment (days)
BASELINE_STRESS = 1.0    # Normal stressor level
CHRONIC_STRESS = 2.0     # Elevated stressor level


def stressor(t):
    """Time-varying stressor input.

    u=1 for t < 50 (baseline)
    u=2 for t >= 50 (chronic stress, continues during treatment)
    """
    if t < STRESS_START:
        return BASELINE_STRESS
    else:
        return CHRONIC_STRESS


def reward_fn(state, time=None):
    """Reward function: stabilize Cortisol at 1.0.

    Only track Cortisol (state index 2), other states are free.
    """
    target_cortisol = 1.0
    return -((state[2] - target_cortisol) ** 2)


def create_base_ode():
    """Create base HPA ODE (for MPC or uncontrolled simulation)."""
    return HPA(stressor=stressor)


def create_ode(controller_order=2, include_constant=True):
    """Create controlled HPA ODE instance."""
    # Create HPA ODE
    hpa_ode = create_base_ode()

    # Calculate basis size to create initial params
    from rpa_control.controllers.basis import get_basis_size
    n_basis = get_basis_size(n_vars=5, order=controller_order, include_constant=include_constant)

    # Initialize controller to output u=[1,1,1,1,1,1,1,1,1] by default
    # The constant term "1" is the first basis function (index 0)
    # With 9 control outputs, set all constant terms to 1.0
    initial_params = torch.zeros(9, n_basis)
    if include_constant:
        initial_params[:, 0] = 1.0  # Constant term for all 9 control outputs

    # Create static controller
    # 5 state vars, 9 control outputs
    controller = StaticController(
        n_state_vars=5,
        n_control_vars=9,
        order=controller_order,
        include_constant=include_constant,
        initial_params=initial_params,
    )

    # Create controlled ODE
    # control_indices is required but HPA handles control internally
    controlled_ode = ControlledODE(
        base_ode=hpa_ode,
        controller=controller,
        control_indices=list(range(9))  # Placeholder - HPA uses control internally
    )

    return controlled_ode


# Environment configuration
ENV_CONFIG = {
    'name': 'hpa',
    'experiment_name': 'hpa_static',
    'has_controller': True,  # This is a control problem

    # ODE setup
    'create_base_ode': create_base_ode,
    'create_ode': create_ode,
    'reward_fn': reward_fn,

    # Initial conditions - all states start at 1.0
    'initial_state': torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0]),
    'initial_state_range': None,  # Single IC only

    # Fixed evaluation initial conditions
    'eval_initial_states': [
        torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0]),  # Target state
    ],

    # Perturbation settings (for ODE parameters)
    'perturb_param_indices': [],  # No perturbations

    # Display settings
    'param_names': None,  # Controller parameters (will be basis function names)
    'state_var_names': ['CRH', 'ACTH', 'Cortisol', 'Pituitary', 'Adrenal'],
    'control_names': ['I1', 'I2', 'I3', 'C1', 'C2', 'C3', 'A1', 'A2', 'A3'],
    'target_vars': {2: 1.0},  # Only track Cortisol

    # Description for display
    'description': """HPA Axis (Stress Response) with Static Controller

State Variables:
  x1: CRH, x2: ACTH, x3: Cortisol, P: Pituitary, A: Adrenal

Equations:
  dx1/dt = γ_x1 * (I1*u(t)*MR(C3*x3)*GR(C3*x3) - A1*x1)
  dx2/dt = γ_x2 * (I2*(C1*x1)*P*GR(C3*x3) - A2*x2)
  dx3/dt = γ_x3 * (I3*(C2*x2)*A - A3*x3)
  dP/dt = γ_P * P * ((C1*x1)*(1 - P/K_P) - 1)
  dA/dt = γ_A * A * ((C2*x2)*(1 - A/K_A) - 1)

Scenario:
  Days 0-50: Baseline (u=1)
  Days 50-150: Chronic stress (u=2)
  Days 150+: Treatment (controller active, u=2)

Controller: [I1-I3, C1-C3, A1-A3] = theta * Phi(states)

Control objective: restore Cortisol to 1.0

Control inputs (all default to 1.0):
  I1-I3: synthesis inhibitors [0, 1]
  C1-C3: receptor antagonists [0, 1]
  A1-A3: neutralizing antibodies [1, inf)""",

    # Default training settings
    'defaults': {
        'time_horizon': 250.0,  # 250 days total (50 baseline + 100 stress + 100 treatment)
        'n_reward_steps': 2000,
        'steady_state_fraction': 0.2,  # Track last 50 days of treatment (last 20% of trajectory)
        'learning_rate': 0.1,
        'n_iterations': 500,
        'log_interval': 50,
        'eval_interval': 50,
        'controller_order': 2,
        'scale_aware_regularization': True,
        'state_limits': (0.0, 100.0),
        'seed': 42,
        'use_single_ic': True,
    },

    # MPC settings
    'mpc_defaults': {
        'prediction_horizon': 50,
        'dt': 1.0,  # 1 day time step for MPC
        'Q': [0.0, 0.0, 1.0, 0.0, 0.0],  # Only track Cortisol
        'Ru': 0.1,
        'R_deltau': 0.1,
        'u_min': 0.0,   # Controls are non-negative
        'u_max': 10.0,
        'cost_type': 'quadratic',
        'ftol': 1e-3,
        'n_controls': 9,
    }
}
