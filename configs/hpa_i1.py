"""Configuration for HPA axis with I1-only control.

Same as hpa.py but only I1 (CRH synthesis inhibitor) is learnable.
Other controls (I2, I3, C1-C3, A1-A3) are fixed at 1.0.

The HPA model has 5 state variables and 9 control inputs:
    States: [CRH, ACTH, Cortisol, Pituitary, Adrenal]
    Controls: [I1, I2, I3, C1, C2, C3, A1, A2, A3]

Example Scenario:
    - Days 0-50: Baseline stress (u=1)
    - Days 50-150: Chronic stress (u=2)
    - Days 150+: Treatment phase (I1 controller active, u=2 continues)

Goal: Restore Cortisol to 1.0 using only I1. Should be possible according to the paper.
"""
import torch
from rpasim.ode.rpa.hpa import HPA
from rpa_control.controllers import StaticController


BASELINE_END = 0.0      # End of baseline period (days)
STRESS_START = 0.0      # Start of chronic stress (days)
TREATMENT_START = 0.0   # Start of treatment (days)
BASELINE_STRESS = 1.0    # Normal stressor level
CHRONIC_STRESS = 2.0     # Elevated stressor level
# initial state after 50 days of baseline stress and 200 days of chronic stress
HIGH_CORTISOL_IC = torch.tensor([0.981693, 0.996662, 1.853058, 1.116182, 1.859223], dtype=torch.float32)

# Which controls are active (learnable). Others fixed at 1.0.
# Control order: [I1, I2, I3, C1, C2, C3, A1, A2, A3]
ACTIVE_CONTROLS = [0]  # Only I1 is learnable


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


from rpasim.ode import ODE


class PaddedControlledODE(ODE):
    """ControlledODE variant that pads control outputs to a fixed size.

    Used when the base ODE expects more control inputs than we want to learn.
    Non-active controls are fixed at default_value (1.0).
    """

    def __init__(self, base_ode, controller, active_indices, n_total_controls=9, default_value=1.0):
        self.base_ode = base_ode
        self.controller = controller
        self.active_indices = active_indices
        self.n_total_controls = n_total_controls
        self.default_value = default_value

        # Get base ODE state dimension
        if hasattr(base_ode, 'variable_names'):
            self.base_state_dim = len(base_ode.variable_names)
        else:
            raise ValueError("Cannot infer state dimension from base ODE")

        self.state_dim = self.base_state_dim

        # Initialize parent ODE class with controller params as differentiable
        super().__init__(
            differentiable_params=controller.params,
            fixed_params=base_ode.fixed_params
        )

    def forward(self, t, state, differentiable_params=None, fixed_params=None):
        """Compute state derivative with padded control."""
        # Get active control from controller
        active_control = self.controller(state)

        # Pad to full control vector
        full_control = torch.ones(self.n_total_controls, dtype=state.dtype) * self.default_value
        for i, idx in enumerate(self.active_indices):
            full_control[idx] = active_control[i]

        # Compute base ODE dynamics with full control
        derivative = self.base_ode(t, state, None, fixed_params, control=full_control)

        return derivative

    def update_controller_params(self):
        """Sync controller params from differentiable_params."""
        self.controller.params = self.differentiable_params.reshape(self.controller.params.shape)

    def get_controller_summary(self, state_var_names, control_names=None, threshold=1e-3):
        """Get human-readable summary of controller."""
        self.update_controller_params()
        # Use only active control names
        if control_names is not None:
            active_names = [control_names[i] for i in self.active_indices]
        else:
            active_names = None
        return self.controller.get_param_summary(state_var_names, active_names, threshold)

    def __str__(self):
        return f"PaddedControlledODE({self.base_ode.name}, active={self.active_indices})"


def create_ode(controller_order=2, include_constant=True):
    """Create controlled HPA ODE instance with I1-only control."""
    # Create HPA ODE
    hpa_ode = create_base_ode()

    # Calculate basis size to create initial params
    from rpa_control.controllers.basis import get_basis_size
    n_basis = get_basis_size(n_vars=5, order=controller_order, include_constant=include_constant)

    # Number of active controls (just I1)
    n_active = len(ACTIVE_CONTROLS)

    # Initialize controller to output 1.0 by default
    initial_params = torch.zeros(n_active, n_basis, dtype=torch.float32)
    if include_constant:
        initial_params[:, 0] = 1.0  # Constant term

    # Create static controller for active controls only
    controller = StaticController(
        n_state_vars=5,
        n_control_vars=n_active,
        order=controller_order,
        include_constant=include_constant,
        initial_params=initial_params,
    )

    # Create padded controlled ODE
    controlled_ode = PaddedControlledODE(
        base_ode=hpa_ode,
        controller=controller,
        active_indices=ACTIVE_CONTROLS,
        n_total_controls=9,
        default_value=1.0,
    )

    return controlled_ode


# Environment configuration
ENV_CONFIG = {
    'name': 'hpa_i1',
    'experiment_name': 'hpa_i1_static',
    'has_controller': True,  # This is a control problem (vs circuit design)

    # ODE setup
    'create_base_ode': create_base_ode,
    'create_ode': create_ode,
    'reward_fn': reward_fn,

    # Initial conditions - all states start at 1.0
    # initial state after 50 days of baseline stress and 200 days of chronic stress
    'initial_state': HIGH_CORTISOL_IC,
    'initial_state_range': None,  # Single IC only

    # Fixed evaluation initial conditions
    'eval_initial_states': [
        torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0], dtype=torch.float32),  # Target state
    ],

    # Perturbation settings (for ODE parameters)
    'perturb_param_indices': [],  # No perturbations

    # Display settings
    'param_names': None,  # Controller parameters (will be basis function names)
    'state_var_names': ['CRH', 'ACTH', 'Cortisol', 'Pituitary', 'Adrenal'],
    'control_names': ['I1'],  # Only I1 is learnable
    'target_vars': {2: 1.0},  # Only track Cortisol

    # Description for display
    'description': """HPA Axis with I1-only Control

State Variables:
  x1: CRH, x2: ACTH, x3: Cortisol, P: Pituitary, A: Adrenal

Equations:
  dx1/dt = γ_x1 * (I1*u(t)*MR(C3*x3)*GR(C3*x3) - A1*x1)
  dx2/dt = γ_x2 * (I2*(C1*x1)*P*GR(C3*x3) - A2*x2)
  dx3/dt = γ_x3 * (I3*(C2*x2)*A - A3*x3)
  dP/dt = γ_P * P * ((C1*x1)*(1 - P/K_P) - 1)
  dA/dt = γ_A * A * ((C2*x2)*(1 - A/K_A) - 1)

Scenario (scaled to 50 days):
  Days 0-10: Baseline (u=1)
  Days 10-30: Chronic stress (u=2)
  Days 30-50: Treatment (I1 controller active, u=2)

Controller: I1 = theta * Phi(states)
Fixed controls: I2=I3=C1=C2=C3=A1=A2=A3=1.0

Control objective: restore Cortisol to 1.0 using only I1""",

    # Default training settings
    'defaults': {
        'time_horizon': 200.0,  # Reduced for testing (baseline only)
        'n_reward_steps': 100,
        'steady_state_fraction': 0.2,  # Track last 10 days (second half of treatment, days 40-50)
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
        'u_min': 0.0,   # I1 is non-negative
        'u_max': 10.0,
        'cost_type': 'quadratic',
        'ftol': 1e-3,
        'n_controls': 1,  # Only I1
    }
}
