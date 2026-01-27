"""Configuration for Population Dynamics (Lotka-Volterra) environment with static controller."""
import torch
from rpasim.ode.classic_control.population import PopulationDynamics
from rpa_control.controllers import StaticController, ControlledODE


def reward_fn(state, time=None):
    """Reward function: stabilize critical point (prey=100, predator=20)."""
    critical_point = torch.tensor([100.0, 20.0])
    return -((state - critical_point) ** 2).sum()


def create_base_ode():
    """Create base population ODE (for MPC or uncontrolled simulation)."""
    return PopulationDynamics()


def create_ode(controller_order=2, include_constant=True):
    """Create controlled population ODE instance."""
    # Create population ODE
    pop_ode = create_base_ode()

    # Create static controller
    # 2 state vars (prey, predator), 1 control output (affects predator only)
    controller = StaticController(
        n_state_vars=2,
        n_control_vars=1,
        order=controller_order,
        include_constant=include_constant
    )

    # Create controlled ODE
    controlled_ode = ControlledODE(
        base_ode=pop_ode,
        controller=controller,
        control_indices=[1]  # Control affects predator (index 1)
    )

    return controlled_ode


# Environment configuration
ENV_CONFIG = {
    'name': 'population',
    'experiment_name': 'population_static',
    'has_controller': True,  # This is a control problem

    # ODE setup
    'create_base_ode': create_base_ode,
    'create_ode': create_ode,
    'reward_fn': reward_fn,

    # Initial conditions
    'initial_state': torch.tensor([80.0, 25.0]),
    'initial_state_range': [(60.0, 140.0), (10.0, 30.0)],  # Prey: [60, 140], Predator: [10, 30]

    # Fixed evaluation initial conditions (corners + center)
    'eval_initial_states': [
        torch.tensor([60.0, 10.0]),   # min-min corner
        torch.tensor([60.0, 30.0]),   # min-max corner
        torch.tensor([140.0, 10.0]),  # max-min corner
        torch.tensor([140.0, 30.0]),  # max-max corner
        torch.tensor([100.0, 20.0]),  # Critical point (center)
    ],

    # Perturbation settings (for ODE parameters [a, b, c, d])
    'perturb_param_indices': [0, 1, 2, 3],  # Perturb all 4 params

    # Display settings
    'param_names': None,  # Controller parameters (will be basis function names)
    'state_var_names': ['prey', 'predator'],
    'control_names': ['u'],
    'target_vars': {0: 100.0, 1: 20.0},  # Target values for plotting: {state_idx: value}

    # Description for display
    'description': """Population Dynamics (Lotka-Volterra) with Static Controller

System:
  dprey/dt = 0.50*prey - 0.03*prey*predator + u
  dpredator/dt = -0.50*predator + 0.00*prey*predator

Controller: u = θ · Φ(prey, predator)

Control objective: stabilize critical point
  prey = 100.0
  predator = 20.0

Parameters:
  ODE fixed params: a=0.50, b=0.03, c=0.50, d=0.00
  Controller params: θ (polynomial basis coefficients)""",

    # Default training settings
    'defaults': {
        'time_horizon': 20.0,
        'n_reward_steps': 100,
        'steady_state_fraction': 0.5,
        'learning_rate': 0.1,
        'n_iterations': 400,
        'log_interval': 50,
        'eval_interval': 50,
        'controller_order': 2,
        'scale_aware_regularization': True,
        'state_limits': (0.0, 200.0),  # Tighter limits for more reasonable penalties
        'seed': 42,  # Random seed for reproducibility
    },

    # MPC settings
    'mpc_defaults': {
        'prediction_horizon': 10,     # Number of steps to predict ahead
        'dt': 0.1,                     # MPC time step
        'Q': [1.0, 1.0],               # State tracking weights [prey, predator]
        'Ru': 0.5,                     # Control magnitude weight
        'R_deltau': 0.5,               # Control rate-of-change weight
        'u_min': -20.0,                # Minimum control input
        'u_max': 20.0,                 # Maximum control input
        'cost_type': 'quadratic',      # 'quadratic' or 'l1'
        'ftol': 1e-3,                  # Optimization tolerance
        'n_controls': 1,               # Number of control inputs
    }
}
