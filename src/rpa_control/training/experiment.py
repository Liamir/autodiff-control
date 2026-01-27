"""Experiment configuration and setup utilities."""
import importlib.util
import torch
import numpy as np
import random
from pathlib import Path
from rpasim.env.base import DifferentiableEnv
from rpa_control.optimization.gradient import TrainingConfig
from rpa_control.utils import ExperimentLogger


def load_config(config_name: str):
    """Load environment configuration from configs directory.

    Args:
        config_name: Name of config file (without .py extension) or path to config file

    Returns:
        ENV_CONFIG dictionary from the config file
    """
    # Handle both config name and file path
    if config_name.endswith('.py'):
        config_path = Path(config_name)
    else:
        config_path = Path(__file__).parent.parent.parent.parent / 'configs' / f'{config_name}.py'

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    # Load the config module
    spec = importlib.util.spec_from_file_location("config", config_path)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)

    return config_module.ENV_CONFIG


def extract_model_params(ode):
    """Extract model parameters from an ODE for logging.

    Args:
        ode: ODE instance (either direct parameter ODE or ControlledODE)

    Returns:
        dict: Dictionary with parameter names and values
    """
    params_dict = {}

    # Check if this is a ControlledODE
    if hasattr(ode, 'base_ode'):
        # Controller-based ODE: extract base ODE parameters
        base_ode = ode.base_ode
        if hasattr(base_ode, 'fixed_params') and base_ode.fixed_params is not None:
            param_values = base_ode.fixed_params.detach().tolist()
            # Try to get parameter names from class attribute
            if hasattr(base_ode.__class__, 'fixed_param_names'):
                param_names = base_ode.__class__.fixed_param_names
                params_dict = dict(zip(param_names, param_values))
            else:
                params_dict = {f'param_{i}': v for i, v in enumerate(param_values)}
    else:
        # Direct parameter ODE: extract fixed_params
        if hasattr(ode, 'fixed_params') and ode.fixed_params is not None:
            param_values = ode.fixed_params.detach().tolist()
            if hasattr(ode.__class__, 'fixed_param_names'):
                param_names = ode.__class__.fixed_param_names
                params_dict = dict(zip(param_names, param_values))
            else:
                params_dict = {f'param_{i}': v for i, v in enumerate(param_values)}

    return params_dict


def setup_experiment(
    env_config: dict,
    experiment_name_suffix: str = None,
    seed: int = None,
):
    """Set up experiment logging and random seeds.

    Args:
        env_config: Environment configuration dictionary
        experiment_name_suffix: Optional suffix for experiment name
        seed: Random seed (None = use config default, -1 = disable seeding)

    Returns:
        tuple: (logger, actual_seed_used)
    """
    # Get defaults
    defaults = env_config.get('defaults', {})

    # Handle random seed
    if seed is None:
        seed = defaults.get('seed', None)

    if seed is not None and seed >= 0:
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        print(f"Random seed set to: {seed}")
        print()
    elif seed == -1:
        print("Random seeding disabled (using system randomness)")
        print()
        seed = None

    # Create experiment logger
    experiment_name = env_config.get('experiment_name', env_config['name'])
    if experiment_name_suffix:
        experiment_name = f"{experiment_name}_{experiment_name_suffix}"

    logger = ExperimentLogger(
        log_dir="logs",
        experiment_name=experiment_name
    )

    return logger, seed


def create_ode_from_config(
    env_config: dict,
    controller_order: int = None,
    include_constant: bool = None,
    model_params: dict = None,
):
    """Create ODE instance from environment configuration.

    Args:
        env_config: Environment configuration dictionary
        controller_order: Controller polynomial order (for control problems)
        include_constant: Include constant in controller (for control problems)
        model_params: Optional dict of model parameters to override defaults

    Returns:
        ODE instance
    """
    create_ode = env_config['create_ode']
    has_controller = env_config.get('has_controller', True)

    if has_controller:
        # Controller-based ODE
        defaults = env_config.get('defaults', {})
        if controller_order is None:
            controller_order = defaults.get('controller_order', 2)
        if include_constant is None:
            include_constant = defaults.get('include_constant', True)

        ode = create_ode(controller_order=controller_order, include_constant=include_constant)
    else:
        # Circuit design ODE
        ode = create_ode()

    # Override model parameters if provided
    if model_params is not None:
        _apply_model_params(ode, model_params)

    return ode


def _apply_model_params(ode, model_params: dict):
    """Apply model parameter overrides to an ODE.

    Args:
        ode: ODE instance
        model_params: Dictionary of parameter names and values
    """
    # Get the base ODE (could be wrapped in ControlledODE)
    base_ode = ode.base_ode if hasattr(ode, 'base_ode') else ode

    if not hasattr(base_ode, 'fixed_params') or base_ode.fixed_params is None:
        return

    # Get parameter names
    if not hasattr(base_ode.__class__, 'fixed_param_names'):
        print("Warning: Cannot apply model params - no parameter names defined")
        return

    param_names = base_ode.__class__.fixed_param_names
    current_params = base_ode.fixed_params.clone()

    # Update parameters
    for name, value in model_params.items():
        if name in param_names:
            idx = param_names.index(name)
            current_params[idx] = value
        else:
            print(f"Warning: Unknown parameter '{name}' ignored")

    # Apply updated parameters
    base_ode.fixed_params = current_params


def create_environment(
    ode,
    initial_state,
    initial_state_range,
    eval_initial_states,
    time_horizon: float,
    n_reward_steps: int,
    reward_fn,
    state_limits=None,
):
    """Create differentiable environment for training.

    Args:
        ode: ODE instance
        initial_state: Initial state tensor
        initial_state_range: Range for sampling initial states
        eval_initial_states: Fixed initial states for evaluation
        time_horizon: Simulation time horizon
        n_reward_steps: Number of reward computation steps
        reward_fn: Reward function
        state_limits: Optional (min, max) tuple for state bounds

    Returns:
        DifferentiableEnv instance
    """
    env = DifferentiableEnv(
        initial_ode=ode,
        reward_fn=reward_fn,
        initial_state=initial_state,
        initial_state_range=initial_state_range,
        time_horizon=time_horizon,
        n_reward_steps=n_reward_steps,
        state_limits=state_limits,
    )
    return env


def create_training_config(
    n_iterations: int,
    learning_rate: float,
    l1_penalty: float = 0.0,
    l2_penalty: float = 0.0,
    steady_state_fraction: float = 0.0,
    scale_aware_regularization: bool = False,
    perturb_param_indices=None,
    perturb_fold_change: float = 2.0,
    eval_initial_states=None,
    n_param_samples_eval: int = 10,
    log_interval: int = 50,
    eval_interval: int = 50,
    use_three_stage: bool = False,
    n_iterations_stage1: int = 500,
    n_iterations_stage2: int = 300,
    l1_penalty_stage2: float = 0.01,
    n_iterations_stage3: int = 300,
    threshold_value_stage3: float = 1e-3,
):
    """Create training configuration.

    Args:
        n_iterations: Number of training iterations
        learning_rate: Learning rate
        l1_penalty: L1 regularization penalty
        l2_penalty: L2 regularization penalty
        steady_state_fraction: Fraction of trajectory to skip
        scale_aware_regularization: Enable basis normalization
        perturb_param_indices: Indices of parameters to perturb
        perturb_fold_change: Fold change for perturbations
        eval_initial_states: Fixed states for evaluation
        n_param_samples_eval: Number of parameter samples for evaluation
        log_interval: Logging interval
        eval_interval: Evaluation interval
        use_three_stage: Enable three-stage training
        n_iterations_stage1: Stage 1 iterations
        n_iterations_stage2: Stage 2 iterations
        l1_penalty_stage2: Stage 2 L1 penalty
        n_iterations_stage3: Stage 3 iterations
        threshold_value_stage3: Stage 3 threshold

    Returns:
        TrainingConfig instance
    """
    return TrainingConfig(
        n_iterations=n_iterations,
        learning_rate=learning_rate,
        l1_penalty=l1_penalty,
        l2_penalty=l2_penalty,
        steady_state_fraction=steady_state_fraction,
        scale_aware_regularization=scale_aware_regularization,
        perturb_param_indices=perturb_param_indices,
        perturb_fold_change=perturb_fold_change,
        eval_initial_states=eval_initial_states,
        n_param_samples_eval=n_param_samples_eval,
        log_interval=log_interval,
        eval_interval=eval_interval,
        use_three_stage=use_three_stage,
        n_iterations_stage1=n_iterations_stage1,
        n_iterations_stage2=n_iterations_stage2,
        l1_penalty_stage2=l1_penalty_stage2,
        n_iterations_stage3=n_iterations_stage3,
        threshold_value_stage3=threshold_value_stage3,
    )
