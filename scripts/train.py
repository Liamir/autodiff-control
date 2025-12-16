"""Generic training script that works with any environment configuration."""
import fire
import torch
import importlib.util
from pathlib import Path
from rpasim.env.base import DifferentiableEnv
from rpa_control.optimization.gradient import train_ode_parameters, TrainingConfig
from rpa_control.utils.plotting import plot_training_comparison, plot_training_curves
from rpa_control.utils import ExperimentLogger
from rpa_control.style import set_style


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
        config_path = Path(__file__).parent.parent / 'configs' / f'{config_name}.py'

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    # Load the config module
    spec = importlib.util.spec_from_file_location("config", config_path)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)

    return config_module.ENV_CONFIG


def train(
    config: str,
    # Training parameters
    n_iterations: int = None,
    learning_rate: float = None,
    l1_penalty: float = 0.0,
    l2_penalty: float = 0.0,
    time_horizon: float = None,
    log_interval: int = None,
    eval_interval: int = None,
    save_plot: bool = True,
    steady_state_fraction: float = None,
    # Scale-aware regularization
    scale_aware_regularization: bool = None,
    # Three-stage training
    use_three_stage: bool = False,
    n_iterations_stage1: int = 500,
    n_iterations_stage2: int = 300,
    l1_penalty_stage2: float = 0.01,
    n_iterations_stage3: int = 300,
    threshold_value_stage3: float = 1e-3,
    # Perturbation settings
    perturb_params: bool = False,
    perturb_fold_change: float = 2.0,
    n_param_samples_eval: int = 10,
    # Controller-specific (for static/dynamic controllers)
    controller_order: int = None,
    include_constant: bool = None,
):
    """
    Generic training script for any environment configuration.

    Args:
        config: Name of config file (e.g., 'ab_circuit', 'population') or path to config file
        n_iterations: Number of training iterations (overrides default)
        learning_rate: Learning rate for optimizer (overrides default)
        l1_penalty: L1 regularization coefficient (for sparsity)
        l2_penalty: L2 regularization coefficient
        time_horizon: Simulation time horizon (overrides default)
        log_interval: Print progress every N iterations (overrides default)
        eval_interval: Evaluate on fixed ICs every N iterations (overrides default)
        save_plot: Whether to save training curves
        steady_state_fraction: Fraction of trajectory to skip before computing reward (overrides default)
        scale_aware_regularization: Use basis normalization (overrides default)
        use_three_stage: Use 3-stage training: normal → L1 reg → thresholding
        n_iterations_stage1: Iterations for stage 1
        n_iterations_stage2: Iterations for stage 2
        l1_penalty_stage2: L1 penalty for stage 2
        n_iterations_stage3: Iterations for stage 3
        threshold_value_stage3: Threshold for stage 3
        perturb_params: Whether to perturb parameters during training
        perturb_fold_change: Fold change for parameter perturbations
        n_param_samples_eval: Number of parameter samples per IC during evaluation
        controller_order: Polynomial order for controller (overrides default if applicable)
        include_constant: Include constant term in controller (overrides default if applicable)
    """
    set_style()

    # Load environment configuration
    env_config = load_config(config)

    # Get defaults from config
    defaults = env_config.get('defaults', {})

    # Apply defaults for parameters not specified
    if n_iterations is None:
        n_iterations = defaults.get('n_iterations', 1000)
    if learning_rate is None:
        learning_rate = defaults.get('learning_rate', 0.01)
    if time_horizon is None:
        time_horizon = defaults.get('time_horizon', 20.0)
    if log_interval is None:
        log_interval = defaults.get('log_interval', 100)
    if eval_interval is None:
        eval_interval = defaults.get('eval_interval', 100)
    if steady_state_fraction is None:
        steady_state_fraction = defaults.get('steady_state_fraction', 0.5)
    if scale_aware_regularization is None:
        scale_aware_regularization = defaults.get('scale_aware_regularization', False)
    if controller_order is None:
        controller_order = defaults.get('controller_order', 2)
    if include_constant is None:
        include_constant = defaults.get('include_constant', True)

    # Create experiment logger
    experiment_name = env_config.get('experiment_name', env_config['name'])
    logger = ExperimentLogger(
        log_dir="logs",
        experiment_name=experiment_name
    )

    # Start capturing console output
    logger.start_capture()

    print("="*60)
    print(f"Experiment: {logger.experiment_name}")
    print(f"Config: {config}")
    print(f"Log directory: {logger.get_experiment_path()}")
    print("="*60)
    print()

    # Print environment description
    if 'description' in env_config:
        print(env_config['description'])
        print()

    # Create ODE
    # Check if create_ode accepts controller_order (for population)
    create_ode = env_config['create_ode']
    import inspect
    sig = inspect.signature(create_ode)

    if 'controller_order' in sig.parameters:
        # Controller-based ODE
        ode = create_ode(controller_order=controller_order, include_constant=include_constant)
        print(f"Controller order: {controller_order}")
        print(f"Include constant: {include_constant}")
    else:
        # Direct ODE (like AB circuit)
        ode = create_ode()

    print(f"ODE created: {type(ode).__name__}")
    print()

    # Get initial state and evaluation states
    initial_state = env_config['initial_state']
    initial_state_range = env_config.get('initial_state_range', None)
    eval_initial_states = env_config.get('eval_initial_states', [initial_state])

    print(f"Initial state: {initial_state.tolist()}")
    if initial_state_range is not None:
        print(f"Initial state range: {initial_state_range}")
    print(f"Evaluation: {len(eval_initial_states)} fixed initial conditions")
    print()

    # Get perturbation settings
    perturb_param_indices = env_config.get('perturb_param_indices', None) if perturb_params else None

    # Get state limits from config if available
    state_limits = defaults.get('state_limits', None)

    # Create environment
    env = DifferentiableEnv(
        initial_ode=ode,
        reward_fn=env_config['reward_fn'],
        initial_state=initial_state,
        initial_state_range=initial_state_range,
        time_horizon=time_horizon,
        n_reward_steps=100,
        state_limits=state_limits,
    )

    # Training configuration
    training_config = TrainingConfig(
        n_iterations=n_iterations,
        learning_rate=learning_rate,
        l1_penalty=l1_penalty,
        l2_penalty=l2_penalty,
        log_interval=log_interval,
        eval_interval=eval_interval,
        verbose=True,
        steady_state_fraction=steady_state_fraction,
        scale_aware_regularization=scale_aware_regularization,
        use_three_stage=use_three_stage,
        n_iterations_stage1=n_iterations_stage1,
        n_iterations_stage2=n_iterations_stage2,
        l1_penalty_stage2=l1_penalty_stage2,
        n_iterations_stage3=n_iterations_stage3,
        threshold_value_stage3=threshold_value_stage3,
        perturb_param_indices=perturb_param_indices,
        perturb_fold_change=perturb_fold_change,
        n_param_samples_eval=n_param_samples_eval,
        eval_initial_states=eval_initial_states,
    )

    # Log configuration
    config_dict = {
        'config_name': config,
        'environment': env_config['name'],
        'experiment_name': experiment_name,
        'n_iterations': n_iterations,
        'learning_rate': learning_rate,
        'l1_penalty': l1_penalty,
        'l2_penalty': l2_penalty,
        'time_horizon': time_horizon,
        'steady_state_fraction': steady_state_fraction,
        'scale_aware_regularization': scale_aware_regularization,
        'use_three_stage': use_three_stage,
        'n_iterations_stage1': n_iterations_stage1,
        'n_iterations_stage2': n_iterations_stage2,
        'l1_penalty_stage2': l1_penalty_stage2,
        'n_iterations_stage3': n_iterations_stage3,
        'threshold_value_stage3': threshold_value_stage3,
        'perturb_params': perturb_params,
        'perturb_param_indices': perturb_param_indices,
        'perturb_fold_change': perturb_fold_change,
        'n_param_samples_eval': n_param_samples_eval,
        'eval_interval': eval_interval,
        'initial_state': initial_state.tolist(),
        'initial_state_range': initial_state_range,
        'eval_initial_states': [ic.tolist() for ic in eval_initial_states],
    }

    # Add controller-specific config
    if 'controller_order' in sig.parameters:
        config_dict['controller_order'] = controller_order
        config_dict['include_constant'] = include_constant

    logger.log_config(config_dict)

    print("Starting training...")
    print(f"Time horizon: {time_horizon}")
    print(f"Learning rate: {learning_rate}")
    print(f"Iterations: {n_iterations}")
    if perturb_params:
        print(f"Parameter perturbations: enabled (fold change: {perturb_fold_change})")
        print(f"  Perturbing indices: {perturb_param_indices}")
        print(f"  Eval samples per IC: {n_param_samples_eval}")
    if eval_interval > 0:
        print(f"Evaluation interval: {eval_interval}")
    print()

    # Train
    history = train_ode_parameters(
        env=env,
        ode=ode,
        config=training_config,
    )

    print()
    print("Training complete!")
    print(f"Final loss: {history['loss'][-1]:.6f}")
    print(f"Final reward: {history['reward'][-1]:.6f}")
    print(f"Best reward (single iter): {history['best_reward']:.6f}")

    if eval_interval > 0 and 'eval_reward' in history:
        eval_rewards = [r for r in history['eval_reward'] if r is not None]
        if eval_rewards:
            best_eval_reward = max(eval_rewards)
            print(f"Best eval reward: {best_eval_reward:.6f}")

    print(f"Non-zero params: {history['num_nonzero_params'][-1]}")

    # Format best parameters
    best_params_flat = history['best_params'].flatten()
    best_params_str = "[" + ", ".join([f"{p.item():.6f}" for p in best_params_flat]) + "]"
    print(f"Best params: {best_params_str}")
    print()

    # Print parameter summary (if param_names provided)
    param_names = env_config.get('param_names', None)
    param_summary = ""

    if hasattr(ode, 'get_controller_summary'):
        # Controller-based ODE
        state_var_names = env_config.get('state_var_names', None)
        control_names = env_config.get('control_names', ['u'])
        print("Trained Controller:")
        print("-"*60)
        param_summary = ode.get_controller_summary(state_var_names, control_names)
        print(param_summary)
    elif param_names is not None:
        # Direct parameter ODE (like AB circuit)
        print("Trained Parameters:")
        print("-"*60)
        if hasattr(ode, 'differentiable_params'):
            params = ode.differentiable_params.detach()
            for name, value in zip(param_names, params):
                print(f"  {name} = {value.item():.6f}")
                param_summary += f"  {name} = {value.item():.6f}\n"
        else:
            print("  (parameters stored in ODE)")

    print()

    # Log results
    logger.log_history(history)

    summary = {
        'final_loss': history['loss'][-1],
        'final_reward': history['reward'][-1],
        'best_reward': history['best_reward'],
        'num_nonzero_params': history['num_nonzero_params'][-1],
        'best_params': history['best_params'].tolist(),
        'total_iterations': len(history['loss']),
    }

    if eval_interval > 0 and 'eval_reward' in history:
        eval_rewards = [r for r in history['eval_reward'] if r is not None]
        if eval_rewards:
            summary['best_eval_reward'] = max(eval_rewards)

    logger.log_results(summary, param_summary)

    # Plot trajectories and training curves
    if save_plot:
        plot_dir = logger.get_experiment_path()

        # Determine what to plot
        target_var_idx = env_config.get('target_var_idx', None)
        target_value = env_config.get('target_value', None)

        # For controller-based ODEs, create uncontrolled version for comparison
        if hasattr(ode, 'base_ode'):
            # Controlled ODE - compare with uncontrolled
            uncontrolled_ode = type(ode.base_ode)()
            plot_training_comparison(
                ode_initial=uncontrolled_ode,
                ode_final=ode,
                initial_state=initial_state,
                time_horizon=time_horizon,
                target_var_idx=target_var_idx,
                target_value=target_value,
                initial_state_range=initial_state_range,
                n_initial_states=5 if initial_state_range is not None else 1,
                filename=str(plot_dir / 'trajectories')
            )
        else:
            # Direct ODE - create initial version with default parameters for comparison
            # Get the initial parameters from config (first call to create_ode)
            if 'controller_order' in sig.parameters:
                ode_initial = create_ode(controller_order=controller_order, include_constant=include_constant)
            else:
                ode_initial = create_ode()

            plot_training_comparison(
                ode_initial=ode_initial,
                ode_final=ode,
                initial_state=initial_state,
                time_horizon=time_horizon,
                target_var_idx=target_var_idx,
                target_value=target_value,
                initial_state_range=initial_state_range,
                n_initial_states=5 if initial_state_range is not None else 1,
                filename=str(plot_dir / 'trajectories')
            )

        plot_training_curves(
            history=history,
            filename=str(plot_dir / 'training')
        )

    print("Note: Parameters have been restored to best (not final iteration)")
    print()

    # Stop logging and print location
    logger.stop_capture()
    print("="*60)
    print(f"Experiment logs saved to: {logger.get_experiment_path()}")
    print("="*60)


if __name__ == "__main__":
    fire.Fire(train)
