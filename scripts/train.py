"""Generic training script that works with any environment configuration."""
import fire
import torch
import json as json_module
from rpa_control.training import (
    load_config,
    extract_model_params,
    setup_experiment,
    create_ode_from_config,
    create_environment,
    create_training_config,
    run_training,
    run_testing,
    save_training_results,
    save_testing_results,
)
from rpa_control.style import set_style


def _create_training_ode_with_mismatch(
    env_config,
    true_ode,
    estimated_params_dict,
):
    """Create training ODE with model mismatch.

    Args:
        env_config: Environment configuration dict
        true_ode: True ODE instance (reality)
        estimated_params_dict: Dict of estimated parameter values

    Returns:
        tuple: (training_ode, training_model_params)
    """
    from rpa_control.controllers import ControlledODE

    if hasattr(true_ode, 'base_ode'):
        # Controller-based ODE: need to create new base ODE with estimated params
        base_ode = true_ode.base_ode

        # Get param names and create new tensor
        if hasattr(base_ode.__class__, 'fixed_param_names'):
            param_names_list = base_ode.__class__.fixed_param_names
            # Build new param tensor from estimated_params_dict
            new_params = []
            for pname in param_names_list:
                if pname in estimated_params_dict:
                    new_params.append(estimated_params_dict[pname])
                else:
                    # Use true value if not specified in estimate
                    idx = param_names_list.index(pname)
                    new_params.append(base_ode.fixed_params[idx].item())
            new_fixed_params = torch.tensor(new_params)

            # Create new base ODE with estimated parameters
            create_base_ode = env_config.get('create_base_ode')
            if create_base_ode is None:
                raise ValueError("Config must provide 'create_base_ode' for model mismatch")
            estimated_base_ode = create_base_ode()
            estimated_base_ode.fixed_params = new_fixed_params

            # Create controlled ODE with same controller setup but estimated base ODE
            training_ode = ControlledODE(
                base_ode=estimated_base_ode,
                controller=true_ode.controller,  # Share controller
                control_indices=true_ode.control_indices
            )
        else:
            raise ValueError("Cannot determine parameter names for training ODE")
    else:
        # Direct parameter ODE: replace fixed_params
        if hasattr(true_ode.__class__, 'fixed_param_names'):
            param_names_list = true_ode.__class__.fixed_param_names
            # Build new param tensor
            new_params = []
            for pname in param_names_list:
                if pname in estimated_params_dict:
                    new_params.append(estimated_params_dict[pname])
                else:
                    # Use true value if not specified
                    idx = param_names_list.index(pname)
                    new_params.append(true_ode.fixed_params[idx].item())
            new_fixed_params = torch.tensor(new_params)

            # Create new ODE with estimated parameters
            create_ode = env_config['create_ode']
            training_ode = create_ode(
                differentiable_params=true_ode.differentiable_params.detach().clone(),
                fixed_params=new_fixed_params
            )
        else:
            raise ValueError("Cannot determine parameter names for training ODE")

    training_model_params = extract_model_params(training_ode)
    return training_ode, training_model_params


def train(
    config: str,
    # Experiment naming
    name: str = None,
    # Training parameters
    n_iterations: int = None,
    learning_rate: float = None,
    l1_penalty: float = 0.0,
    l2_penalty: float = 0.0,
    time_horizon: float = None,
    log_interval: int = None,
    eval_interval: int = None,
    steady_state_fraction: float = None,
    # Reproducibility
    seed: int = None,
    # Scale-aware regularization
    scale_aware_regularization: bool = None,
    # Three-stage training
    use_three_stage: bool = False,
    n_iterations_stage1: int = 500,
    n_iterations_stage2: int = 300,
    l1_penalty_stage2: float = 0.01,
    n_iterations_stage3: int = 300,
    threshold_value_stage3: float = 1e-3,
    # TBPTT (Truncated Backpropagation Through Time)
    use_tbptt: bool = False,
    tbptt_truncation_steps: int = 5,
    # Perturbation settings (for robust training)
    perturb_params: bool = False,
    perturb_fold_change: float = 2.0,
    n_param_samples_eval: int = 10,
    # Controller-specific (for static/dynamic controllers)
    controller_order: int = None,
    include_constant: bool = None,
    # Initial condition settings
    use_single_ic: bool = None,
    # Estimated model parameters (for model mismatch: train on estimated, test on true)
    estimated_model_params: str = None,
):
    """
    Generic training script for any environment configuration.

    Args:
        config: Name of config file (e.g., 'ab_circuit', 'population') or path to config file
        name: Custom experiment name suffix for easy identification (e.g., 'high_l1', 'test_run')
        n_iterations: Number of training iterations (overrides default)
        learning_rate: Learning rate for optimizer (overrides default)
        l1_penalty: L1 regularization coefficient (for sparsity)
        l2_penalty: L2 regularization coefficient
        time_horizon: Simulation time horizon (overrides default)
        log_interval: Print progress every N iterations (overrides default)
        eval_interval: Evaluate on fixed ICs every N iterations (overrides default)
        steady_state_fraction: Fraction of trajectory to skip before computing reward (overrides default)
        seed: Random seed for reproducibility (overrides default)
        scale_aware_regularization: Use basis normalization (overrides default)
        use_three_stage: Use 3-stage training: normal → L1 reg → thresholding
        n_iterations_stage1: Iterations for stage 1
        n_iterations_stage2: Iterations for stage 2
        l1_penalty_stage2: L1 penalty for stage 2
        n_iterations_stage3: Iterations for stage 3
        threshold_value_stage3: Threshold for stage 3
        perturb_params: Whether to perturb parameters during training (for robust training)
        perturb_fold_change: Fold change for parameter perturbations during training
        n_param_samples_eval: Number of parameter samples per IC during evaluation
        controller_order: Polynomial order for controller (overrides default if applicable)
        include_constant: Include constant term in controller (overrides default if applicable)
        use_single_ic: If True, train from single initial state (ignores initial_state_range from config)
        estimated_model_params: JSON string of model parameters for training (e.g., '{"a": 0.6, "b": 0.03}')
                               If provided, controller is trained on this estimated/incorrect model
                               but tested on the true model (config defaults). This simulates model
                               mismatch where training uses an incorrect model estimate.
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
    if use_single_ic is None:
        use_single_ic = defaults.get('use_single_ic', False)

    # Setup experiment (seeds, logger)
    logger, seed = setup_experiment(
        env_config=env_config,
        experiment_name_suffix=name,
        seed=seed,
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

    # Create TRUE ODE (with config defaults - this represents reality)
    true_ode = create_ode_from_config(
        env_config=env_config,
        controller_order=controller_order,
        include_constant=include_constant,
    )

    has_controller = env_config.get('has_controller', True)
    if has_controller:
        print(f"Controller order: {controller_order}")
        print(f"Include constant: {include_constant}")

    print(f"ODE created: {type(true_ode).__name__}")

    # Extract true model parameters (reality)
    true_model_params = extract_model_params(true_ode)
    if 'model_params' in true_model_params:
        print(f"True model parameters (reality): {true_model_params['model_params']}")
    print()

    # Create TRAINING ODE (might be different if estimated_model_params provided)
    training_ode = true_ode  # Default: perfect model knowledge
    training_model_params = true_model_params  # Default: same as reality

    if estimated_model_params is not None:
        print("Creating training ODE with estimated model parameters (imperfect knowledge)...")
        # Parse estimated model params (fire may pass it as dict or string)
        if isinstance(estimated_model_params, str):
            estimated_params_dict = json_module.loads(estimated_model_params)
        else:
            estimated_params_dict = estimated_model_params
        print(f"Estimated model parameters (what we think): {estimated_params_dict}")

        training_ode, training_model_params = _create_training_ode_with_mismatch(
            env_config, true_ode, estimated_params_dict
        )
        print(f"Training with estimated model parameters: {training_model_params.get('model_params')}")
        print()
    else:
        print("Training with perfect model knowledge (training = reality)")
        print()

    # Use training_ode for environment and optimization
    ode = training_ode

    # Get initial state and evaluation states
    initial_state = env_config['initial_state']
    initial_state_range = env_config.get('initial_state_range', None)
    eval_initial_states = env_config.get('eval_initial_states', [initial_state])

    # Override initial_state_range and eval_initial_states if use_single_ic is True
    if use_single_ic:
        initial_state_range = None
        eval_initial_states = [initial_state]

    print(f"Initial state: {initial_state.tolist()}")
    if initial_state_range is not None:
        print(f"Initial state range: {initial_state_range}")
    else:
        print("Training from single initial condition")
    print(f"Evaluation: {len(eval_initial_states)} fixed initial conditions")
    print()

    # Get perturbation settings
    perturb_param_indices = env_config.get('perturb_param_indices', None) if perturb_params else None

    # Get state limits from config if available
    state_limits = defaults.get('state_limits', None)

    # Create environment
    env = create_environment(
        ode=ode,
        initial_state=initial_state,
        initial_state_range=initial_state_range,
        eval_initial_states=eval_initial_states,
        time_horizon=time_horizon,
        n_reward_steps=100,
        reward_fn=env_config['reward_fn'],
        state_limits=state_limits,
    )

    # Training configuration
    training_config = create_training_config(
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
        use_tbptt=use_tbptt,
        tbptt_truncation_steps=tbptt_truncation_steps,
    )

    # Log configuration
    config_dict = {
        'config_name': config,
        'environment': env_config['name'],
        'experiment_name': logger.experiment_name,
        'method': 'training',  # Distinguish from MPC experiments
        'n_iterations': n_iterations,
        'learning_rate': learning_rate,
        'l1_penalty': l1_penalty,
        'l2_penalty': l2_penalty,
        'time_horizon': time_horizon,
        'steady_state_fraction': steady_state_fraction,
        'seed': seed,
        'scale_aware_regularization': scale_aware_regularization,
        'use_three_stage': use_three_stage,
        'n_iterations_stage1': n_iterations_stage1,
        'n_iterations_stage2': n_iterations_stage2,
        'l1_penalty_stage2': l1_penalty_stage2,
        'n_iterations_stage3': n_iterations_stage3,
        'threshold_value_stage3': threshold_value_stage3,
        'use_tbptt': use_tbptt,
        'tbptt_truncation_steps': tbptt_truncation_steps,
        'perturb_params': perturb_params,
        'perturb_param_indices': perturb_param_indices,
        'perturb_fold_change': perturb_fold_change,
        'n_param_samples_eval': n_param_samples_eval,
        'eval_interval': eval_interval,
        'initial_state': initial_state.tolist(),
        'initial_state_range': initial_state_range,
        'eval_initial_states': [ic.tolist() for ic in eval_initial_states],
        'training_model_params': training_model_params.get('model_params'),
    }

    # Add controller-specific config
    if has_controller:
        config_dict['controller_order'] = controller_order
        config_dict['include_constant'] = include_constant

    # Add display parameters (for analysis script)
    config_dict['target_vars'] = env_config.get('target_vars', {})
    config_dict['state_var_names'] = env_config.get('state_var_names', None)
    config_dict['control_names'] = env_config.get('control_names', None)
    config_dict['param_names'] = env_config.get('param_names', None)

    logger.log_config(config_dict)

    # ========== TRAINING PHASE ==========
    history = run_training(
        env=env,
        ode=ode,
        config=training_config,
        perturb_params=perturb_params,
        perturb_param_indices=perturb_param_indices,
        perturb_fold_change=perturb_fold_change,
        n_param_samples_eval=n_param_samples_eval,
        eval_interval=eval_interval,
    )

    # Save training results
    save_training_results(
        logger=logger,
        history=history,
        ode=ode,
        env_config=env_config,
        eval_interval=eval_interval,
    )

    # ========== TESTING PHASE ==========
    trajectory_data = run_testing(
        test_ode=true_ode,
        initial_state=initial_state,
        time_horizon=time_horizon,
        n_reward_steps=env.n_reward_steps,
        target_vars=env_config.get('target_vars', {}),
        estimated_model_params=estimated_model_params,
        testing_model_params=true_model_params,
    )

    # Save testing results
    save_testing_results(
        logger=logger,
        trajectory_data=trajectory_data,
        training_model_params=training_model_params,
        testing_model_params=true_model_params,
        estimated_model_params_override=estimated_model_params,
    )

    print("Results saved. Use scripts/analyze.py to view detailed results and generate plots.")
    print()

    # Stop logging and print location
    logger.stop_capture()
    print("="*60)
    print(f"Experiment logs saved to: {logger.get_experiment_path()}")
    print("="*60)


if __name__ == "__main__":
    fire.Fire(train)
