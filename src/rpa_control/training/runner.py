"""Training and testing execution logic."""
import torch
from torchdiffeq import odeint
from rpa_control.optimization.gradient import train_ode_parameters


def run_training(
    env,
    ode,
    config,
    perturb_params=False,
    perturb_param_indices=None,
    perturb_fold_change=2.0,
    n_param_samples_eval=10,
    eval_interval=0,
    logger=None,
    full_config=None,
):
    """Execute training phase.

    Args:
        env: DifferentiableEnv instance
        ode: ODE instance to train
        config: TrainingConfig instance
        perturb_params: Whether to perturb parameters during training
        perturb_param_indices: Indices of parameters to perturb
        perturb_fold_change: Fold change for parameter perturbations
        n_param_samples_eval: Number of parameter samples per IC during evaluation
        eval_interval: Evaluation interval
        logger: ExperimentLogger for saving plots
        full_config: Full configuration dict for trajectory plotting

    Returns:
        dict: Training history
    """
    print("="*60)
    print("TRAINING PHASE")
    print("="*60)
    print()
    print("Starting training...")
    print(f"Time horizon: {env.time_horizon}")
    print(f"Learning rate: {config.learning_rate}")
    print(f"Iterations: {config.n_iterations}")
    if perturb_params:
        print(f"Parameter perturbations: enabled (fold change: {perturb_fold_change})")
        print(f"  Perturbing indices: {perturb_param_indices}")
        print(f"  Eval samples per IC: {n_param_samples_eval}")
    if eval_interval > 0:
        print(f"Evaluation interval: {eval_interval}")
    print()

    # Get output directory from logger
    output_dir = logger.get_experiment_path() if logger is not None else None

    # Train
    history = train_ode_parameters(
        env=env,
        ode=ode,
        config=config,
        output_dir=output_dir,
        full_config=full_config,
    )

    print()
    print("="*60)
    print("Training complete!")
    print("="*60)
    print()

    return history


def run_testing(
    test_ode,
    initial_state,
    time_horizon,
    n_reward_steps,
    target_vars=None,
    estimated_model_params=None,
    testing_model_params=None,
):
    """Execute testing phase by simulating trajectory with trained controller.

    Args:
        test_ode: ODE instance to test (usually the true/reality model)
        initial_state: Initial state tensor
        time_horizon: Simulation time horizon
        n_reward_steps: Number of reward computation steps
        target_vars: Dict mapping state indices to target values
        estimated_model_params: Optional dict of estimated parameters (if model mismatch)
        testing_model_params: Dict of testing model parameters

    Returns:
        dict: Trajectory data with keys 'times', 'states', 'controls', 'reference_state'
    """
    print()
    print("="*60)
    print("TESTING PHASE")
    print("="*60)
    print()

    if estimated_model_params is not None:
        print("Testing on true model (reality) - different from training model")
        print(f"Testing model parameters: {testing_model_params.get('model_params')}")
        print()
    else:
        print("Testing with same model as training (no model mismatch)")
        print()

    # Simulate and save trajectory with best controller on test ODE
    print("Simulating trajectory with best controller on test model...")

    # Simulate full trajectory
    t_span = torch.linspace(0, time_horizon, n_reward_steps + 1)

    # Augment initial state with controller states if needed
    if hasattr(test_ode, 'controller') and hasattr(test_ode.controller, 'n_controller_states'):
        n_controller_states = test_ode.controller.n_controller_states
        controller_init = torch.zeros(n_controller_states)
        initial_state_full = torch.cat([initial_state, controller_init])
    else:
        initial_state_full = initial_state

    # Integrate using test ODE
    with torch.no_grad():
        states_full = odeint(test_ode, initial_state_full, t_span, method='dopri5')

    # Extract base states (without controller states)
    n_base = len(initial_state)
    states_base = states_full[:, :n_base]

    # Compute control inputs at each time step
    controls = []
    with torch.no_grad():
        for state in states_full[:-1]:  # Exclude last time point
            if hasattr(test_ode, 'controller'):
                u = test_ode.controller(state[:n_base])
            else:
                u = torch.zeros(1)  # No controller
            controls.append(u)
    controls = torch.stack(controls)

    # Construct reference_state from target_vars
    # Only variables in target_vars contribute to tracking error; others are zero
    if target_vars is None:
        target_vars = {}
    reference_state = torch.zeros_like(initial_state)
    for idx, value in target_vars.items():
        reference_state[idx] = value

    trajectory_data = {
        'times': t_span.tolist(),
        'states': states_base.tolist(),
        'controls': controls.tolist(),
        'reference_state': reference_state.tolist(),
    }

    print("Trajectory simulation complete.")
    print()

    return trajectory_data
