"""Generic MPC script that works with any environment configuration."""
import fire
import torch
import matplotlib.pyplot as plt
import importlib.util
import time
from pathlib import Path
from rpa_control.mpc import MPCController, MPCConfig, simulate_mpc
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


def extract_model_params(ode):
    """Extract model parameters from an ODE for logging.

    Args:
        ode: ODE instance

    Returns:
        dict: Dictionary with parameter names and values
    """
    params_dict = {}

    if hasattr(ode, 'fixed_params') and ode.fixed_params is not None:
        param_values = ode.fixed_params.detach().tolist()
        # Try to get parameter names from class attribute
        if hasattr(ode.__class__, 'fixed_param_names'):
            param_names = ode.__class__.fixed_param_names
            params_dict['model_params'] = dict(zip(param_names, param_values))
        else:
            params_dict['model_params'] = param_values

    return params_dict


def run_mpc(
    config: str,
    # Experiment naming
    name: str = None,
    # Simulation settings
    initial_state: str = None,
    time_horizon: float = None,
    # MPC parameters
    prediction_horizon: int = None,
    dt: float = None,
    Q: str = None,
    Ru: float = None,
    R_deltau: float = None,
    u_min: float = None,
    u_max: float = None,
    cost_type: str = None,
    ftol: float = None,
    # Solver
    solver: str = None,
    warm_start: bool = None,
    # Estimated model parameters (for model mismatch: plan with estimated, simulate with true)
    estimated_model_params: str = None,
):
    """
    Run MPC on any environment configuration and save results to logs.

    Args:
        config: Name of config file (e.g., 'population', 'flight')
        name: Custom experiment name suffix for easy identification (e.g., 'high_Q', 'long_horizon')
        initial_state: Initial state as string (e.g., "[80, 25]", default: from config)
        time_horizon: Simulation time horizon (default: from config)
        prediction_horizon: MPC prediction horizon (default: from config)
        dt: MPC time step (default: from config)
        Q: State tracking weights as string (e.g., "[1.0, 1.0]", default: from config)
        Ru: Control magnitude weight (default: from config)
        R_deltau: Control rate-of-change weight (default: from config)
        u_min: Minimum control input (default: from config)
        u_max: Maximum control input (default: from config)
        cost_type: Cost function type ('quadratic' or 'l1', default: from config)
        ftol: Optimization tolerance (default: from config)
        solver: Optimization solver ('slsqp' or 'ipopt', default: from config or 'slsqp')
        warm_start: Use previous solution to initialize next optimization (default: True)
        estimated_model_params: JSON string of model parameters for planning (e.g., '{"a": 0.6, "b": 0.03}')
                               If provided, MPC uses this estimated/incorrect model for planning
                               but the true model (config defaults) for simulation. This simulates
                               model mismatch where MPC has imperfect model knowledge.
    """
    set_style()

    # Load environment configuration
    env_config = load_config(config)

    # Create experiment logger
    experiment_name = env_config.get('experiment_name', env_config['name']) + '_mpc'
    if name:
        experiment_name = f"{experiment_name}_{name}"
    logger = ExperimentLogger(
        log_dir="logs",
        experiment_name=experiment_name
    )

    # Start capturing console output
    logger.start_capture()

    print("="*60)
    print("MPC Control Simulation")
    print(f"Config: {config}")
    print(f"Log directory: {logger.get_experiment_path()}")
    print("="*60)
    print()

    # Get defaults
    defaults = env_config.get('defaults', {})
    mpc_defaults = env_config.get('mpc_defaults', {})

    # Parse initial state
    if initial_state is None:
        initial_state = env_config['initial_state']
    else:
        import json
        initial_state = torch.tensor(json.loads(initial_state), dtype=torch.float32)

    # Check if using custom MPC cost function or reference tracking
    mpc_cost_fn = env_config.get('mpc_stage_cost_fn', None)
    use_custom_cost = mpc_cost_fn is not None

    if use_custom_cost:
        print("Using custom MPC cost function from config")
        reference_state = None  # Not needed for custom cost
        target_vars = {}
    else:
        # Construct reference_state from target_vars (reference tracking mode)
        target_vars = env_config.get('target_vars', {})
        if not target_vars:
            raise ValueError("Config must specify 'target_vars' or 'mpc_stage_cost_fn' for MPC")

        reference_state = torch.zeros_like(initial_state)
        for idx, value in target_vars.items():
            reference_state[idx] = value

    # Get time horizon
    if time_horizon is None:
        time_horizon = defaults.get('time_horizon', 20.0)

    # MPC parameters with defaults
    if prediction_horizon is None:
        prediction_horizon = mpc_defaults.get('prediction_horizon', 10)
    if dt is None:
        dt = mpc_defaults.get('dt', 0.1)
    if Q is None:
        Q = mpc_defaults.get('Q', [1.0, 1.0])
    else:
        import json
        Q = json.loads(Q)
    if Ru is None:
        Ru = mpc_defaults.get('Ru', 0.5)
    if R_deltau is None:
        R_deltau = mpc_defaults.get('R_deltau', 0.5)
    if u_min is None:
        u_min = mpc_defaults.get('u_min', -20.0)
    if u_max is None:
        u_max = mpc_defaults.get('u_max', 20.0)
    if cost_type is None:
        cost_type = mpc_defaults.get('cost_type', 'quadratic')
    if ftol is None:
        ftol = mpc_defaults.get('ftol', 1e-3)

    if solver is None:
        solver = mpc_defaults.get('solver', 'slsqp')
    if warm_start is None:
        warm_start = mpc_defaults.get('warm_start', True)

    n_controls = mpc_defaults.get('n_controls', 1)
    integration_substeps = mpc_defaults.get('integration_substeps', 1)

    # Print environment description
    if 'description' in env_config:
        print(env_config['description'])
        print()

    # Create true ODE for simulation (always uses default parameters from config)
    create_base_ode = env_config.get('create_base_ode')
    if create_base_ode is None:
        raise ValueError("Config must provide 'create_base_ode' function for MPC")

    true_ode = create_base_ode()
    print(f"Simulation ODE created: {type(true_ode).__name__}")

    # Extract and log simulation model parameters (the true system)
    simulation_model_params = extract_model_params(true_ode)
    if 'model_params' in simulation_model_params:
        print(f"Simulation model parameters (true system): {simulation_model_params['model_params']}")
    print()

    # Create planning ODE for MPC predictions (might have wrong parameters)
    import json as json_module
    planning_ode = true_ode  # Default: assume perfect model knowledge
    planning_model_params = simulation_model_params  # Default: same as true system

    if estimated_model_params is not None:
        print("Creating planning ODE with estimated model parameters (imperfect knowledge)...")
        # Parse estimated model params (fire may pass it as dict or string)
        if isinstance(estimated_model_params, str):
            estimated_params_dict = json_module.loads(estimated_model_params)
        else:
            estimated_params_dict = estimated_model_params
        print(f"Estimated model parameters (what MPC thinks): {estimated_params_dict}")

        # Get param names and create new tensor
        if hasattr(true_ode.__class__, 'fixed_param_names'):
            param_names_list = true_ode.__class__.fixed_param_names
            # Build new param tensor from estimated_params_dict
            new_params = []
            for pname in param_names_list:
                if pname in estimated_params_dict:
                    new_params.append(estimated_params_dict[pname])
                else:
                    # Use true value if not specified
                    idx = param_names_list.index(pname)
                    new_params.append(true_ode.fixed_params[idx].item())
            new_fixed_params = torch.tensor(new_params)

            # Create new ODE with planning parameters (MPC's internal model)
            planning_ode = create_base_ode()
            planning_ode.fixed_params = new_fixed_params

            # Extract and log planning model parameters
            planning_model_params = extract_model_params(planning_ode)
            print(f"Planning model parameters (full): {planning_model_params.get('model_params')}")
        else:
            raise ValueError("Cannot determine parameter names for planning ODE")
        print()
    else:
        print("Using perfect model knowledge (planning = simulation)")
        print()

    # Display settings
    print(f"Initial state: {initial_state.tolist()}")
    if reference_state is not None:
        print(f"Reference state: {reference_state.tolist()}")
    print(f"Time horizon: {time_horizon}")
    print()

    print("MPC Configuration:")
    print(f"  Prediction horizon: {prediction_horizon} steps")
    print(f"  Time step: {dt}")
    if use_custom_cost:
        print(f"  Cost function: Custom (from config.mpc_stage_cost_fn)")
    else:
        print(f"  Q weights: {Q}")
        print(f"  Ru (control magnitude): {Ru}")
        print(f"  Cost type: {cost_type}")
    print(f"  R_deltau (rate-of-change): {R_deltau}")
    print(f"  Control bounds: [{u_min}, {u_max}]")
    print(f"  Solver: {solver}")
    print(f"  Warm start: {warm_start}")
    print()

    # Create MPC configuration
    mpc_config = MPCConfig(
        prediction_horizon=prediction_horizon,
        dt=dt,
        Q=torch.tensor(Q, dtype=torch.float32) if not use_custom_cost else None,
        Ru=Ru,
        R_deltau=R_deltau,
        u_min=u_min,
        u_max=u_max,
        ftol=ftol,
        cost_type=cost_type if not use_custom_cost else None,
        stage_cost_fn=mpc_cost_fn,  # Custom cost function (or None)
        solver=solver,
        warm_start=warm_start,
        integration_substeps=integration_substeps,
    )

    # Create MPC controller (uses planning ODE for predictions)
    print("Creating MPC controller with planning model...")
    mpc = MPCController(
        ode=planning_ode,
        config=mpc_config,
        reference_state=reference_state,
        n_controls=n_controls,
    )
    print("MPC controller created!")
    print()

    # Simulate MPC control (uses true ODE for actual simulation)
    print("Simulating MPC control on true system...")
    start_time = time.time()
    times, states, controls = simulate_mpc(true_ode, mpc, initial_state, time_horizon, dt)
    simulation_time = time.time() - start_time
    print(f"Simulation complete! ({len(times)} time steps)")
    print(f"Total MPC simulation time: {simulation_time:.2f} seconds")
    print(f"Average time per step: {simulation_time / len(times):.4f} seconds")
    print()

    # Print final state
    final_state = states[-1]
    print(f"Final state: {final_state.tolist()}")
    if reference_state is not None:
        print(f"Reference state: {reference_state.tolist()}")

        # Compute tracking error
        final_error = torch.norm(final_state - reference_state).item()
        print(f"Final tracking error: {final_error:.4f}")
    else:
        final_error = None
        print("(No reference state - using custom cost function)")
    print()

    # Control statistics
    control_min = controls.min().item()
    control_max = controls.max().item()
    control_mean = controls.mean().item()
    control_std = controls.std().item()

    print(f"Control statistics:")
    print(f"  Min: {control_min:.4f}")
    print(f"  Max: {control_max:.4f}")
    print(f"  Mean: {control_mean:.4f}")
    print(f"  Std: {control_std:.4f}")
    print()

    # Compute reward statistics (if reward_fn available)
    reward_fn = env_config.get('reward_fn', None)
    if reward_fn is not None:
        rewards = torch.tensor([reward_fn(s).item() if torch.is_tensor(reward_fn(s)) else reward_fn(s) for s in states])
        total_reward = rewards.sum().item()
        mean_reward = rewards.mean().item()
        final_reward = rewards[-1].item()
        print(f"Reward statistics:")
        print(f"  Total:  {total_reward:.4f}")
        print(f"  Mean:   {mean_reward:.4f}")
        print(f"  Final:  {final_reward:.4f}")
        print()
    else:
        total_reward = None
        mean_reward = None
        final_reward = None

    # Compute tracking metrics (only if using reference tracking)
    if reference_state is not None:
        tracking_errors = torch.norm(states - reference_state, dim=1)
        mean_tracking_error = tracking_errors.mean().item()
        max_tracking_error = tracking_errors.max().item()
    else:
        mean_tracking_error = None
        max_tracking_error = None

    # Total control effort
    total_control_effort = (controls ** 2).sum().item()

    # Log configuration
    config_dict = {
        'config_name': config,
        'environment': env_config['name'],
        'experiment_name': experiment_name,
        'method': 'mpc',
        'use_custom_cost': use_custom_cost,
        'initial_state': initial_state.tolist(),
        'reference_state': reference_state.tolist() if reference_state is not None else None,
        'time_horizon': time_horizon,
        'prediction_horizon': prediction_horizon,
        'dt': dt,
        'Q': Q if not use_custom_cost else None,
        'Ru': Ru,
        'R_deltau': R_deltau,
        'u_min': u_min,
        'u_max': u_max,
        'cost_type': cost_type if not use_custom_cost else 'custom',
        'ftol': ftol,
        'solver': solver,
        'warm_start': warm_start,
        'n_controls': n_controls,
        # Model parameters
        'planning_model_params': planning_model_params.get('model_params'),
        'simulation_model_params': simulation_model_params.get('model_params'),
        'estimated_model_params_override': estimated_model_params,  # Store the override if provided
        # Display parameters
        'target_vars': target_vars,
        'state_var_names': env_config.get('state_var_names', None),
        'control_names': env_config.get('control_names', None),
    }

    logger.log_config(config_dict)

    # Log MPC trajectory data (full precision)
    mpc_data = {
        'times': times.tolist(),
        'states': states.tolist(),
        'controls': controls.tolist(),
        'reference_state': reference_state.tolist() if reference_state is not None else None,
    }

    import json
    mpc_data_path = logger.get_experiment_path() / 'mpc_trajectory.json'
    with open(mpc_data_path, 'w') as f:
        json.dump(mpc_data, f, indent=2)

    print(f"MPC trajectory saved to: {mpc_data_path}")

    # Log results summary
    summary = {
        'final_state': final_state.tolist(),
        'final_tracking_error': final_error,
        'mean_tracking_error': mean_tracking_error,
        'max_tracking_error': max_tracking_error,
        'total_reward': total_reward,
        'mean_reward': mean_reward,
        'final_reward': final_reward,
        'control_min': control_min,
        'control_max': control_max,
        'control_mean': control_mean,
        'control_std': control_std,
        'total_control_effort': total_control_effort,
        'n_timesteps': len(times),
        'simulation_time_seconds': simulation_time,
        'avg_time_per_step_seconds': simulation_time / len(times),
    }

    logger.log_results(summary, controller_summary="")

    print()

    # Stop logging and print location
    logger.stop_capture()
    print("="*60)
    print(f"Experiment logs saved to: {logger.get_experiment_path()}")
    print("="*60)
    print()

    print("Results saved. Use scripts/analyze.py to view detailed results and generate plots.")
    print()


if __name__ == "__main__":
    fire.Fire(run_mpc)
