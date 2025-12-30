"""Analyze and visualize training results from saved experiment data."""
import fire
import json
import torch
import numpy as np
import importlib.util
import inspect
from pathlib import Path
from rpa_control.utils.plotting import plot_training_comparison, plot_training_curves
from rpa_control.style import set_style


def load_experiment_data(experiment_dir: str):
    """Load experiment data from directory.

    Args:
        experiment_dir: Path to experiment directory

    Returns:
        config: Experiment configuration
        data: Training history or MPC trajectory data
    """
    exp_path = Path(experiment_dir)

    if not exp_path.exists():
        raise FileNotFoundError(f"Experiment directory not found: {experiment_dir}")

    # Load config
    config_path = exp_path / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r') as f:
        config = json.load(f)

    # Check if this is MPC or training data
    method = config.get('method', 'training')

    if method == 'mpc':
        # Load MPC trajectory
        mpc_path = exp_path / "mpc_trajectory.json"
        if not mpc_path.exists():
            raise FileNotFoundError(f"MPC trajectory file not found: {mpc_path}")

        with open(mpc_path, 'r') as f:
            data = json.load(f)
    else:
        # Load training history
        history_path = exp_path / "history.json"
        if not history_path.exists():
            raise FileNotFoundError(f"History file not found: {history_path}")

        with open(history_path, 'r') as f:
            data = json.load(f)

    return config, data


def load_config_module(config_name: str):
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
        # Look in configs directory relative to this script
        config_path = Path(__file__).parent.parent / 'configs' / f'{config_name}.py'

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    # Load the config module
    spec = importlib.util.spec_from_file_location("config", config_path)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)

    return config_module.ENV_CONFIG


def analyze(
    experiment_dir: str,
    plot_training: bool = True,
    plot_trajectories: bool = True,
):
    """Analyze and visualize results (training or MPC).

    Args:
        experiment_dir: Path to experiment directory (e.g., logs/population_static_20251230_142706)
        plot_training: Whether to plot training curves (for training) or trajectory (for MPC)
        plot_trajectories: Whether to plot trajectory comparisons (for training only)
    """
    set_style()

    print("="*60)
    print("Analyzing experiment results")
    print(f"Directory: {experiment_dir}")
    print("="*60)
    print()

    # Load data
    config, data = load_experiment_data(experiment_dir)
    exp_path = Path(experiment_dir)

    # Check experiment type
    method = config.get('method', 'training')

    if method == 'mpc':
        # Analyze MPC results
        analyze_mpc(config, data, exp_path, plot_trajectory=plot_training)
    else:
        # Analyze training results
        analyze_training(config, data, exp_path, plot_training, plot_trajectories)


def analyze_training(config, history, exp_path, plot_training, plot_trajectories):
    """Analyze training experiment results."""
    # Display configuration
    print("Experiment Configuration:")
    print(f"  Name: {config.get('experiment_name', 'unknown')}")
    print(f"  Environment: {config.get('environment', 'unknown')}")
    print(f"  Config file: {config.get('config_name', 'unknown')}")
    print(f"  Iterations: {config.get('n_iterations', 'unknown')}")
    print(f"  Learning rate: {config.get('learning_rate', 'unknown')}")
    print(f"  Time horizon: {config.get('time_horizon', 'unknown')}")
    if config.get('l1_penalty', 0.0) > 0:
        print(f"  L1 penalty: {config.get('l1_penalty')}")
    if config.get('l2_penalty', 0.0) > 0:
        print(f"  L2 penalty: {config.get('l2_penalty')}")
    if config.get('use_three_stage', False):
        print(f"  Three-stage training: enabled")
        print(f"    Stage 1: {config.get('n_iterations_stage1')} iterations")
        print(f"    Stage 2: {config.get('n_iterations_stage2')} iterations (L1={config.get('l1_penalty_stage2')})")
        print(f"    Stage 3: {config.get('n_iterations_stage3')} iterations (threshold={config.get('threshold_value_stage3')})")
    if config.get('controller_order') is not None:
        print(f"  Controller order: {config.get('controller_order')}")
        print(f"  Include constant: {config.get('include_constant')}")
    print(f"  Seed: {config.get('seed', 'unknown')}")
    print()

    # Display training results
    print("Training Results:")
    print(f"  Final loss: {history['loss'][-1]:.6f}")
    print(f"  Final reward: {history['reward'][-1]:.6f}")
    print(f"  Best reward (single iter): {history.get('best_reward', 'N/A'):.6f}")

    # Check for eval rewards
    if 'eval_reward' in history:
        eval_rewards = [r for r in history.get('eval_reward', [])
                       if r is not None and not (isinstance(r, float) and (r != r))]  # filter None and NaN
        if eval_rewards:
            print(f"  Best eval reward: {max(eval_rewards):.6f}")

    print(f"  Non-zero params: {history['num_nonzero_params'][-1]}")
    print()

    # Display best parameters
    if 'best_params' in history:
        best_params = history['best_params']
        # Flatten nested lists if needed
        if isinstance(best_params, list):
            # Check if it's a nested list
            if best_params and isinstance(best_params[0], list):
                best_params_flat = [item for sublist in best_params for item in sublist]
            else:
                best_params_flat = best_params
        else:
            best_params_flat = best_params.tolist()
        best_params_str = "[" + ", ".join([f"{p:.6f}" for p in best_params_flat]) + "]"
        print(f"Best params: {best_params_str}")
        print()

    # Display controller/parameter summary from results.txt
    results_path = exp_path / "results.txt"
    if results_path.exists():
        with open(results_path, 'r') as f:
            content = f.read()

        # Extract and display the controller/parameters section
        if "Trained Controller:" in content or "Trained Parameters:" in content:
            lines = content.split('\n')
            in_section = False
            section_lines = []

            for line in lines:
                if 'Trained Controller:' in line or 'Trained Parameters:' in line:
                    in_section = True
                    print(line)
                    continue

                if in_section:
                    if line.startswith('-'*60):
                        print(line)
                        continue
                    if line.strip() and not line.startswith('='):
                        print(line)
                    elif not line.strip():
                        break
            print()

    # Plot training curves
    if plot_training:
        print("Generating training curves...")
        plot_training_curves(
            history=history,
            filename=str(exp_path / 'training')
        )
        print(f"  Saved to: {exp_path / 'training.pdf'}")
        print()

    # Plot trajectories
    if plot_trajectories:
        print("Generating trajectory plots...")

        try:
            # Load the environment config to get create_ode function
            config_name = config.get('config_name')
            if not config_name:
                print("  Error: config_name not found in saved config")
            else:
                env_config = load_config_module(config_name)
                create_ode = env_config['create_ode']

                # Check if create_ode accepts controller_order
                sig = inspect.signature(create_ode)

                # Create initial ODE (for comparison)
                if 'controller_order' in sig.parameters:
                    controller_order = config.get('controller_order', 2)
                    include_constant = config.get('include_constant', True)
                    ode_initial = create_ode(controller_order=controller_order, include_constant=include_constant)
                    ode_final = create_ode(controller_order=controller_order, include_constant=include_constant)
                else:
                    ode_initial = create_ode()
                    ode_final = create_ode()

                # Load best parameters into final ODE
                best_params = torch.tensor(history['best_params'])
                ode_final.differentiable_params.data.copy_(best_params)

                # Sync parameters to controller if needed
                if hasattr(ode_final, 'update_controller_params'):
                    ode_final.update_controller_params()

                # Get initial state and plotting parameters
                initial_state = torch.tensor(config.get('initial_state'))
                initial_state_range = config.get('initial_state_range', None)
                time_horizon = config.get('time_horizon', 20.0)
                target_var_idx = config.get('target_var_idx', None)
                target_value = config.get('target_value', None)

                # For controller-based ODEs, create uncontrolled version for comparison
                if hasattr(ode_final, 'base_ode'):
                    # Controlled ODE - compare with uncontrolled
                    uncontrolled_ode = type(ode_final.base_ode)()
                    plot_training_comparison(
                        ode_initial=uncontrolled_ode,
                        ode_final=ode_final,
                        initial_state=initial_state,
                        time_horizon=time_horizon,
                        target_var_idx=target_var_idx,
                        target_value=target_value,
                        initial_state_range=initial_state_range,
                        n_initial_states=5 if initial_state_range is not None else 1,
                        filename=str(exp_path / 'trajectories')
                    )
                else:
                    # Direct ODE - compare initial vs final parameters
                    plot_training_comparison(
                        ode_initial=ode_initial,
                        ode_final=ode_final,
                        initial_state=initial_state,
                        time_horizon=time_horizon,
                        target_var_idx=target_var_idx,
                        target_value=target_value,
                        initial_state_range=initial_state_range,
                        n_initial_states=5 if initial_state_range is not None else 1,
                        filename=str(exp_path / 'trajectories')
                    )

                print(f"  Saved to: {exp_path / 'trajectories.pdf'}")
                print()

        except Exception as e:
            print(f"  Error generating trajectory plots: {e}")
            import traceback
            traceback.print_exc()
            print()

    print("="*60)
    print(f"Experiment logs: {exp_path}")
    print("="*60)


def analyze_mpc(config, mpc_data, exp_path, plot_trajectory=True):
    """Analyze MPC experiment results."""
    import matplotlib.pyplot as plt

    # Display configuration
    print("Experiment Configuration:")
    print(f"  Name: {config.get('experiment_name', 'unknown')}")
    print(f"  Environment: {config.get('environment', 'unknown')}")
    print(f"  Config file: {config.get('config_name', 'unknown')}")
    print(f"  Method: MPC")
    print(f"  Initial state: {config.get('initial_state')}")
    print(f"  Reference state: {config.get('reference_state')}")
    print(f"  Time horizon: {config.get('time_horizon')}")
    print(f"  Prediction horizon: {config.get('prediction_horizon')} steps")
    print(f"  MPC time step: {config.get('dt')}")
    print(f"  Q weights: {config.get('Q')}")
    print(f"  Ru (control magnitude): {config.get('Ru')}")
    print(f"  R_deltau (rate-of-change): {config.get('R_deltau')}")
    print(f"  Control bounds: [{config.get('u_min')}, {config.get('u_max')}]")
    print(f"  Cost type: {config.get('cost_type')}")
    print()

    # Extract data
    times = np.array(mpc_data['times'])
    states = np.array(mpc_data['states'])
    controls = np.array(mpc_data['controls'])
    reference_state = np.array(mpc_data['reference_state'])

    # Compute metrics
    final_state = states[-1]
    tracking_errors = np.linalg.norm(states - reference_state, axis=1)
    final_error = tracking_errors[-1]
    mean_error = tracking_errors.mean()
    max_error = tracking_errors.max()

    # Display results
    print("MPC Results:")
    print(f"  Final state: {final_state.tolist()}")
    print(f"  Reference state: {reference_state.tolist()}")
    print(f"  Final tracking error: {final_error:.6f}")
    print(f"  Mean tracking error: {mean_error:.6f}")
    print(f"  Max tracking error: {max_error:.6f}")
    print()

    print(f"Control statistics:")
    print(f"  Min: {controls.min():.6f}")
    print(f"  Max: {controls.max():.6f}")
    print(f"  Mean: {controls.mean():.6f}")
    print(f"  Std: {controls.std():.6f}")
    print(f"  Total effort: {(controls**2).sum():.6f}")
    print()

    # Plot trajectory
    if plot_trajectory:
        print("Generating MPC trajectory plots...")

        # Get variable names from config
        config_name = config.get('config_name')
        if config_name:
            try:
                env_config = load_config_module(config_name)
                state_var_names = env_config.get('state_var_names', [f'x{i}' for i in range(states.shape[1])])
                control_names = env_config.get('control_names', ['u'])
            except:
                state_var_names = [f'x{i}' for i in range(states.shape[1])]
                control_names = ['u']
        else:
            state_var_names = [f'x{i}' for i in range(states.shape[1])]
            control_names = ['u']

        n_states = states.shape[1]
        n_controls = controls.shape[1] if len(controls.shape) > 1 else 1

        fig, axes = plt.subplots(n_states + 1, 1, figsize=(10, 3 * (n_states + 1)))
        if n_states == 1:
            axes = [axes[0], axes[1]]

        # Plot states
        for i in range(n_states):
            ax = axes[i]
            ax.plot(times, states[:, i], 'b-', linewidth=2, label=f'{state_var_names[i]} (actual)')
            ax.axhline(reference_state[i], color='r', linestyle='--', alpha=0.5, label='target')
            ax.set_xlabel('Time')
            ax.set_ylabel(state_var_names[i])
            ax.set_title(f'{state_var_names[i]} Evolution with MPC')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)

        # Plot control
        ax = axes[-1]
        control_times = times[:-1]  # Control is one step shorter
        if n_controls == 1:
            ax.plot(control_times, controls, 'g-', linewidth=2, label=control_names[0])
        else:
            for i in range(n_controls):
                ax.plot(control_times, controls[:, i], linewidth=2,
                       label=control_names[i] if i < len(control_names) else f'u{i}')
        ax.axhline(0.0, color='k', linestyle='--', alpha=0.3)
        ax.set_xlabel('Time')
        ax.set_ylabel('Control')
        ax.set_title('MPC Control Input')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

        plt.tight_layout()

        filename = exp_path / 'mpc_trajectory.pdf'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"  Saved to: {filename}")
        print()

    print("="*60)
    print(f"Experiment logs: {exp_path}")
    print("="*60)


def plot_controller(
    config: str,
    controller_params: str = "zero",
    controller_order: int = None,
    include_constant: bool = None,
    initial_state: str = None,
    initial_state_range: str = None,
    time_horizon: float = None,
    n_initial_states: int = 5,
    output: str = None,
):
    """Plot trajectories for a specific controller configuration.

    Args:
        config: Name of config file (e.g., 'population')
        controller_params: Controller parameters as JSON array string (e.g., "[0.5, -0.3, 0.1]"),
                          "zero" for all zeros, or "constant:VALUE" for constant control
        controller_order: Controller polynomial order (default: from config)
        include_constant: Include constant term (default: from config)
        initial_state: Initial state as JSON array (default: from config)
        initial_state_range: Initial state range as JSON array of tuples (default: from config)
        time_horizon: Simulation time (default: from config)
        n_initial_states: Number of initial states to plot if range provided
        output: Output filename (default: trajectories_{config}.pdf)
    """
    set_style()

    print("="*60)
    print("Plotting controller trajectories")
    print(f"Config: {config}")
    print("="*60)
    print()

    # Load environment config
    env_config = load_config_module(config)
    create_ode = env_config['create_ode']

    # Get defaults from config
    defaults = env_config.get('defaults', {})
    if controller_order is None:
        controller_order = defaults.get('controller_order', 2)
    if include_constant is None:
        include_constant = defaults.get('include_constant', True)
    if time_horizon is None:
        time_horizon = defaults.get('time_horizon', 20.0)

    # Parse initial state
    if initial_state is None:
        initial_state = env_config['initial_state']
    else:
        initial_state = torch.tensor(json.loads(initial_state))

    # Parse initial state range
    if initial_state_range is None:
        initial_state_range = env_config.get('initial_state_range', None)
    else:
        initial_state_range = json.loads(initial_state_range)

    # Create ODE
    sig = inspect.signature(create_ode)
    if 'controller_order' in sig.parameters:
        ode = create_ode(controller_order=controller_order, include_constant=include_constant)
        print(f"Controller order: {controller_order}")
        print(f"Include constant: {include_constant}")
    else:
        ode = create_ode()

    # Set controller parameters
    if controller_params == "zero":
        # Zero controller
        params = torch.zeros_like(ode.differentiable_params)
        print("Using zero controller (all parameters = 0)")
    elif isinstance(controller_params, str) and controller_params.startswith("constant:"):
        # Constant controller: only set the constant term (first parameter)
        value = float(controller_params.split(":")[1])
        params = torch.zeros_like(ode.differentiable_params)
        # Flatten params to ensure we're setting a single element
        params_flat = params.flatten()
        params_flat[0] = value
        params = params_flat.reshape(ode.differentiable_params.shape)
        print(f"Using constant controller (u = {value})")
    else:
        # Parse JSON array (Fire may have already parsed it to a list)
        if isinstance(controller_params, str):
            params_list = json.loads(controller_params)
        else:
            params_list = controller_params
        params = torch.tensor(params_list, dtype=torch.float32)
        # Reshape if needed to match ODE parameter shape
        if params.shape != ode.differentiable_params.shape:
            params = params.reshape(ode.differentiable_params.shape)
        print(f"Using custom controller parameters: {params.flatten().tolist()}")

    # Set parameters
    ode.differentiable_params.data.copy_(params)
    if hasattr(ode, 'update_controller_params'):
        ode.update_controller_params()

    # Display controller summary
    if hasattr(ode, 'get_controller_summary'):
        state_var_names = env_config.get('state_var_names', None)
        control_names = env_config.get('control_names', ['u'])
        print()
        print("Controller:")
        print("-"*60)
        print(ode.get_controller_summary(state_var_names, control_names))

    # Get plotting parameters
    target_var_idx = env_config.get('target_var_idx', None)
    target_value = env_config.get('target_value', None)

    # Determine output filename
    if output is None:
        output = f"trajectories_{config}.pdf"

    print()
    print(f"Plotting trajectories...")
    print(f"  Initial state: {initial_state.tolist()}")
    if initial_state_range is not None:
        print(f"  Initial state range: {initial_state_range}")
        print(f"  Number of initial states: {n_initial_states}")
    print(f"  Time horizon: {time_horizon}")
    print()

    # For controlled ODEs, compare with uncontrolled version
    if hasattr(ode, 'base_ode'):
        uncontrolled_ode = type(ode.base_ode)()
        plot_training_comparison(
            ode_initial=uncontrolled_ode,
            ode_final=ode,
            initial_state=initial_state,
            time_horizon=time_horizon,
            target_var_idx=target_var_idx,
            target_value=target_value,
            initial_state_range=initial_state_range,
            n_initial_states=n_initial_states,
            filename=output.replace('.pdf', '')
        )
    else:
        # For non-controlled ODEs, just plot the single trajectory
        # Create a dummy "initial" ODE that's the same
        plot_training_comparison(
            ode_initial=ode,
            ode_final=ode,
            initial_state=initial_state,
            time_horizon=time_horizon,
            target_var_idx=target_var_idx,
            target_value=target_value,
            initial_state_range=initial_state_range,
            n_initial_states=n_initial_states,
            filename=output.replace('.pdf', '')
        )

    print(f"Saved to: {output}")
    print("="*60)


if __name__ == "__main__":
    fire.Fire({
        'analyze': analyze,
        'plot': plot_controller,
    })
