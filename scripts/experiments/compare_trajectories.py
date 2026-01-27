"""Generic trajectory comparison script for any two experiments."""
import fire
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from rpa_control.style import set_style
import importlib.util


def load_config_module(config_name: str):
    """Load environment configuration from configs directory."""
    if config_name.endswith('.py'):
        config_path = Path(config_name)
    else:
        config_path = Path(__file__).parent.parent / 'configs' / f'{config_name}.py'

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    spec = importlib.util.spec_from_file_location("config", config_path)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)

    return config_module.ENV_CONFIG


def load_experiment_trajectory(experiment_dir: str):
    """Load trajectory data from experiment directory.

    Args:
        experiment_dir: Path to experiment directory

    Returns:
        config: Experiment configuration
        trajectory: Trajectory data (times, states, controls, reference_state)
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

    # Check if this is MPC or training
    method = config.get('method', 'training')

    if method == 'mpc':
        traj_path = exp_path / "mpc_trajectory.json"
    else:
        traj_path = exp_path / "trajectory.json"

    if not traj_path.exists():
        raise FileNotFoundError(f"Trajectory file not found: {traj_path}")

    with open(traj_path, 'r') as f:
        trajectory = json.load(f)

    return config, trajectory


def compare_trajectories(
    experiment1: str,
    experiment2: str,
    label1: str = None,
    label2: str = None,
    output: str = None,
    Q_weights: str = None,
    Ru: float = None,
):
    """Compare trajectories from two experiments.

    Args:
        experiment1: Path to first experiment directory
        experiment2: Path to second experiment directory
        label1: Label for first experiment (default: auto-detect from config)
        label2: Label for second experiment (default: auto-detect from config)
        output: Output filename (default: comparison.pdf)
        Q_weights: State tracking weights as JSON array (e.g., "[1.0, 1.0]", default: from config)
        Ru: Control magnitude weight (default: from config)
    """
    set_style()

    print("="*60)
    print("Comparing Trajectories")
    print("="*60)
    print()

    # Load both experiments
    print(f"Loading experiment 1: {experiment1}")
    config1, traj1 = load_experiment_trajectory(experiment1)
    method1 = config1.get('method', 'training')

    print(f"Loading experiment 2: {experiment2}")
    config2, traj2 = load_experiment_trajectory(experiment2)
    method2 = config2.get('method', 'training')
    print()

    # Auto-generate labels if not provided
    if label1 is None:
        label1 = f"{method1.upper()}" if method1 == 'mpc' else "Trained"
    if label2 is None:
        label2 = f"{method2.upper()}" if method2 == 'mpc' else "Trained"

    # Convert to numpy arrays
    times1 = np.array(traj1['times'])
    states1 = np.array(traj1['states'])
    controls1 = np.array(traj1['controls'])
    ref_state1 = np.array(traj1['reference_state'])

    times2 = np.array(traj2['times'])
    states2 = np.array(traj2['states'])
    controls2 = np.array(traj2['controls'])
    ref_state2 = np.array(traj2['reference_state'])

    # Verify reference states match
    if not np.allclose(ref_state1, ref_state2, atol=1e-6):
        print(f"WARNING: Reference states differ!")
        print(f"  {label1}: {ref_state1}")
        print(f"  {label2}: {ref_state2}")
        print()

    reference_state = ref_state1  # Use first one

    # Get variable names from config
    config_name = config1.get('config_name') or config2.get('config_name')
    if config_name:
        try:
            env_config = load_config_module(config_name)
            state_var_names = env_config.get('state_var_names', [f'x{i}' for i in range(states1.shape[1])])
            control_names = env_config.get('control_names', ['u'])
        except:
            state_var_names = [f'x{i}' for i in range(states1.shape[1])]
            control_names = ['u']
    else:
        state_var_names = [f'x{i}' for i in range(states1.shape[1])]
        control_names = ['u']

    # Get cost parameters (prefer MPC config, then user override, then defaults)
    if Q_weights is None:
        # Try to get from MPC experiment
        if method1 == 'mpc':
            Q_weights = np.array(config1.get('Q', [1.0] * states1.shape[1]))
        elif method2 == 'mpc':
            Q_weights = np.array(config2.get('Q', [1.0] * states2.shape[1]))
        else:
            Q_weights = np.ones(states1.shape[1])
    else:
        Q_weights = np.array(json.loads(Q_weights))

    if Ru is None:
        # Try to get from MPC experiment
        if method1 == 'mpc':
            Ru = config1.get('Ru', 0.5)
        elif method2 == 'mpc':
            Ru = config2.get('Ru', 0.5)
        else:
            Ru = 0.5

    # Compute metrics
    tracking_error1 = np.linalg.norm(states1 - reference_state, axis=1)
    tracking_error2 = np.linalg.norm(states2 - reference_state, axis=1)

    # Compute time steps (assuming uniform spacing)
    dt1 = np.mean(np.diff(times1))
    dt2 = np.mean(np.diff(times2))

    # Compute cumulative control effort
    control_effort1 = controls1 ** 2
    control_effort2 = controls2 ** 2

    # Pad with zero at start for alignment with times
    cumulative_control1 = np.concatenate([[0], np.cumsum(control_effort1.squeeze()) * dt1])
    cumulative_control2 = np.concatenate([[0], np.cumsum(control_effort2.squeeze()) * dt2])

    # Compute cumulative total cost (state tracking + control)
    state_cost1 = np.sum((states1 - reference_state) ** 2 * Q_weights, axis=1)
    state_cost2 = np.sum((states2 - reference_state) ** 2 * Q_weights, axis=1)

    control_cost1 = np.concatenate([[0], Ru * control_effort1.squeeze()])
    control_cost2 = np.concatenate([[0], Ru * control_effort2.squeeze()])

    total_cost1 = state_cost1 + control_cost1
    total_cost2 = state_cost2 + control_cost2

    cumulative_total1 = np.cumsum(total_cost1) * dt1
    cumulative_total2 = np.cumsum(total_cost2) * dt2

    # Print comparison metrics
    print("Comparison Metrics:")
    print("-"*60)
    print(f"{label1}:")
    print(f"  Final tracking error: {tracking_error1[-1]:.6f}")
    print(f"  Mean tracking error: {tracking_error1.mean():.6f}")
    print(f"  Total control effort: {cumulative_control1[-1]:.6f}")
    print(f"  Total cost: {cumulative_total1[-1]:.6f}")
    print()
    print(f"{label2}:")
    print(f"  Final tracking error: {tracking_error2[-1]:.6f}")
    print(f"  Mean tracking error: {tracking_error2.mean():.6f}")
    print(f"  Total control effort: {cumulative_control2[-1]:.6f}")
    print(f"  Total cost: {cumulative_total2[-1]:.6f}")
    print()

    # Create comparison plots
    print("Generating comparison plots...")

    n_states = states1.shape[1]
    n_plots = n_states + 4  # states + control + tracking error + cumulative control + cumulative total cost

    # Determine grid layout
    if n_states == 1:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
    elif n_states == 2:
        fig, axes = plt.subplots(3, 2, figsize=(12, 12))
        axes = axes.flatten()
    else:
        n_cols = 2
        n_rows = int(np.ceil(n_plots / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4 * n_rows))
        axes = axes.flatten()

    plot_idx = 0

    # Plot state trajectories
    for i in range(n_states):
        ax = axes[plot_idx]
        ax.plot(times1, states1[:, i], 'b-', label=label1, linewidth=2)
        ax.plot(times2, states2[:, i], 'r--', label=label2, linewidth=2)
        ax.axhline(reference_state[i], color='k', linestyle=':', label='Target', alpha=0.5)
        ax.set_xlabel('Time')
        ax.set_ylabel(state_var_names[i])
        ax.set_title(f'{state_var_names[i]} Dynamics')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        plot_idx += 1

    # Plot control inputs
    ax = axes[plot_idx]
    ax.plot(times1[:-1], controls1, 'b-', label=label1, linewidth=2)
    ax.plot(times2[:-1], controls2, 'r--', label=label2, linewidth=2)
    ax.axhline(0.0, color='k', linestyle=':', alpha=0.3)
    ax.set_xlabel('Time')
    ax.set_ylabel('Control')
    ax.set_title('Control Signals')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plot_idx += 1

    # Plot tracking error
    ax = axes[plot_idx]
    ax.plot(times1, tracking_error1, 'b-', label=label1, linewidth=2)
    ax.plot(times2, tracking_error2, 'r--', label=label2, linewidth=2)
    ax.set_xlabel('Time')
    ax.set_ylabel('Tracking Error')
    ax.set_title('Distance to Target')
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plot_idx += 1

    # Plot cumulative control effort
    ax = axes[plot_idx]
    ax.plot(times1, cumulative_control1, 'b-', label=label1, linewidth=2)
    ax.plot(times2, cumulative_control2, 'r--', label=label2, linewidth=2)
    ax.set_xlabel('Time')
    ax.set_ylabel('Cumulative Control Effort')
    ax.set_title('Accumulated Control Cost')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plot_idx += 1

    # Plot cumulative total cost
    ax = axes[plot_idx]
    ax.plot(times1, cumulative_total1, 'b-', label=label1, linewidth=2)
    ax.plot(times2, cumulative_total2, 'r--', label=label2, linewidth=2)
    ax.set_xlabel('Time')
    ax.set_ylabel('Cumulative Total Cost')
    ax.set_title(f'Accumulated Objective (Q={Q_weights.tolist()}, Ru={Ru})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plot_idx += 1

    # Hide unused subplots
    for i in range(plot_idx, len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout()

    # Save plot
    if output is None:
        output = "comparison.pdf"

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved to: {output_path}")
    print()

    print("="*60)
    print("Comparison complete!")
    print("="*60)


if __name__ == "__main__":
    fire.Fire(compare_trajectories)
