"""Plot trajectory with gray shading when control < 0.5."""
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
from rpa_control.style import set_style


def plot_trajectory_with_shading(experiment_dir: str, control_threshold: float = 0.5):
    """Plot trajectory with background shading based on control value.

    Gray shading is added to state plots when control < threshold.

    Args:
        experiment_dir: Path to experiment directory (training or MPC)
        control_threshold: Control threshold for shading (default: 0.5)
    """
    set_style()

    exp_path = Path(experiment_dir)

    # Load config
    config_path = exp_path / "config.json"
    with open(config_path, 'r') as f:
        config = json.load(f)

    # Determine if this is training or MPC
    method = config.get('method', 'training')

    if method == 'mpc':
        # Load MPC trajectory
        traj_path = exp_path / 'mpc_trajectory.json'
    else:
        # Load training trajectory
        traj_path = exp_path / 'trajectory.json'

    if not traj_path.exists():
        print(f"No trajectory file found at {traj_path}")
        return

    with open(traj_path, 'r') as f:
        traj_data = json.load(f)

    times = np.array(traj_data['times'])
    states = np.array(traj_data['states'])
    controls = np.array(traj_data['controls'])
    reference_state = traj_data.get('reference_state')

    # Get variable names
    state_var_names = config.get('state_var_names', [f'x{i}' for i in range(states.shape[1])])
    control_names = config.get('control_names', ['u'])

    n_states = states.shape[1]
    n_controls = controls.shape[1] if len(controls.shape) > 1 else 1

    # Create figure
    fig, axes = plt.subplots(n_states + 1, 1, figsize=(10, 3 * (n_states + 1)))
    if n_states == 1:
        axes = [axes[0], axes[1]]

    # Flatten controls for easier access
    u_flat = controls.flatten() if len(controls.shape) > 1 else controls
    control_times = times[:-1]  # Control is one step shorter

    # Plot states with shading
    for i in range(n_states):
        ax = axes[i]

        # Add gray shading for regions where control < threshold
        for j in range(len(u_flat)):
            if u_flat[j] < control_threshold:
                # Shade this time interval
                t_start = control_times[j]
                t_end = control_times[j+1] if j+1 < len(control_times) else times[-1]
                ax.axvspan(t_start, t_end, alpha=0.15, color='gray', zorder=0)

        # Plot state trajectory
        ax.plot(times, states[:, i], 'b-', linewidth=2,
               label=f'{state_var_names[i]} ({"trained" if method == "training" else "MPC"})')

        # Plot reference if available
        if reference_state is not None:
            ref_array = np.array(reference_state)
            ax.axhline(ref_array[i], color='r', linestyle='--', alpha=0.5, label='target')

        ax.set_xlabel('Time')
        ax.set_ylabel(state_var_names[i])
        ax.set_title(f'{state_var_names[i]} Evolution (gray = u < {control_threshold})')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

    # Plot control
    ax = axes[-1]
    control_bounds = config.get('control_bounds', None)

    if n_controls == 1:
        # Plot raw control
        ax.plot(control_times, u_flat, 'g--', linewidth=2, alpha=0.6,
               label=f'{control_names[0]} (raw)')

        # Plot clamped control if bounds available
        if control_bounds is not None:
            controls_clamped = np.clip(u_flat, control_bounds[0], control_bounds[1])
            ax.plot(control_times, controls_clamped, 'g-', linewidth=2,
                   label=f'{control_names[0]} (clamped)')

            # Add horizontal lines for bounds
            ax.axhline(control_bounds[0], color='r', linestyle=':', alpha=0.5,
                      label=f'lower bound ({control_bounds[0]})')
            ax.axhline(control_bounds[1], color='r', linestyle=':', alpha=0.5,
                      label=f'upper bound ({control_bounds[1]})')

        # Add threshold line
        ax.axhline(control_threshold, color='gray', linestyle='--', alpha=0.5,
                  label=f'threshold ({control_threshold})')
    else:
        for i in range(n_controls):
            ax.plot(control_times, controls[:, i], linewidth=2, alpha=0.6, linestyle='--',
                   label=f'{control_names[i] if i < len(control_names) else f"u{i}"} (raw)')

            if control_bounds is not None:
                controls_clamped = np.clip(controls[:, i], control_bounds[0], control_bounds[1])
                ax.plot(control_times, controls_clamped, linewidth=2,
                       label=f'{control_names[i] if i < len(control_names) else f"u{i}"} (clamped)')

    ax.axhline(0.0, color='k', linestyle='--', alpha=0.3)
    ax.set_xlabel('Time')
    ax.set_ylabel('Control')
    ax.set_title('Control Signal')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    plt.tight_layout()

    # Save
    output_path = exp_path / 'trajectory_with_shading.pdf'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved to: {output_path}")

    # Print statistics
    print(f"\nControl statistics:")
    print(f"  Control < {control_threshold}: {(u_flat < control_threshold).sum()} / {len(u_flat)} steps ({(u_flat < control_threshold).mean()*100:.1f}%)")
    print(f"  Control >= {control_threshold}: {(u_flat >= control_threshold).sum()} / {len(u_flat)} steps ({(u_flat >= control_threshold).mean()*100:.1f}%)")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python plot_trajectory_with_control_shading.py <experiment_dir> [threshold]")
        print("Example: python plot_trajectory_with_control_shading.py logs/ab_controlled_mlp_20260203_111814 0.5")
        sys.exit(1)

    experiment_dir = sys.argv[1]
    threshold = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5

    plot_trajectory_with_shading(experiment_dir, threshold)
