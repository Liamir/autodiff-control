"""Plot MPC trajectory in state space with control signal."""
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
from rpa_control.style import set_style


def plot_mpc_state_space(experiment_dir: str, save: bool = True):
    """Plot MPC trajectory as scatter plot in state space.

    Args:
        experiment_dir: Path to MPC experiment directory
        save: Whether to save the plot
    """
    set_style()

    exp_path = Path(experiment_dir)

    # Load config
    config_path = exp_path / "config.json"
    with open(config_path, 'r') as f:
        config = json.load(f)

    # Load MPC trajectory
    mpc_path = exp_path / "mpc_trajectory.json"
    with open(mpc_path, 'r') as f:
        mpc_data = json.load(f)

    # Extract data
    states = np.array(mpc_data['states'])
    controls = np.array(mpc_data['controls'])

    # Get variable names
    state_var_names = config.get('state_var_names', ['x0', 'x1'])
    control_names = config.get('control_names', ['u'])

    # For 2D state space (A, B)
    if states.shape[1] >= 2:
        A = states[:, 0]
        B = states[:, 1]

        # Controls are one step shorter than states
        A_control = A[:-1]
        B_control = B[:-1]

        # Flatten controls if needed
        if len(controls.shape) > 1:
            u = controls[:, 0]
        else:
            u = controls

        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))

        # Scatter plot with control as color
        scatter = ax.scatter(A_control, B_control, c=u, cmap='viridis',
                            s=50, alpha=0.7, edgecolors='black', linewidth=0.5)

        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label(f'{control_names[0]} (control signal)', rotation=270, labelpad=20)

        # Mark start and end points
        ax.plot(A[0], B[0], 'go', markersize=12, label='Start', zorder=5)
        ax.plot(A[-1], B[-1], 'rs', markersize=12, label='End', zorder=5)

        # Add trajectory line (faint)
        ax.plot(A, B, 'k-', alpha=0.2, linewidth=1, zorder=1)

        # Reference state if available
        reference_state = mpc_data.get('reference_state')
        if reference_state is not None:
            ax.plot(reference_state[0], reference_state[1], 'r*',
                   markersize=15, label='Reference', zorder=5)

        ax.set_xlabel(state_var_names[0], fontsize=12)
        ax.set_ylabel(state_var_names[1], fontsize=12)
        ax.set_title('MPC Trajectory in State Space (colored by control)', fontsize=14)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

        plt.tight_layout()

        if save:
            output_path = exp_path / 'mpc_state_space.pdf'
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Saved to: {output_path}")
        else:
            plt.show()

        # Print some statistics
        print("\nState space statistics:")
        print(f"  {state_var_names[0]} range: [{A.min():.2f}, {A.max():.2f}]")
        print(f"  {state_var_names[1]} range: [{B.min():.2f}, {B.max():.2f}]")
        print(f"  Control range: [{u.min():.2f}, {u.max():.2f}]")
        print(f"  Number of points: {len(u)}")

    else:
        print(f"Error: State space must be at least 2D, got {states.shape[1]}D")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python plot_mpc_state_space.py <experiment_dir>")
        print("Example: python plot_mpc_state_space.py logs/ab_controlled_mlp_mpc_20260203_151247")
        sys.exit(1)

    experiment_dir = sys.argv[1]
    plot_mpc_state_space(experiment_dir)
