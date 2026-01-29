"""Compare ODE solver trajectories to assess accuracy vs speed tradeoffs."""
import torch
import matplotlib.pyplot as plt
import importlib
import sys
from pathlib import Path
from torchdiffeq import odeint
import numpy as np

# Add src and configs to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root))

from rpa_control.style import set_style


def compare_solvers(env_name, solvers=['dopri5', 'euler']):
    """Compare trajectories from different ODE solvers.

    Args:
        env_name: Name of the config module (e.g., 'hpa_i1')
        solvers: List of solver names to compare
    """
    # Import config
    config_module = importlib.import_module(f"configs.{env_name}")
    config = config_module.ENV_CONFIG

    # Create base ODE
    base_ode = config['create_base_ode']()

    # Get initial state and time horizon
    initial_state = config['initial_state']
    time_horizon = config['defaults']['time_horizon']
    n_reward_steps = config['defaults']['n_reward_steps']

    # Create time points
    t_span = torch.linspace(0, time_horizon, n_reward_steps + 1)

    # Get state variable names
    state_var_names = config.get('state_var_names', [f'x{i}' for i in range(len(initial_state))])
    target_vars = config.get('target_vars', {})

    # Simulate with each solver
    print(f"\nComparing solvers for {env_name}:")
    print(f"  Time horizon: {time_horizon}")
    print(f"  Time steps: {len(t_span)}")
    print(f"  State dimension: {len(initial_state)}")
    print()

    trajectories = {}
    times_taken = {}

    for solver in solvers:
        print(f"Simulating with {solver}...")
        import time
        start = time.time()
        trajectory = odeint(base_ode, initial_state, t_span, method=solver)
        elapsed = time.time() - start
        trajectories[solver] = trajectory
        times_taken[solver] = elapsed
        print(f"  Time: {elapsed:.4f}s")

    # Compute differences
    if len(solvers) == 2:
        ref_solver = solvers[0]
        test_solver = solvers[1]

        ref_traj = trajectories[ref_solver].detach().numpy()
        test_traj = trajectories[test_solver].detach().numpy()

        # Compute error metrics
        abs_error = np.abs(ref_traj - test_traj)
        rel_error = abs_error / (np.abs(ref_traj) + 1e-8)

        print(f"\nError metrics ({test_solver} vs {ref_solver}):")
        print(f"  Max absolute error: {abs_error.max():.6f}")
        print(f"  Mean absolute error: {abs_error.mean():.6f}")
        print(f"  Max relative error: {rel_error.max():.6f}")
        print(f"  Mean relative error: {rel_error.mean():.6f}")

        # Per-variable errors
        print(f"\nPer-variable max absolute errors:")
        for i, name in enumerate(state_var_names):
            max_err = abs_error[:, i].max()
            mean_err = abs_error[:, i].mean()
            print(f"  {name}: max={max_err:.6f}, mean={mean_err:.6f}")

    # Plot trajectories
    set_style()
    n_states = len(initial_state)
    fig, axes = plt.subplots(n_states, 1, figsize=(10, 2.5 * n_states), sharex=True)
    if n_states == 1:
        axes = [axes]

    t_np = t_span.numpy()
    colors = plt.cm.tab10(np.linspace(0, 1, len(solvers)))

    for i, name in enumerate(state_var_names):
        ax = axes[i]

        # Plot trajectories
        for j, solver in enumerate(solvers):
            traj = trajectories[solver][:, i].detach().numpy()
            speedup = times_taken[solvers[0]] / times_taken[solver] if solver != solvers[0] else 1.0
            label = f"{solver} ({times_taken[solver]:.3f}s)"
            if speedup != 1.0:
                label += f" [{speedup:.1f}x]"
            ax.plot(t_np, traj, label=label, color=colors[j], linewidth=2, alpha=0.8)

        # Plot target if exists
        if i in target_vars:
            ax.axhline(target_vars[i], color='gray', linestyle='--', linewidth=1,
                      label=f'Target ({target_vars[i]:.2f})', alpha=0.5)

        ax.set_ylabel(name, fontsize=12)
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel('Time', fontsize=12)

    # Add title
    speedup_str = f"{times_taken[solvers[0]]/times_taken[solvers[1]]:.1f}x" if len(solvers) == 2 else ""
    fig.suptitle(f'{env_name.upper()}: Solver Comparison {speedup_str}', fontsize=14, fontweight='bold')

    plt.tight_layout()

    # Save figure
    output_path = project_root / f'solver_comparison_{env_name}.pdf'
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    print(f"\nPlot saved to: {output_path}")

    return trajectories, times_taken


def main():
    """Compare different solvers for HPA environment."""
    import sys

    # Try euler first
    print("="*70)
    print("EULER vs DOPRI5")
    print("="*70)
    try:
        compare_solvers('hpa_i1', solvers=['dopri5', 'euler'])
    except Exception as e:
        print(f"Error with euler: {e}")

    # Try rk4
    print("\n" + "="*70)
    print("RK4 vs DOPRI5")
    print("="*70)
    try:
        compare_solvers('hpa_i1', solvers=['dopri5', 'rk4'])
    except Exception as e:
        print(f"Error with rk4: {e}")


if __name__ == "__main__":
    main()
