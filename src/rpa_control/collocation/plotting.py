"""Plotting utilities for collocation optimal control results."""
import numpy as np
import matplotlib.pyplot as plt


def plot_collocation_result(
    result: dict,
    state_names: list[str] = None,
    control_names: list[str] = None,
    target_state: np.ndarray = None,
    tracked_indices: list[int] = None,
    title: str = None,
    save_path: str = None,
    separate_controls: bool = False,
    u_ref: np.ndarray = None,
    highlight_states: list[int] = None,
):
    """Plot state trajectories and control profiles from collocation solution.

    Args:
        result: Output dict from solve_optimal_control.
        state_names: Names for state variables.
        control_names: Names for control inputs.
        target_state: Target state for reference lines.
        tracked_indices: Which states are tracked (for highlighting).
        title: Figure title.
        save_path: Path to save figure (if None, calls plt.show()).
        separate_controls: If True, plot each control on its own subplot.
        u_ref: Reference control (null action) for reference lines.
        highlight_states: State indices to plot on their own subplots (e.g. [2] for Cortisol).
    """
    t = result['t']
    x = result['x']
    u = result['u']
    n_states = x.shape[1]
    n_controls = u.shape[1]

    if state_names is None:
        state_names = [f'x{i+1}' for i in range(n_states)]
    if control_names is None:
        control_names = [f'u{i+1}' for i in range(n_controls)]

    if highlight_states is None:
        highlight_states = []

    # Determine active controls for separate_controls mode
    if separate_controls:
        active_controls = [i for i in range(n_controls) if not np.allclose(u[:, i], u[0, i])]
        if not active_controls:
            active_controls = list(range(n_controls))
    else:
        active_controls = []

    # Build subplot layout: all states | highlighted states | controls
    n_rows = 1 + len(highlight_states) + (len(active_controls) if separate_controls else 1)
    fig, axes = plt.subplots(n_rows, 1, figsize=(10, 2.5 * n_rows), sharex=True)
    if n_rows == 1:
        axes = [axes]

    row = 0

    # All state trajectories (excluding highlighted ones)
    ax = axes[row]
    for i in range(n_states):
        if i in highlight_states:
            continue
        ax.plot(t, x[:, i], lw=1.5, label=state_names[i])
        if target_state is not None and tracked_indices is not None and i in tracked_indices:
            ax.axhline(target_state[i], color='black', ls='--', lw=0.8, alpha=0.4)
    ax.set_ylabel('state')
    ax.legend(fontsize=8)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    if title:
        ax.set_title(title, fontsize=11)
    row += 1

    # Highlighted individual state subplots
    for si in highlight_states:
        ax = axes[row]
        ax.plot(t, x[:, si], lw=1.5, color=f'C{si}')
        if target_state is not None and tracked_indices is not None and si in tracked_indices:
            ax.axhline(target_state[si], color='black', ls='--', lw=0.8, alpha=0.4)
        ax.set_ylabel(state_names[si])
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        row += 1

    # Control profiles
    t_ctrl = t[:-1]
    if separate_controls:
        for ci in active_controls:
            ax = axes[row]
            ax.step(t_ctrl, u[:, ci], where='post', lw=1.5, color=f'C{ci}')
            if u_ref is not None:
                ax.axhline(u_ref[ci], color='black', ls='--', lw=0.8, alpha=0.4)
            ax.set_ylabel(control_names[ci])
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            row += 1
    else:
        ax = axes[row]
        for i in range(n_controls):
            ax.step(t_ctrl, u[:, i], where='post', lw=1.5, label=control_names[i])
        ax.set_ylabel('control')
        ax.legend(fontsize=8)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

    axes[-1].set_xlabel('time')

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    else:
        plt.show()

    return fig
