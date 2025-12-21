"""Plotting utilities for training visualization."""
import torch
import numpy as np
import matplotlib.pyplot as plt
from rpasim.plot.ode import plot_trajectory
from rpa_control.paths import save_fig


def plot_training_comparison(
    ode_initial,
    ode_final,
    initial_state,
    time_horizon,
    target_var_idx=None,
    target_value=None,
    perturb_indices=None,
    perturb_fold_change=2.0,
    n_perturbations=10,
    figsize=(12, 8),
    filename='training_comparison',
    initial_state_range=None,
    n_initial_states=5,
):
    """Plot ODE trajectories before and after training.

    Args:
        ode_initial: ODE with initial parameters
        ode_final: ODE with final (trained) parameters
        initial_state: Initial state for simulation (base state for uncontrolled, or augmented for controlled)
        time_horizon: Simulation time
        target_var_idx: Index of target variable (optional, for plotting target line)
        target_value: Target value for the target variable (optional)
        perturb_indices: Indices of fixed_params to perturb (None = no perturbation)
        perturb_fold_change: Fold change for perturbations (params multiplied by random factor in [1/fold, fold])
        n_perturbations: Number of perturbed trajectories to plot
        figsize: Figure size
        filename: Filename for saving the plot
        initial_state_range: Range for sampling random initial states (tuple of (lower, upper))
        n_initial_states: Number of random initial states to plot (only used if initial_state_range is provided)

    Returns:
        fig, axes: Matplotlib figure and axes
    """
    # Determine if we have a dynamic controlled ODE
    is_final_dynamic = hasattr(ode_final, 'is_dynamic') and ode_final.is_dynamic

    # For dynamic controllers, we need different initial states and only plot base state variables
    if is_final_dynamic:
        # initial_state should be the base state (for uncontrolled ODE)
        base_initial_state = initial_state
        # Get augmented initial state for controlled ODE
        augmented_initial_state = ode_final.get_initial_state(base_initial_state)
        n_vars = len(base_initial_state)  # Only plot base state variables
    else:
        # For static controllers or uncontrolled, use initial_state as-is
        base_initial_state = initial_state
        augmented_initial_state = initial_state
        n_vars = len(initial_state)

    # Create subplots - one row per variable, two columns (before/after)
    fig, axes = plt.subplots(n_vars, 2, figsize=figsize)

    # Handle single variable case
    if n_vars == 1:
        axes = axes.reshape(1, -1)

    # Before training (always use base initial state)
    fig_before, axes_before = plot_trajectory(ode_initial, base_initial_state, time_horizon)

    # Format initial parameters
    if hasattr(ode_initial, 'differentiable_params') and ode_initial.differentiable_params is not None:
        params_init = ode_initial.differentiable_params
        if params_init.dim() == 0:
            params_init_str = f"θ={params_init.item():.2f}"
        else:
            # Flatten if multi-dimensional
            params_init_flat = params_init.flatten()
            params_init_str = "θ=[" + ", ".join([f"{p.item():.2f}" for p in params_init_flat]) + "]"
    else:
        params_init_str = ""

    # Copy plots to comparison figure
    for i, ax_before in enumerate(axes_before):
        lines = ax_before.get_lines()
        for line in lines:
            axes[i, 0].plot(line.get_xdata(), line.get_ydata())
        axes[i, 0].set_xlabel('time')
        axes[i, 0].set_ylabel(ax_before.get_ylabel())
        title = f'{ax_before.get_ylabel()} - before training'
        if params_init_str:
            title += f'\n{params_init_str}'
        axes[i, 0].set_title(title)

        # Add target line if specified
        if target_var_idx == i and target_value is not None:
            axes[i, 0].axhline(y=target_value, color='red', linestyle='--',
                             label='target', alpha=0.5)
    plt.close(fig_before)

    # After training (use augmented state for dynamic controllers)
    fig_after, axes_after = plot_trajectory(ode_final, augmented_initial_state, time_horizon)

    # Format final parameters
    if hasattr(ode_final, 'differentiable_params') and ode_final.differentiable_params is not None:
        params_final = ode_final.differentiable_params
        if params_final.dim() == 0:
            params_final_str = f"θ={params_final.item():.2f}"
        else:
            # Flatten if multi-dimensional
            params_final_flat = params_final.flatten()
            params_final_str = "θ=[" + ", ".join([f"{p.item():.2f}" for p in params_final_flat]) + "]"
    else:
        params_final_str = ""

    # Copy plots to comparison figure (only base state variables for dynamic controllers)
    for i in range(n_vars):
        ax_after = axes_after[i]
        lines = ax_after.get_lines()
        for line in lines:
            axes[i, 1].plot(line.get_xdata(), line.get_ydata(), color='orange')
        axes[i, 1].set_xlabel('time')
        axes[i, 1].set_ylabel(ax_after.get_ylabel())
        title = f'{ax_after.get_ylabel()} - after training'
        if params_final_str:
            title += f'\n{params_final_str}'
        axes[i, 1].set_title(title)

        # Add target line if specified
        if target_var_idx == i and target_value is not None:
            axes[i, 1].axhline(y=target_value, color='red', linestyle='--',
                             label='target', alpha=0.5)
    plt.close(fig_after)

    # Add trajectories from multiple random initial states if range is provided
    if initial_state_range is not None:
        # Handle both single tuple (lower, upper) and list of tuples [(lower1, upper1), (lower2, upper2), ...]
        if isinstance(initial_state_range, tuple):
            # Single tuple: apply to all variables
            lower = torch.tensor([initial_state_range[0]] * len(base_initial_state))
            upper = torch.tensor([initial_state_range[1]] * len(base_initial_state))
        else:
            # List of tuples: one per variable
            lower = torch.tensor([lim[0] for lim in initial_state_range])
            upper = torch.tensor([lim[1] for lim in initial_state_range])

        for _ in range(n_initial_states - 1):  # -1 because we already plotted one initial state
            # Sample random initial state
            random_factors = torch.rand_like(base_initial_state)
            random_init_state = lower + random_factors * (upper - lower)

            # For dynamic controllers, get augmented state
            if is_final_dynamic:
                random_augmented_state = ode_final.get_initial_state(random_init_state)
            else:
                random_augmented_state = random_init_state

            # Plot uncontrolled trajectory
            fig_rand_before, axes_rand_before = plot_trajectory(ode_initial, random_init_state, time_horizon)
            for i in range(n_vars):
                ax_rand_before = axes_rand_before[i]
                lines = ax_rand_before.get_lines()
                for line in lines:
                    axes[i, 0].plot(line.get_xdata(), line.get_ydata(),
                                   alpha=0.4, linewidth=1.0, zorder=1)
            plt.close(fig_rand_before)

            # Plot controlled trajectory
            fig_rand_after, axes_rand_after = plot_trajectory(ode_final, random_augmented_state, time_horizon)
            for i in range(n_vars):
                ax_rand_after = axes_rand_after[i]
                lines = ax_rand_after.get_lines()
                for line in lines:
                    axes[i, 1].plot(line.get_xdata(), line.get_ydata(),
                                   color='orange', alpha=0.4, linewidth=1.0, zorder=1)
            plt.close(fig_rand_after)

    # Add perturbed trajectories to show robustness (always shown, even if training didn't use perturbations)
    if perturb_indices is not None and len(perturb_indices) > 0 and hasattr(ode_final, 'fixed_params') and ode_final.fixed_params is not None:
        # Save original fixed params
        original_fixed_params = ode_final.fixed_params.clone()

        for _ in range(n_perturbations):
            # Create perturbed version of fixed params
            perturbed_fixed_params = ode_final.fixed_params.clone()

            # Perturb specified parameters (same logic as training)
            for idx in perturb_indices:
                fold = perturb_fold_change
                # Sample uniformly in log-space: log(1/fold) to log(fold)
                log_factor = torch.rand(1).item() * 2 * torch.log(torch.tensor(fold)).item() - torch.log(torch.tensor(fold)).item()
                random_factor = torch.exp(torch.tensor(log_factor)).item()
                perturbed_fixed_params[idx] = perturbed_fixed_params[idx] * random_factor

            # Temporarily set perturbed params
            ode_final.fixed_params = perturbed_fixed_params

            # Generate trajectory with perturbed params (use augmented state for dynamic)
            fig_perturbed, axes_perturbed = plot_trajectory(ode_final, augmented_initial_state, time_horizon)

            # Add to comparison plot as gray lines (only base state variables)
            for i in range(n_vars):
                ax_perturbed = axes_perturbed[i]
                lines = ax_perturbed.get_lines()
                for line in lines:
                    axes[i, 1].plot(line.get_xdata(), line.get_ydata(),
                                   color='gray', alpha=0.5, linewidth=0.8, zorder=1)
            plt.close(fig_perturbed)

        # Restore original fixed params
        ode_final.fixed_params = original_fixed_params

    # Remove top and right spines from all subplots
    for ax_row in axes:
        for ax in ax_row:
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

    plt.tight_layout()
    save_fig(fig, filename)
    print(f"saved trajectory comparison to plots/{filename}.pdf")

    return fig, axes


def plot_training_curves(history, figsize=(12, 8), filename='training_curves'):
    """Plot training curves (loss, reward, sparsity, regularization).

    Args:
        history: Training history dict from train_ode_parameters
        figsize: Figure size
        filename: Filename for saving the plot

    Returns:
        fig, axes: Matplotlib figure and axes
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)

    # Loss
    axes[0, 0].plot(history['loss'])
    axes[0, 0].set_xlabel('iteration')
    axes[0, 0].set_ylabel('loss')
    axes[0, 0].set_title('training loss')

    # Reward
    axes[0, 1].plot(history['reward'])
    axes[0, 1].set_xlabel('iteration')
    axes[0, 1].set_ylabel('reward')
    axes[0, 1].set_title('reward (sum)')

    # Sparsity
    axes[1, 0].plot(history['num_nonzero_params'])
    axes[1, 0].set_xlabel('iteration')
    axes[1, 0].set_ylabel('non-zero parameters')
    axes[1, 0].set_title('parameter sparsity')

    # Regularization
    has_l1 = any(x > 0 for x in history['l1_penalty'])
    has_l2 = any(x > 0 for x in history['l2_penalty'])

    if has_l1:
        axes[1, 1].plot(history['l1_penalty'], label='l1')
    if has_l2:
        axes[1, 1].plot(history['l2_penalty'], label='l2')

    axes[1, 1].set_xlabel('iteration')
    axes[1, 1].set_ylabel('penalty')
    axes[1, 1].set_title('regularization penalties')

    if has_l1 or has_l2:
        axes[1, 1].legend()

    # Remove top and right spines from all subplots
    for ax_row in axes:
        for ax in ax_row:
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

    plt.tight_layout()
    save_fig(fig, filename)
    print(f"saved training curves to plots/{filename}.pdf")

    return fig, axes


def plot_controller_heatmap(
    controller,
    state_ranges,
    state_names,
    critical_point=None,
    resolution=100,
    figsize=(10, 8),
    filename='controller_heatmap',
):
    """Plot controller behavior as a heatmap over 2D state space.

    Args:
        controller: Trained controller (StaticController or DynamicController)
        state_ranges: List of (min, max) tuples for each state variable
        state_names: List of state variable names
        critical_point: Optional target equilibrium point [state1, state2]
        resolution: Number of grid points in each direction
        figsize: Figure size
        filename: Filename for saving the plot

    Returns:
        fig, ax: Matplotlib figure and axes
    """
    if len(state_ranges) != 2 or len(state_names) != 2:
        raise ValueError("Controller heatmap only supports 2D state spaces")

    # Create state space grid
    state1_vals = np.linspace(state_ranges[0][0], state_ranges[0][1], resolution)
    state2_vals = np.linspace(state_ranges[1][0], state_ranges[1][1], resolution)
    state1_grid, state2_grid = np.meshgrid(state1_vals, state2_vals)

    # Compute control signal at each point
    control_grid = np.zeros_like(state1_grid)

    for i in range(resolution):
        for j in range(resolution):
            state = torch.tensor([state1_grid[i, j], state2_grid[i, j]], dtype=torch.float32)

            with torch.no_grad():
                # For static controller
                if hasattr(controller, 'params'):
                    control = controller(state)
                    control_grid[i, j] = control.item()
                # For dynamic controller (use zero controller state)
                elif hasattr(controller, 'output'):
                    control = controller.output(torch.zeros(controller.n_controller_states))
                    control_grid[i, j] = control.item()

    # Create plot
    fig, ax = plt.subplots(figsize=figsize)

    # Plot heatmap
    im = ax.contourf(state1_grid, state2_grid, control_grid, levels=50, cmap='RdBu_r')

    # Add contour lines
    contours = ax.contour(state1_grid, state2_grid, control_grid, levels=10,
                          colors='black', alpha=0.3, linewidths=0.5)
    ax.clabel(contours, inline=True, fontsize=8, fmt='%.1f')

    # Mark critical point if provided
    if critical_point is not None:
        ax.plot(critical_point[0], critical_point[1], 'k*', markersize=20,
                label=f'Critical Point ({critical_point[0]:.0f}, {critical_point[1]:.0f})',
                markeredgewidth=1.5, markeredgecolor='white')

    # Add zero control contour (dashed line)
    zero_contour = ax.contour(state1_grid, state2_grid, control_grid, levels=[0],
                               colors='green', linewidths=2, linestyles='--')
    ax.clabel(zero_contour, inline=True, fontsize=10, fmt='u=%.1f')

    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Control Signal u', rotation=270, labelpad=20)

    # Labels and title
    ax.set_xlabel(state_names[0])
    ax.set_ylabel(state_names[1])
    ax.set_title('Controller Behavior: Control Signal Heatmap')
    if critical_point is not None:
        ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    save_fig(fig, filename)
    print(f"saved controller heatmap to plots/{filename}.pdf")

    return fig, ax
