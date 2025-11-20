"""Plotting utilities for training visualization."""
import torch
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
    filename='training_comparison'
):
    """Plot ODE trajectories before and after training.

    Args:
        ode_initial: ODE with initial parameters
        ode_final: ODE with final (trained) parameters
        initial_state: Initial state for simulation
        time_horizon: Simulation time
        target_var_idx: Index of target variable (optional, for plotting target line)
        target_value: Target value for the target variable (optional)
        perturb_indices: Indices of fixed_params to perturb (None = no perturbation)
        perturb_fold_change: Fold change for perturbations (params multiplied by random factor in [1/fold, fold])
        n_perturbations: Number of perturbed trajectories to plot
        figsize: Figure size
        filename: Filename for saving the plot

    Returns:
        fig, axes: Matplotlib figure and axes
    """
    # Get number of state variables
    n_vars = len(initial_state)

    # Create subplots - one row per variable, two columns (before/after)
    fig, axes = plt.subplots(n_vars, 2, figsize=figsize)

    # Handle single variable case
    if n_vars == 1:
        axes = axes.reshape(1, -1)

    # Before training
    fig_before, axes_before = plot_trajectory(ode_initial, initial_state, time_horizon)

    # Format initial parameters
    if hasattr(ode_initial, 'differentiable_params') and ode_initial.differentiable_params is not None:
        params_init = ode_initial.differentiable_params
        if params_init.dim() == 0:
            params_init_str = f"θ={params_init.item():.2f}"
        else:
            params_init_str = "θ=[" + ", ".join([f"{p.item():.2f}" for p in params_init]) + "]"
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

    # After training
    fig_after, axes_after = plot_trajectory(ode_final, initial_state, time_horizon)

    # Format final parameters
    if hasattr(ode_final, 'differentiable_params') and ode_final.differentiable_params is not None:
        params_final = ode_final.differentiable_params
        if params_final.dim() == 0:
            params_final_str = f"θ={params_final.item():.2f}"
        else:
            params_final_str = "θ=[" + ", ".join([f"{p.item():.2f}" for p in params_final]) + "]"
    else:
        params_final_str = ""

    # Copy plots to comparison figure
    for i, ax_after in enumerate(axes_after):
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

            # Generate trajectory with perturbed params
            fig_perturbed, axes_perturbed = plot_trajectory(ode_final, initial_state, time_horizon)

            # Add to comparison plot as gray lines
            for i, ax_perturbed in enumerate(axes_perturbed):
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
