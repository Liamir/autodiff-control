"""Simple script to plot ODE trajectories with optional perturbations."""
import fire
import torch
import matplotlib.pyplot as plt
from rpasim.ode.ab import AB
from rpasim.plot.ode import plot_trajectory
from rpa_control.paths import save_fig
from rpa_control.style import set_style


def plot_ab_ode(
    alpha1: float = 1.0,
    alpha2: float = 0.0,
    alpha3: float = -1.0,
    beta1: float = 1.0,
    beta2: float = 1.0,
    initial_a: float = 1.0,
    initial_b: float = 0.5,
    time_horizon: float = 100.0,
    target_b: float = 1.0,
    perturb_betas: bool = False,
    perturb_fold_change: float = 5.0,
    n_perturbations: int = 10,
    filename: str = 'ode_plot'
):
    """Plot AB ODE trajectory with optional perturbations.

    Args:
        alpha1, alpha2, alpha3: Alpha parameters for the ODE
        beta1, beta2: Beta parameters (can be perturbed)
        initial_a, initial_b: Initial state values
        time_horizon: Simulation time
        target_b: Target value for B (plotted as horizontal line)
        perturb_betas: Whether to show perturbed trajectories
        perturb_fold_change: Fold change for perturbations
        n_perturbations: Number of perturbed trajectories to show
        filename: Output filename for the plot
    """
    set_style()

    # Create ODE
    alphas = torch.tensor([alpha1, alpha2, alpha3])
    betas = torch.tensor([beta1, beta2])
    ode = AB(differentiable_params=alphas, fixed_params=betas)

    # Initial state
    initial_state = torch.tensor([initial_a, initial_b])

    # Create base plot
    fig, axes = plot_trajectory(ode, initial_state, time_horizon)

    # Get number of variables
    n_vars = len(initial_state)

    # Make axes iterable
    if n_vars == 1:
        axes = [axes]

    # Add perturbed trajectories if requested
    if perturb_betas:
        # Save original betas
        original_betas = ode.fixed_params.clone()

        for _ in range(n_perturbations):
            # Create perturbed betas
            perturbed_betas = ode.fixed_params.clone()

            # Perturb both beta1 and beta2
            for idx in [0, 1]:
                fold = perturb_fold_change
                # Sample uniformly in log-space
                log_factor = torch.rand(1).item() * 2 * torch.log(torch.tensor(fold)).item() - torch.log(torch.tensor(fold)).item()
                random_factor = torch.exp(torch.tensor(log_factor)).item()
                perturbed_betas[idx] = perturbed_betas[idx] * random_factor

            # Temporarily set perturbed params
            ode.fixed_params = perturbed_betas

            # Generate trajectory
            fig_perturbed, axes_perturbed = plot_trajectory(ode, initial_state, time_horizon)

            # Make perturbed axes iterable
            if n_vars == 1:
                axes_perturbed = [axes_perturbed]

            # Add to main plot as gray lines
            for i, ax_perturbed in enumerate(axes_perturbed):
                lines = ax_perturbed.get_lines()
                for line in lines:
                    axes[i].plot(line.get_xdata(), line.get_ydata(),
                               color='gray', alpha=0.5, linewidth=0.8, zorder=1)
            plt.close(fig_perturbed)

        # Restore original betas
        ode.fixed_params = original_betas

    # Add target line for B (second variable)
    if target_b is not None and n_vars >= 2:
        axes[1].axhline(y=target_b, color='red', linestyle='--',
                       label='target', alpha=0.5)
        axes[1].legend()

    # Update titles with parameter values
    param_str = f"α=[{alpha1:.2f}, {alpha2:.2f}, {alpha3:.2f}], β=[{beta1:.2f}, {beta2:.2f}]"
    if n_vars == 2:
        axes[0].set_title(f"A\n{param_str}")
        axes[1].set_title(f"B")

    # Remove top and right spines
    for ax in axes:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.tight_layout()
    save_fig(fig, filename)
    print(f"saved plot to plots/{filename}.pdf")
    print(f"Parameters: {param_str}")

    return fig, axes


if __name__ == "__main__":
    fire.Fire(plot_ab_ode)
