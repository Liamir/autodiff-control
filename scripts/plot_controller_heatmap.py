"""Visualize controller behavior as a heatmap over state space.

This script creates a 2D heatmap showing the control signal as a function
of the observed state variables, providing insight into controller behavior.
"""
import fire
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from rpasim.ode.classic_control.population import PopulationDynamics
from rpa_control.controllers import StaticController, DynamicController
from rpa_control.controllers.controlled_ode import ControlledODE
from rpa_control.style import set_style


def plot_controller_heatmap(
    controller_path: str = None,
    controller_order: int = 2,
    include_constant: bool = True,
    controller_type: str = "static",  # "static" or "dynamic"
    # State space range
    prey_min: float = 60.0,
    prey_max: float = 140.0,
    predator_min: float = 10.0,
    predator_max: float = 30.0,
    resolution: int = 100,
    # Critical point (for reference)
    critical_point: list[float] = [100.0, 20.0],
    # Output
    save_plot: bool = True,
    output_path: str = "plots/controller_heatmap.pdf",
):
    """Create a heatmap visualization of controller behavior.

    Args:
        controller_path: Path to trained controller parameters (results.txt)
        controller_order: Polynomial order for controller
        include_constant: Include constant term in controller
        controller_type: Type of controller ("static" or "dynamic")
        prey_min: Minimum prey population for heatmap
        prey_max: Maximum prey population for heatmap
        predator_min: Minimum predator population for heatmap
        predator_max: Maximum predator population for heatmap
        resolution: Number of grid points in each direction
        critical_point: Target equilibrium point [prey, predator]
        save_plot: Whether to save the plot
        output_path: Path to save the plot
    """
    set_style()

    print("="*60)
    print("Controller Explainability: Control Heatmap")
    print("="*60)
    print()

    # Create controller
    if controller_type == "static":
        controller = StaticController(
            n_state_vars=2,
            n_control_vars=1,
            order=controller_order,
            include_constant=include_constant,
        )
    elif controller_type == "dynamic":
        from rpa_control.controllers import DynamicController
        controller = DynamicController(
            n_state_vars=2,
            n_control_vars=1,
            n_controller_states=2,
            observing_order=controller_order,
            actuating_order=controller_order,
            include_constant=include_constant,
        )
    else:
        raise ValueError(f"Unknown controller type: {controller_type}")

    # Load controller parameters if provided
    if controller_path is not None:
        print(f"Loading controller from: {controller_path}")
        results_path = Path(controller_path)
        if results_path.is_dir():
            results_file = results_path / "results.txt"
        else:
            results_file = results_path

        if results_file.exists():
            with open(results_file, 'r') as f:
                for line in f:
                    if line.startswith('best_params_physical:'):
                        params_str = line.split(':', 1)[1].strip()
                        params_list = eval(params_str)
                        if isinstance(params_list[0], list):
                            params_list = params_list[0]

                        if controller_type == "static":
                            controller.params.data = torch.tensor(params_list, dtype=torch.float32).reshape(1, -1)
                        elif controller_type == "dynamic":
                            # For dynamic controller, split params
                            n_obs = controller.observing_params.numel()
                            controller.observing_params.data = torch.tensor(params_list[:n_obs], dtype=torch.float32).reshape(controller.observing_params.shape)
                            controller.actuating_params.data = torch.tensor(params_list[n_obs:], dtype=torch.float32).reshape(controller.actuating_params.shape)

                        print(f"  Loaded physical parameters: {params_list}")
                        break
        else:
            print(f"  Warning: File not found, using default initialization")
    else:
        print("Using default controller initialization")

    print()

    # Create state space grid
    print(f"Creating state space grid ({resolution}x{resolution})...")
    prey_vals = np.linspace(prey_min, prey_max, resolution)
    predator_vals = np.linspace(predator_min, predator_max, resolution)
    prey_grid, predator_grid = np.meshgrid(prey_vals, predator_vals)

    # Compute control signal at each point
    control_grid = np.zeros_like(prey_grid)

    for i in range(resolution):
        for j in range(resolution):
            state = torch.tensor([prey_grid[i, j], predator_grid[i, j]], dtype=torch.float32)

            if controller_type == "static":
                with torch.no_grad():
                    control = controller(state)
                control_grid[i, j] = control.item()
            elif controller_type == "dynamic":
                # For dynamic controller, use zero controller state (initial condition)
                with torch.no_grad():
                    control = controller.output(torch.zeros(controller.n_controller_states))
                control_grid[i, j] = control.item()

    print(f"Control signal range: [{control_grid.min():.2f}, {control_grid.max():.2f}]")
    print()

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot heatmap
    im = ax.contourf(prey_grid, predator_grid, control_grid, levels=50, cmap='RdBu_r')

    # Add contour lines
    contours = ax.contour(prey_grid, predator_grid, control_grid, levels=10, colors='black', alpha=0.3, linewidths=0.5)
    ax.clabel(contours, inline=True, fontsize=8, fmt='%.1f')

    # Mark critical point
    ax.plot(critical_point[0], critical_point[1], 'k*', markersize=20,
            label=f'Critical Point ({critical_point[0]:.0f}, {critical_point[1]:.0f})',
            markeredgewidth=1.5, markeredgecolor='white')

    # Add zero control contour (dashed line)
    zero_contour = ax.contour(prey_grid, predator_grid, control_grid, levels=[0], colors='green', linewidths=2, linestyles='--')
    ax.clabel(zero_contour, inline=True, fontsize=10, fmt='u=%.1f')

    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Control Signal u', rotation=270, labelpad=20)

    # Labels and title
    ax.set_xlabel('Prey Population')
    ax.set_ylabel('Predator Population')
    ax.set_title('Controller Behavior: Control Signal Heatmap')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_plot:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved heatmap to: {output_file}")

    plt.show()


if __name__ == "__main__":
    fire.Fire(plot_controller_heatmap)
