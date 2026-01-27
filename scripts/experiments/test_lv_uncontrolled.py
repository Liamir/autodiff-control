"""Visualize uncontrolled Lotka-Volterra limit cycle behavior."""
import fire
import torch
import matplotlib.pyplot as plt
from rpasim.ode.classic_control.population import PopulationDynamics
from rpasim.plot.ode import plot_trajectory
from rpa_control.paths import save_fig
from rpa_control.style import set_style


def test_lv_uncontrolled(
    initial_prey: float = 80.0,
    initial_predator: float = 25.0,
    time_horizon: float = 100.0,
    filename: str = 'lv_uncontrolled_oscillation'
):
    """Plot uncontrolled Lotka-Volterra dynamics showing limit cycle.

    Args:
        initial_prey: Initial prey population
        initial_predator: Initial predator population  
        time_horizon: Simulation time
        filename: Output filename
    """
    set_style()

    # Create uncontrolled population ODE
    pop_ode = PopulationDynamics()
    
    print("Uncontrolled Lotka-Volterra Dynamics")
    print("="*60)
    print(pop_ode)
    print()

    # Critical point (equilibrium)
    critical_point = torch.tensor([100.0, 20.0])
    print(f"Critical point (equilibrium): prey={critical_point[0]:.1f}, predator={critical_point[1]:.1f}")
    
    # Initial state
    initial_state = torch.tensor([initial_prey, initial_predator])
    print(f"Initial state: prey={initial_state[0]:.1f}, predator={initial_state[1]:.1f}")
    print()
    print("Expected behavior: Limit cycle oscillations around critical point")
    print("  - Prey population leads")
    print("  - Predator population lags behind prey")
    print()

    # Plot trajectory
    fig, axes = plot_trajectory(pop_ode, initial_state, time_horizon)

    # Add critical point as reference
    axes[0].axhline(y=critical_point[0], color='red', linestyle='--', 
                   label='critical', alpha=0.5, linewidth=1.5)
    axes[1].axhline(y=critical_point[1], color='red', linestyle='--', 
                   label='critical', alpha=0.5, linewidth=1.5)
    
    axes[0].legend()
    axes[1].legend()
    axes[0].set_title('prey (uncontrolled)')
    axes[1].set_title('predator (uncontrolled)')

    # Remove spines
    for ax in axes:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.tight_layout()
    save_fig(fig, filename)
    print(f"Saved plot to plots/{filename}.pdf")


if __name__ == "__main__":
    fire.Fire(test_lv_uncontrolled)
