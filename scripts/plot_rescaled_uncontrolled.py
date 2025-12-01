"""Plot uncontrolled trajectory of rescaled population dynamics."""
import fire
import torch
import matplotlib.pyplot as plt
from rpasim.plot.ode import plot_trajectory
from rpa_control.paths import save_fig
from rpa_control.style import set_style


class RescaledPopulationDynamics:
    """Lotka-Volterra with prey measured in millions (10^-6 scale)."""

    def __init__(self):
        self.name = "rescaled_population"
        self.variable_names = ["prey_millions", "predator"]

        # Parameters adjusted for rescaled prey
        self.fixed_params = torch.tensor([
            0.5,      # a: prey growth rate
            0.005,    # b: predation rate
            1.0,      # c: predator death rate
            1e4,      # d: predator growth from prey (10^6 * 0.01 = 10000)
        ])

    def __call__(self, t, state):
        prey, predator = state[0], state[1]
        a, b, c, d = self.fixed_params[0], self.fixed_params[1], self.fixed_params[2], self.fixed_params[3]

        dprey_dt = a * prey - b * prey * predator
        dpredator_dt = -c * predator + d * prey * predator

        return torch.stack([dprey_dt, dpredator_dt])


def plot_rescaled_uncontrolled(
    initial_prey: float = 0.00008,
    initial_predator: float = 80.0,
    time_horizon: float = 100.0,
    filename: str = 'rescaled_uncontrolled_oscillation'
):
    """Plot uncontrolled rescaled dynamics showing extreme scale mismatch.

    Args:
        initial_prey: Initial prey population (in millions)
        initial_predator: Initial predator population
        time_horizon: Simulation time
        filename: Output filename
    """
    set_style()

    # Create uncontrolled rescaled ODE
    ode = RescaledPopulationDynamics()

    print("Rescaled Population Dynamics (Uncontrolled)")
    print("="*60)
    a, b, c, d = ode.fixed_params
    print(f"dP/dt = {a:.3f}*P - {b:.3f}*P*D")
    print(f"dD/dt = -{c:.3f}*D + {d:.1f}*P*D")
    print(f"Critical point: P={c/d:.6f}, D={a/b:.1f}")
    print()

    # Initial state
    initial_state = torch.tensor([initial_prey, initial_predator])
    print(f"Initial state: prey={initial_state[0]:.6f} million, predator={initial_state[1]:.1f}")
    print(f"Critical point: prey={c/d:.6f} million, predator={a/b:.1f}")
    print()
    print("Expected behavior: Oscillations with extreme scale mismatch")
    print(f"  - Prey scale: ~1e-4 (measured in millions)")
    print(f"  - Predator scale: ~100")
    print(f"  - Scale ratio: ~1e6")
    print()

    # Plot trajectory
    fig, axes = plot_trajectory(ode, initial_state, time_horizon)

    # Add critical point as reference
    critical_point = torch.tensor([c/d, a/b])
    axes[0].axhline(y=critical_point[0], color='red', linestyle='--',
                   label='critical', alpha=0.5, linewidth=1.5)
    axes[1].axhline(y=critical_point[1], color='red', linestyle='--',
                   label='critical', alpha=0.5, linewidth=1.5)

    axes[0].legend()
    axes[1].legend()
    axes[0].set_title('prey (millions) - uncontrolled')
    axes[1].set_title('predator - uncontrolled')

    # Use scientific notation for prey axis
    axes[0].ticklabel_format(style='scientific', axis='y', scilimits=(0,0))

    # Remove spines
    for ax in axes:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.tight_layout()
    save_fig(fig, filename)
    print(f"Saved plot to plots/{filename}.pdf")


if __name__ == "__main__":
    fire.Fire(plot_rescaled_uncontrolled)
