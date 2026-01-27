"""Evaluate a trained controller on an environment."""
import fire
import torch
import matplotlib.pyplot as plt
from rpasim.ode.classic_control.population import PopulationDynamics
from rpasim.plot.ode import plot_trajectory
from rpa_control.controllers import StaticController, ControlledODE
from rpa_control.paths import save_fig
from rpa_control.style import set_style


def eval_population_controller(
    p0: float = 0.0,
    p1: float = 0.0,
    p2: float = 0.0,
    p3: float = 0.0,
    p4: float = 0.0,
    p5: float = 0.0,
    controller_order: int = 1,
    time_horizon: float = 100.0,
    initial_prey: float = 80.0,
    initial_predator: float = 25.0,
    filename: str = 'controller_eval'
):
    """Evaluate a trained controller on population dynamics.

    Args:
        p0, p1, p2, p3, p4, p5: Controller parameters
        controller_order: Polynomial order of the controller
        time_horizon: Simulation time horizon
        initial_prey: Initial prey population
        initial_predator: Initial predator population
        filename: Output filename for the plot
    """
    set_style()

    # Collect non-zero parameters based on controller order
    # Order 1: constant, prey, predator (3 params)
    # Order 2: constant, prey, predator, prey^2, prey*predator, predator^2 (6 params)
    if controller_order == 1:
        param_values = [p0, p1, p2]
    else:  # order 2
        param_values = [p0, p1, p2, p3, p4, p5]

    # Shape should be (n_control_vars, n_basis) = (1, n_basis)
    param_tensor = torch.tensor(param_values).reshape(1, -1)

    # Create population ODE
    pop_ode = PopulationDynamics()

    print("Population Dynamics Controller Evaluation")
    print("="*60)
    print(pop_ode)
    print()

    # Create controller with specified parameters
    controller = StaticController(
        n_state_vars=2,
        n_control_vars=1,
        order=controller_order,
        include_constant=True
    )
    controller.params.data = param_tensor

    print(f"Controller: Static, order {controller_order}")
    print(f"Parameters: {param_values}")
    print(f"Basis functions: {controller.get_basis_names(['prey', 'predator'])}")
    print()

    # Create controlled ODE
    controlled_ode = ControlledODE(
        base_ode=pop_ode,
        controller=controller,
        control_indices=[1]  # Control affects predator
    )

    # Print controller equation
    print("Trained Controller:")
    print("-"*60)
    try:
        print(controlled_ode.get_controller_summary(['prey', 'predator'], ['u']))
    except Exception as e:
        # Fallback: manually print equation
        basis_names = controller.get_basis_names(['prey', 'predator'])
        terms = []
        for i, (param, basis) in enumerate(zip(param_values, basis_names)):
            if param != 0:
                terms.append(f"{param:.3f}*{basis}" if basis != '1' else f"{param:.3f}")
        print(f"u = {' + '.join(terms).replace('+ -', '- ')}")
    print()

    # Initial state
    initial_state = torch.tensor([initial_prey, initial_predator])
    print(f"Initial state: prey={initial_prey:.1f}, predator={initial_predator:.1f}")
    print(f"Time horizon: {time_horizon}")
    print()

    # Plot trajectory
    print("Simulating trajectory...")
    fig, axes = plot_trajectory(controlled_ode, initial_state, time_horizon)

    # Add target lines (critical point)
    target_prey = 100.0
    target_predator = 20.0
    axes[0].axhline(y=target_prey, color='red', linestyle='--', label='target', alpha=0.5)
    axes[1].axhline(y=target_predator, color='red', linestyle='--', label='target', alpha=0.5)

    for ax in axes:
        ax.legend()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    # Update titles
    axes[0].set_title(f'prey (controlled, order={controller_order})')
    axes[1].set_title(f'predator (controlled, order={controller_order})')

    plt.tight_layout()
    save_fig(fig, filename)
    print(f"Saved plot to plots/{filename}.pdf")

    return fig, axes


if __name__ == "__main__":
    fire.Fire(eval_population_controller)
