"""Train controller for population dynamics (Lotka-Volterra)."""
import fire
import torch
from rpasim.ode.classic_control.population import PopulationDynamics
from rpasim.env.base import DifferentiableEnv
from rpa_control.controllers import StaticController, ControlledODE
from rpa_control.optimization.gradient import train_ode_parameters, TrainingConfig
from rpa_control.utils.plotting import plot_training_comparison, plot_training_curves
from rpa_control.style import set_style


def reward_fn(state, time=None):
    """Reward function: stabilize critical point (prey=100, predator=20)."""
    critical_point = torch.tensor([100.0, 20.0])
    # Negative squared error
    return -((state - critical_point) ** 2).sum()


def train_population_controller(
    n_iterations: int = 500,
    learning_rate: float = 1e-3,
    l1_penalty: float = 0.0,
    l2_penalty: float = 0.0,
    time_horizon: float = 20.0,
    log_interval: int = 50,
    save_plot: bool = True,
    steady_state_fraction: float = 0.5,
    controller_order: int = 2,
):
    """
    Train static controller for population dynamics.

    Args:
        n_iterations: Number of training iterations
        learning_rate: Learning rate for optimizer
        l1_penalty: L1 regularization coefficient (for sparsity)
        l2_penalty: L2 regularization coefficient
        time_horizon: Simulation time horizon
        log_interval: Print progress every N iterations
        save_plot: Whether to save training curves
        steady_state_fraction: Fraction of trajectory to skip before computing reward
        controller_order: Polynomial order for controller basis functions
    """
    set_style()

    # Create population ODE
    pop_ode = PopulationDynamics()
    
    print("Population Dynamics (Lotka-Volterra)")
    print("="*60)
    print(pop_ode)
    print()

    # Critical point from paper: (c/d, a/b) = (100, 20)
    critical_point = torch.tensor([100.0, 20.0])
    print(f"Control objective: stabilize critical point")
    print(f"  prey = {critical_point[0]:.1f}")
    print(f"  predator = {critical_point[1]:.1f}")
    print()

    # Create static controller
    # 2 state vars (prey, predator), 1 control output (affects predator only)
    controller = StaticController(
        n_state_vars=2,
        n_control_vars=1,
        order=controller_order,
        include_constant=True
    )

    print(f"Controller: Static, order {controller_order}")
    print(f"  Parameters: {controller.params.numel()}")
    print(f"  Basis functions: {controller.get_basis_names(['prey', 'predator'])}")
    print()

    # Create controlled ODE
    controlled_ode = ControlledODE(
        base_ode=pop_ode,
        controller=controller,
        control_indices=[1]  # Control affects predator (index 1)
    )

    # Initial state (away from critical point)
    initial_state = torch.tensor([80.0, 25.0])
    print(f"Initial state: prey={initial_state[0]:.1f}, predator={initial_state[1]:.1f}")
    print()

    # Create environment
    env = DifferentiableEnv(
        initial_ode=controlled_ode,
        reward_fn=reward_fn,
        initial_state=initial_state,
        time_horizon=time_horizon,
        n_reward_steps=100,
    )

    # Training configuration
    config = TrainingConfig(
        n_iterations=n_iterations,
        learning_rate=learning_rate,
        l1_penalty=l1_penalty,
        l2_penalty=l2_penalty,
        log_interval=log_interval,
        verbose=True,
        steady_state_fraction=steady_state_fraction,
    )

    print("Starting training...")
    print(f"Time horizon: {time_horizon}")
    print(f"Learning rate: {learning_rate}")
    print(f"Iterations: {n_iterations}")
    print()

    # Train
    history = train_ode_parameters(
        env=env,
        ode=controlled_ode,
        config=config,
    )

    print()
    print("Training complete!")
    print(f"Final loss: {history['loss'][-1]:.3f}")
    print(f"Final reward: {history['reward'][-1]:.3f}")
    print(f"Best reward: {history['best_reward']:.3f}")
    print(f"Non-zero params: {history['num_nonzero_params'][-1]}")
    
    # Format best parameters (flatten if multi-dimensional)
    best_params_flat = history['best_params'].flatten()
    best_params_str = "[" + ", ".join([f"{p.item():.3f}" for p in best_params_flat]) + "]"
    print(f"Best params: {best_params_str}")
    print()
    
    # Print controller summary
    print("Trained Controller:")
    print("-"*60)
    print(controlled_ode.get_controller_summary(['prey', 'predator'], ['u']))
    print()

    # Create uncontrolled ODE for comparison
    uncontrolled_ode = PopulationDynamics()

    # Plot trajectory comparison and training curves
    if save_plot:
        plot_training_comparison(
            ode_initial=uncontrolled_ode,
            ode_final=controlled_ode,
            initial_state=initial_state,
            time_horizon=time_horizon,
            target_var_idx=None,  # Show target for both variables
            target_value=None,
            filename='population_trajectories'
        )

        plot_training_curves(
            history=history,
            filename='population_training'
        )

    print("Note: Controller parameters have been restored to best (not final iteration)")


if __name__ == "__main__":
    fire.Fire(train_population_controller)
