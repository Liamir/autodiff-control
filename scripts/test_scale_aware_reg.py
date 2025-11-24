"""Test scale-aware regularization for population controller."""
import fire
import torch
from rpasim.ode.classic_control.population import PopulationDynamics
from rpasim.env.base import DifferentiableEnv
from rpa_control.controllers import StaticController, ControlledODE
from rpa_control.optimization.gradient import train_ode_parameters, TrainingConfig
from rpa_control.style import set_style


def reward_fn(state, time=None):
    """Reward function: stabilize critical point (prey=100, predator=20)."""
    critical_point = torch.tensor([100.0, 20.0])
    return -((state - critical_point) ** 2).sum()


def test_scale_aware_regularization(
    n_iterations: int = 300,
    learning_rate: float = 1e-3,
    l1_penalty: float = 2.0,
    scale_aware: bool = True,
    reg_scale_update_interval: int = 0,
    time_horizon: float = 20.0,
    log_interval: int = 50,
):
    """
    Test scale-aware regularization vs standard regularization.

    Args:
        n_iterations: Number of training iterations
        learning_rate: Learning rate for optimizer
        l1_penalty: L1 regularization coefficient
        scale_aware: Whether to use scale-aware regularization
        reg_scale_update_interval: Update regularization scales every N iterations (0 = only compute once at start)
        time_horizon: Simulation time horizon
        log_interval: Print progress every N iterations
    """
    set_style()

    # Create population ODE
    pop_ode = PopulationDynamics()

    print("Testing Scale-Aware Regularization")
    print("="*60)
    print(f"Mode: {'Scale-aware' if scale_aware else 'Standard'} regularization")
    print(f"L1 penalty: {l1_penalty}")
    if scale_aware:
        if reg_scale_update_interval > 0:
            print(f"Scale update interval: every {reg_scale_update_interval} iterations")
        else:
            print(f"Scale update interval: computed once at start")
    print()

    # Create static controller
    controller = StaticController(
        n_state_vars=2,
        n_control_vars=1,
        order=2,
        include_constant=True
    )

    print(f"Controller basis functions: {controller.get_basis_names(['prey', 'predator'])}")
    print()

    # Create controlled ODE
    controlled_ode = ControlledODE(
        base_ode=pop_ode,
        controller=controller,
        control_indices=[1]
    )

    # Initial state
    initial_state = torch.tensor([80.0, 25.0])

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
        log_interval=log_interval,
        verbose=True,
        steady_state_fraction=0.5,
        scale_aware_regularization=scale_aware,
        reg_scale_update_interval=reg_scale_update_interval,
    )

    print("Starting training...")
    print()

    # Train
    history = train_ode_parameters(
        env=env,
        ode=controlled_ode,
        config=config,
    )

    print()
    print("Training complete!")
    print(f"Final reward: {history['reward'][-1]:.3f}")
    print(f"Best reward: {history['best_reward']:.3f}")
    print(f"Non-zero params: {history['num_nonzero_params'][-1]}")
    print()

    # Print controller summary
    print("Trained Controller:")
    print("-"*60)
    print(controlled_ode.get_controller_summary(['prey', 'predator'], ['u']))
    print()


if __name__ == "__main__":
    fire.Fire(test_scale_aware_regularization)
