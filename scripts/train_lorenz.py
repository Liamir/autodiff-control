"""Train controller for Lorenz system."""
import fire
import torch
from rpasim.ode.classic_control import Lorenz
from rpasim.env.base import DifferentiableEnv
from rpa_control.controllers import StaticController, ControlledODE
from rpa_control.optimization.gradient import train_ode_parameters, TrainingConfig
from rpa_control.utils.plotting import plot_training_comparison, plot_training_curves
from rpa_control.style import set_style


def reward_fn(state, time=None):
    """Reward function: stabilize at fixed point (√72, √72, 27)."""
    # Fixed point for Lorenz system with ρ=28, σ=10, β=8/3
    sqrt_72 = (72.0) ** 0.5  # ≈ 8.485
    target = torch.tensor([sqrt_72, sqrt_72, 27.0])
    # Negative squared error
    return -((state - target) ** 2).sum()


def train_lorenz_controller(
    n_iterations: int = 500,
    learning_rate: float = 1e-3,
    l1_penalty: float = 0.0,
    l2_penalty: float = 0.0,
    time_horizon: float = 10.0,
    log_interval: int = 50,
    save_plot: bool = True,
    steady_state_fraction: float = 0.5,
    controller_order: int = 1,
    # Scale-aware regularization
    scale_aware_regularization: bool = False,
    # Learning rate warmup
    warmup_iterations: int = 0,
    warmup_start_factor: float = 0.01,
    # Gradient clipping
    gradient_clip_norm: float = 0.0,
    # Three-stage training
    use_three_stage: bool = False,
    n_iterations_stage1: int = 500,
    n_iterations_stage2: int = 300,
    l1_penalty_stage2: float = 0.01,
    n_iterations_stage3: int = 300,
    threshold_value_stage3: float = 1e-3,
):
    """
    Train static controller for Lorenz system.

    Args:
        n_iterations: Number of training iterations
        learning_rate: Learning rate for optimizer
        l1_penalty: L1 regularization coefficient
        l2_penalty: L2 regularization coefficient
        time_horizon: Simulation time horizon
        log_interval: Print progress every N iterations
        save_plot: Whether to save training curves
        steady_state_fraction: Fraction of trajectory to skip before computing reward
        controller_order: Polynomial order for controller basis functions
        scale_aware_regularization: Use basis normalization
        warmup_iterations: Number of iterations for LR warmup (0 = no warmup)
        warmup_start_factor: Starting LR = learning_rate * warmup_start_factor
        gradient_clip_norm: Max gradient norm for clipping (0 = no clipping)
        use_three_stage: Use 3-stage training: normal → L1 reg → thresholding
        n_iterations_stage1: Iterations for stage 1
        n_iterations_stage2: Iterations for stage 2
        l1_penalty_stage2: L1 penalty for stage 2
        n_iterations_stage3: Iterations for stage 3
        threshold_value_stage3: Threshold for stage 3
    """
    set_style()

    # Create Lorenz ODE
    lorenz_ode = Lorenz()
    
    print("Lorenz System (Chaotic)")
    print("="*60)
    print(lorenz_ode)
    print()

    # Control objective: stabilize at fixed point
    sqrt_72 = (72.0) ** 0.5
    target = torch.tensor([sqrt_72, sqrt_72, 27.0])
    print(f"Control objective: stabilize at fixed point (√72, √72, 27)")
    print(f"  target = [{sqrt_72:.3f}, {sqrt_72:.3f}, 27.0]")
    print()

    # Create static controller
    # 3 state vars (x1, x2, x3), 1 control output (affects only first state)
    controller = StaticController(
        n_state_vars=3,
        n_control_vars=1,
        order=controller_order,
        include_constant=True
    )

    print(f"Controller: Static, order {controller_order}")
    print(f"  Parameters: {controller.params.numel()}")
    print(f"  Basis functions: {controller.get_basis_names(['x1', 'x2', 'x3'])}")
    print()

    # Create controlled ODE
    controlled_ode = ControlledODE(
        base_ode=lorenz_ode,
        controller=controller,
        control_indices=[0]  # Control affects only first state (x1)
    )

    # Initial state (away from origin)
    initial_state = torch.tensor([1.0, 1.0, 1.0])
    print(f"Initial state: {initial_state.tolist()}")
    print()

    # Create environment
    env = DifferentiableEnv(
        initial_ode=controlled_ode,
        reward_fn=reward_fn,
        initial_state=initial_state,
        time_horizon=time_horizon,
        n_reward_steps=100,
        state_limits=(-50.0, 50.0),  # Prevent explosion during training
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
        scale_aware_regularization=scale_aware_regularization,
        warmup_iterations=warmup_iterations,
        warmup_start_factor=warmup_start_factor,
        gradient_clip_norm=gradient_clip_norm,
        use_three_stage=use_three_stage,
        n_iterations_stage1=n_iterations_stage1,
        n_iterations_stage2=n_iterations_stage2,
        l1_penalty_stage2=l1_penalty_stage2,
        n_iterations_stage3=n_iterations_stage3,
        threshold_value_stage3=threshold_value_stage3,
    )

    print("Starting training...")
    print(f"Time horizon: {time_horizon}")
    print(f"Learning rate: {learning_rate}")
    if warmup_iterations > 0:
        print(f"Learning rate warmup: {warmup_iterations} iterations ({warmup_start_factor * learning_rate:.2e} → {learning_rate:.2e})")
    if gradient_clip_norm > 0:
        print(f"Gradient clipping: max_norm={gradient_clip_norm}")
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
    print(controlled_ode.get_controller_summary(['x1', 'x2', 'x3'], ['u']))
    print()

    # Create uncontrolled ODE for comparison
    uncontrolled_ode = Lorenz()

    # Plot trajectory comparison and training curves
    if save_plot:
        plot_training_comparison(
            ode_initial=uncontrolled_ode,
            ode_final=controlled_ode,
            initial_state=initial_state,
            time_horizon=time_horizon,
            target_var_idx=None,
            target_value=None,
            filename='lorenz_trajectories'
        )

        plot_training_curves(
            history=history,
            filename='lorenz_training'
        )

    print("Note: Controller parameters have been restored to best (not final iteration)")


if __name__ == "__main__":
    fire.Fire(train_lorenz_controller)
