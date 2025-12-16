"""Train controller for population dynamics (Lotka-Volterra)."""
import fire
import torch
from rpasim.ode.classic_control.population import PopulationDynamics
from rpasim.env.base import DifferentiableEnv
from rpa_control.controllers import StaticController, ControlledODE
from rpa_control.optimization.gradient import train_ode_parameters, TrainingConfig
from rpa_control.utils.plotting import plot_training_comparison, plot_training_curves
from rpa_control.utils import ExperimentLogger
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
    # Scale-aware regularization
    scale_aware_regularization: bool = False,
    # Three-stage training (improved approach)
    use_three_stage: bool = False,
    n_iterations_stage1: int = 500,
    n_iterations_stage2: int = 300,
    l1_penalty_stage2: float = 0.01,
    n_iterations_stage3: int = 300,
    threshold_value_stage3: float = 1e-3,
    # Deprecated: Old iterative thresholding
    use_thresholding: bool = False,
    threshold_value: float = 1e-3,
    max_threshold_rounds: int = 10,
):
    """
    Train static controller for population dynamics.

    Args:
        n_iterations: Number of training iterations (for single-stage or old iterative LS)
        learning_rate: Learning rate for optimizer
        l1_penalty: L1 regularization coefficient (for single-stage training)
        l2_penalty: L2 regularization coefficient
        time_horizon: Simulation time horizon
        log_interval: Print progress every N iterations
        save_plot: Whether to save training curves
        steady_state_fraction: Fraction of trajectory to skip before computing reward
        controller_order: Polynomial order for controller basis functions
        use_three_stage: Use 3-stage training: normal → L1 reg → thresholding
        n_iterations_stage1: Iterations for stage 1 (normal training)
        n_iterations_stage2: Iterations for stage 2 (L1 regularization)
        l1_penalty_stage2: L1 penalty coefficient for stage 2
        n_iterations_stage3: Iterations for stage 3 (thresholding)
        threshold_value_stage3: Absolute threshold for stage 3 (|θ| < threshold → 0)
        use_thresholding: DEPRECATED - Use iterative LS (replaced by use_three_stage)
        threshold_value: DEPRECATED - Absolute threshold for iterative LS
        max_threshold_rounds: DEPRECATED - Maximum number of threshold-retrain rounds
    """
    set_style()

    # Create experiment logger
    logger = ExperimentLogger(
        log_dir="logs",
        experiment_name="population_static"
    )

    # Start capturing console output
    logger.start_capture()

    print("="*60)
    print(f"Experiment: {logger.experiment_name}")
    print(f"Log directory: {logger.get_experiment_path()}")
    print("="*60)
    print()

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

    # Fixed evaluation initial conditions matching NARROW range [60-140, 10-30]
    # Using the 4 corners + critical point for comprehensive coverage
    eval_initial_states = [
        torch.tensor([60.0, 10.0]),   # min-min corner
        torch.tensor([60.0, 30.0]),   # min-max corner
        torch.tensor([140.0, 10.0]),  # max-min corner
        torch.tensor([140.0, 30.0]),  # max-max corner
        torch.tensor([100.0, 20.0]),  # Critical point (center)
    ]
    print(f"Evaluation: {len(eval_initial_states)} fixed initial conditions")
    print()

    # Create environment
    # Add state_limits to handle instability gracefully during training
    # Use randomized initial conditions for robustness
    # Using [60-140, 10-30] range to maintain prey > predator and avoid extreme oscillations
    env = DifferentiableEnv(
        initial_ode=controlled_ode,
        reward_fn=reward_fn,
        initial_state=initial_state,  # Used only for dimension inference
        initial_state_range=[(60.0, 140.0), (10.0, 30.0)],  # Prey: [60, 140], Predator: [10, 30]
        time_horizon=time_horizon,
        n_reward_steps=100,
        state_limits=(0.0, 200.0),  # Tighter limits for more reasonable penalties
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
        # Scale-aware regularization
        scale_aware_regularization=scale_aware_regularization,
        # Evaluation on fixed initial conditions
        eval_interval=log_interval,  # Evaluate at same frequency as logging
        eval_initial_states=eval_initial_states,
        # Three-stage training
        use_three_stage=use_three_stage,
        n_iterations_stage1=n_iterations_stage1,
        n_iterations_stage2=n_iterations_stage2,
        l1_penalty_stage2=l1_penalty_stage2,
        n_iterations_stage3=n_iterations_stage3,
        threshold_value_stage3=threshold_value_stage3,
        # Old iterative thresholding (deprecated)
        use_thresholding=use_thresholding,
        threshold_value=threshold_value,
        max_threshold_rounds=max_threshold_rounds,
    )

    # Log configuration
    config_dict = {
        'experiment_type': 'population_static',
        'controller_type': 'static',
        'controller_order': controller_order,
        'n_iterations': n_iterations,
        'learning_rate': learning_rate,
        'l1_penalty': l1_penalty,
        'l2_penalty': l2_penalty,
        'time_horizon': time_horizon,
        'steady_state_fraction': steady_state_fraction,
        'scale_aware_regularization': scale_aware_regularization,
        'use_three_stage': use_three_stage,
        'n_iterations_stage1': n_iterations_stage1,
        'n_iterations_stage2': n_iterations_stage2,
        'l1_penalty_stage2': l1_penalty_stage2,
        'n_iterations_stage3': n_iterations_stage3,
        'threshold_value_stage3': threshold_value_stage3,
        'use_thresholding': use_thresholding,
        'threshold_value': threshold_value,
        'max_threshold_rounds': max_threshold_rounds,
        'initial_state': initial_state.tolist(),
        'initial_state_range': [[60.0, 140.0], [10.0, 30.0]],  # Randomized ICs: Prey [60-140], Predator [10-30]
        'eval_initial_states': [ic.tolist() for ic in eval_initial_states],  # Fixed ICs for evaluation
        'critical_point': critical_point.tolist(),
        'control_indices': [1],
    }
    logger.log_config(config_dict)

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
    controller_summary = controlled_ode.get_controller_summary(['prey', 'predator'], ['u'])
    print(controller_summary)
    print()

    # Log results
    logger.log_history(history)

    summary = {
        'final_loss': history['loss'][-1],
        'final_reward': history['reward'][-1],
        'best_reward': history['best_reward'],
        'num_nonzero_params': history['num_nonzero_params'][-1],
        'best_params': history['best_params'].tolist(),
        'total_iterations': len(history['loss']),
    }
    logger.log_results(summary, controller_summary)

    # Create uncontrolled ODE for comparison
    uncontrolled_ode = PopulationDynamics()

    # Plot trajectory comparison and training curves
    if save_plot:
        # Save plots to experiment log directory
        plot_dir = logger.get_experiment_path()

        # Plot trajectories from multiple random initial states sampled from training range
        plot_training_comparison(
            ode_initial=uncontrolled_ode,
            ode_final=controlled_ode,
            initial_state=initial_state,
            time_horizon=time_horizon,
            target_var_idx=None,  # Show target for both variables
            target_value=None,
            initial_state_range=[(60.0, 140.0), (10.0, 30.0)],  # Sample from NARROW range
            n_initial_states=5,
            filename=str(plot_dir / 'trajectories')
        )

        plot_training_curves(
            history=history,
            filename=str(plot_dir / 'training')
        )

    print("Note: Controller parameters have been restored to best (not final iteration)")
    print()

    # Stop logging and print location
    logger.stop_capture()
    print("="*60)
    print(f"Experiment logs saved to: {logger.get_experiment_path()}")
    print("="*60)


if __name__ == "__main__":
    fire.Fire(train_population_controller)
