"""Train dynamic controller for population dynamics (Lotka-Volterra)."""
import fire
import torch
from rpasim.ode.classic_control.population import PopulationDynamics
from rpasim.env.base import DifferentiableEnv
from rpa_control.controllers import DynamicController, ControlledODE
from rpa_control.optimization.gradient import train_ode_parameters, TrainingConfig
from rpa_control.utils.plotting import plot_training_comparison, plot_training_curves
from rpa_control.utils import ExperimentLogger
from rpa_control.style import set_style


def reward_fn(state, time=None):
    """Reward function: stabilize critical point (prey=100, predator=20).

    Note: For dynamic controllers, state includes controller states.
    We only penalize deviation in base states (prey, predator).
    """
    critical_point = torch.tensor([100.0, 20.0])
    # Extract base state (first 2 elements: prey, predator)
    base_state = state[:2]
    # Negative squared error
    return -((base_state - critical_point) ** 2).sum()


def train_population_dynamic_controller(
    n_iterations: int = 1000,
    learning_rate: float = 1e-2,
    l1_penalty: float = 0.0,
    l2_penalty: float = 0.0,
    time_horizon: float = 20.0,
    log_interval: int = 100,
    save_plot: bool = True,
    steady_state_fraction: float = 0.5,
    n_controller_states: int = 1,
    observing_order: int = 1,
    actuating_order: int = 1,
    # Scale-aware regularization
    scale_aware_regularization: bool = False,
    # Three-stage training
    use_three_stage: bool = False,
    n_iterations_stage1: int = 500,
    n_iterations_stage2: int = 300,
    l1_penalty_stage2: float = 0.01,
    n_iterations_stage3: int = 300,
    threshold_value_stage3: float = 1e-3,
):
    """
    Train dynamic controller for population dynamics.

    Args:
        n_iterations: Number of training iterations
        learning_rate: Learning rate for optimizer
        l1_penalty: L1 regularization coefficient (for sparsity)
        l2_penalty: L2 regularization coefficient
        time_horizon: Simulation time horizon
        log_interval: Print progress every N iterations
        save_plot: Whether to save training curves
        steady_state_fraction: Fraction of trajectory to skip before computing reward
        n_controller_states: Number of internal controller state variables
        observing_order: Polynomial order for observing basis (dC/dt)
        actuating_order: Polynomial order for actuating basis (u)
        scale_aware_regularization: Use basis normalization
        use_three_stage: Use 3-stage training: normal → L1 reg → thresholding
        n_iterations_stage1: Iterations for stage 1
        n_iterations_stage2: Iterations for stage 2
        l1_penalty_stage2: L1 penalty for stage 2
        n_iterations_stage3: Iterations for stage 3
        threshold_value_stage3: Threshold for stage 3
    """
    set_style()

    # Create experiment logger
    logger = ExperimentLogger(
        log_dir="logs",
        experiment_name="population_dynamic"
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

    # Create dynamic controller
    # n_state_vars=2 (prey, predator)
    # n_controller_states: internal state dimension
    # n_control_vars=1 (control affects predator only)
    controller = DynamicController(
        n_state_vars=2,
        n_controller_states=n_controller_states,
        n_control_vars=1,
        observing_order=observing_order,
        actuating_order=actuating_order,
        include_constant=True
    )

    print(f"Controller: Dynamic")
    print(f"  Internal states: {n_controller_states}")
    print(f"  Observing order: {observing_order}")
    print(f"  Actuating order: {actuating_order}")
    print(f"  Total parameters: {controller.observing_params.numel() + controller.actuating_params.numel()}")
    print(f"    Observing params: {controller.observing_params.numel()} (for dC/dt)")
    print(f"    Actuating params: {controller.actuating_params.numel()} (for u)")
    print()

    # Create controlled ODE
    controlled_ode = ControlledODE(
        base_ode=pop_ode,
        controller=controller,
        control_indices=[1]  # Control affects predator (index 1)
    )

    # Initial state (away from critical point)
    # For dynamic controller, augment with controller state initialized to zero
    base_initial_state = torch.tensor([80.0, 25.0])
    controller_initial_state = torch.zeros(n_controller_states)
    initial_state = torch.cat([base_initial_state, controller_initial_state])

    print(f"Initial state:")
    print(f"  Base (prey, predator): {base_initial_state.tolist()}")
    print(f"  Controller: {controller_initial_state.tolist()}")
    print()

    # Fixed evaluation initial conditions for base state matching NARROW range [60-140, 10-30]
    # Using the 4 corners + critical point for comprehensive coverage
    # For dynamic controllers, these will be augmented with controller states during evaluation
    eval_base_initial_states = [
        torch.tensor([60.0, 10.0]),   # min-min corner
        torch.tensor([60.0, 30.0]),   # min-max corner
        torch.tensor([140.0, 10.0]),  # max-min corner
        torch.tensor([140.0, 30.0]),  # max-max corner
        torch.tensor([100.0, 20.0]),  # Critical point (center)
    ]
    # Augment with controller states for evaluation
    eval_initial_states = [
        torch.cat([base_ic, torch.zeros(n_controller_states)])
        for base_ic in eval_base_initial_states
    ]
    print(f"Evaluation: {len(eval_initial_states)} fixed initial conditions")
    print()

    # Create environment
    # Add state_limits to handle instability gracefully during training
    # Use randomized initial conditions for robustness
    # Using [60-140, 10-30] range for base state to maintain prey > predator and avoid extreme oscillations
    # Controller states are always initialized to zero
    env = DifferentiableEnv(
        initial_ode=controlled_ode,
        reward_fn=reward_fn,
        initial_state=initial_state,  # Used only for dimension inference
        initial_state_range=[(60.0, 140.0), (10.0, 30.0)] + [(0.0, 0.0)] * n_controller_states,  # Prey, Predator, Controller states
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
        scale_aware_regularization=scale_aware_regularization,
        use_three_stage=use_three_stage,
        n_iterations_stage1=n_iterations_stage1,
        n_iterations_stage2=n_iterations_stage2,
        l1_penalty_stage2=l1_penalty_stage2,
        n_iterations_stage3=n_iterations_stage3,
        threshold_value_stage3=threshold_value_stage3,
    )

    # Log configuration
    config_dict = {
        'experiment_type': 'population_dynamic',
        'controller_type': 'dynamic',
        'n_controller_states': n_controller_states,
        'observing_order': observing_order,
        'actuating_order': actuating_order,
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
        'initial_state_base': base_initial_state.tolist(),
        'initial_state_controller': controller_initial_state.tolist(),
        'initial_state_range': [[60.0, 140.0], [10.0, 30.0]] + [[0.0, 0.0]] * n_controller_states,  # Prey, Predator, Controller states
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

    # Format best parameters
    best_params_flat = history['best_params'].flatten()
    best_params_str = "[" + ", ".join([f"{p.item():.3f}" for p in best_params_flat]) + "]"
    print(f"Best params: {best_params_str}")
    print()

    # Print controller summary
    print("Trained Controller:")
    print("-"*60)
    # Note: get_controller_summary expects only base state variable names
    # (it will add controller state names internally)
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

        # Pass base initial state; plotting function will handle dynamic controllers automatically
        # Plot trajectories from multiple random initial states sampled from training range
        plot_training_comparison(
            ode_initial=uncontrolled_ode,
            ode_final=controlled_ode,
            initial_state=base_initial_state,
            time_horizon=time_horizon,
            target_var_idx=None,
            target_value=None,
            initial_state_range=[(60.0, 140.0), (10.0, 30.0)],  # Sample from NARROW range (base state only)
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
    fire.Fire(train_population_dynamic_controller)
