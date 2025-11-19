"""Train ODE parameters (circuit design) using gradient descent."""
import fire
import torch
from rpasim.ode.ab import AB
from rpasim.env.base import DifferentiableEnv
from rpa_control.optimization.gradient import train_ode_parameters, TrainingConfig
from rpa_control.utils.plotting import plot_training_comparison, plot_training_curves
from rpa_control.style import set_style


def reward_fn(state):
    """Reward function: drive B (second variable) to 1.0."""
    # state[1] is B
    target_b = 1.0
    return -(state[1] - target_b) ** 2


def train_ab_ode(
    n_iterations: int = 100,
    learning_rate: float = 1e-2,
    l1_penalty: float = 0.0,
    l2_penalty: float = 0.0,
    time_horizon: float = 10.0,
    log_interval: int = 10,
    save_plot: bool = True,
    steady_state_fraction: float = 0.5,
    perturb_betas: bool = False,
    perturb_fold_change: float = 2.0,
):
    """
    Train AB ODE parameters.

    Args:
        n_iterations: Number of training iterations
        learning_rate: Learning rate for optimizer
        l1_penalty: L1 regularization coefficient (for sparsity)
        l2_penalty: L2 regularization coefficient
        time_horizon: Simulation time horizon
        log_interval: Print progress every N iterations
        save_plot: Whether to save training curves
        steady_state_fraction: Fraction of trajectory to skip before computing reward (0.0=use all, 0.5=use second half)
        perturb_betas: Whether to perturb beta parameters during training (for robustness)
        perturb_fold_change: Fold change for perturbations (params multiplied by random factor in [1/fold, fold])
    """
    set_style()

    # Create AB ODE with initial parameters
    # alphas are differentiable, betas are fixed
    initial_alphas = torch.tensor([1.0, 0.25, 0.25], requires_grad=True)
    initial_betas = torch.tensor([1.0, 1.0])

    ode = AB(
        differentiable_params=initial_alphas,
        fixed_params=initial_betas,
    )

    # Create environment
    initial_state = torch.tensor([1.0, 0.5])
    env = DifferentiableEnv(
        initial_ode=ode,
        reward_fn=reward_fn,
        initial_state=initial_state,
        time_horizon=time_horizon,
        n_reward_steps=100,  # Number of steps for reward computation grid
    )

    # Training configuration
    # For AB ODE: beta1 is at index 0, beta2 is at index 1
    perturb_indices = [0, 1] if perturb_betas else None

    config = TrainingConfig(
        n_iterations=n_iterations,
        learning_rate=learning_rate,
        l1_penalty=l1_penalty,
        l2_penalty=l2_penalty,
        log_interval=log_interval,
        verbose=True,
        steady_state_fraction=steady_state_fraction,
        perturb_param_indices=perturb_indices,
        perturb_fold_change=perturb_fold_change,
    )

    print("starting training...")
    print(f"initial alphas: [1.000, 0.250, 0.250]")
    if perturb_betas:
        print(f"perturbations enabled: beta params will be randomly perturbed by {perturb_fold_change}x each iteration")
    print()

    # Train
    history = train_ode_parameters(
        env=env,
        ode=ode,
        config=config,
    )

    print()
    print("training complete!")
    print(f"final loss: {history['loss'][-1]:.3f}")
    print(f"final reward: {history['reward'][-1]:.3f}")
    print(f"non-zero params: {history['num_nonzero_params'][-1]}")

    # Create ODE with initial parameters for comparison
    ode_initial = AB(
        differentiable_params=torch.tensor([1.0, 0.25, 0.25]),
        fixed_params=initial_betas,
    )

    # Plot trajectory comparison and training curves
    if save_plot:
        plot_training_comparison(
            ode_initial=ode_initial,
            ode_final=ode,
            initial_state=initial_state,
            time_horizon=time_horizon,
            target_var_idx=1,  # B is the second variable
            target_value=1.0,
            filename='circuit_trajectories'
        )

        plot_training_curves(
            history=history,
            filename='circuit_training'
        )


if __name__ == "__main__":
    fire.Fire(train_ab_ode)
