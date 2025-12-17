"""Train ODE parameters (circuit design) using gradient descent."""
import fire
import torch
from rpasim.ode.ab import AB
from rpasim.env.base import DifferentiableEnv
from rpa_control.optimization.gradient import train_ode_parameters, TrainingConfig
from rpa_control.utils.plotting import plot_training_comparison, plot_training_curves
from rpa_control.utils import ExperimentLogger
from rpa_control.style import set_style


def reward_fn(state, time=None):
    """Reward function: drive B (second variable) to 1.0."""
    # state[1] is B
    target_b = 1.0
    return -(state[1] - target_b) ** 2


def train_ab_ode(
    n_iterations: int = 100,
    learning_rate: float = 1e-2,
    l1_penalty: float = 0.0,
    l2_penalty: float = 0.0,
    time_horizon: float = 20.0,
    log_interval: int = 10,
    save_plot: bool = True,
    steady_state_fraction: float = 0.5,
    perturb_betas: bool = False,
    perturb_fold_change: float = 2.0,
    eval_interval: int = 0,
    n_param_samples_eval: int = 10,
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
        eval_interval: Evaluate every N iterations on fixed ICs (0 = no evaluation, recommended: same as log_interval)
        n_param_samples_eval: Number of parameter samples per IC during evaluation (only used if perturb_betas is True)
    """
    set_style()

    # Create experiment logger
    logger = ExperimentLogger(
        log_dir="logs",
        experiment_name="circuit_ab"
    )

    # Start capturing console output
    logger.start_capture()

    print("="*60)
    print(f"Experiment: {logger.experiment_name}")
    print(f"Log directory: {logger.get_experiment_path()}")
    print("="*60)
    print()

    # Create AB ODE with initial parameters
    # alpha1, alpha2, alpha3 are differentiable; beta1, beta2 are fixed
    initial_alphas_diff = torch.tensor([1.0, -0.25, -0.25], requires_grad=True)  # alpha1, alpha2, alpha3
    initial_fixed = torch.tensor([1.0, 1.0])  # beta1, beta2

    ode = AB(
        differentiable_params=initial_alphas_diff,
        fixed_params=initial_fixed,
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

    # Fixed evaluation initial conditions (if eval_interval > 0)
    # Test from different starting points: corners and center
    eval_initial_states = [
        torch.tensor([0.5, 0.25]),   # Low A, Low B
        torch.tensor([0.5, 0.75]),   # Low A, High B
        torch.tensor([1.5, 0.25]),   # High A, Low B
        torch.tensor([1.5, 0.75]),   # High A, High B
        torch.tensor([1.0, 0.5]),    # Center
    ] if eval_interval > 0 else None

    # Training configuration
    # For AB ODE with fixed_params=[beta1, beta2]: beta1 is at index 0, beta2 is at index 1
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
        eval_interval=eval_interval,
        eval_initial_states=eval_initial_states,
        n_param_samples_eval=n_param_samples_eval,
    )

    # Log configuration
    config_dict = {
        'experiment_type': 'circuit_ab',
        'n_iterations': n_iterations,
        'learning_rate': learning_rate,
        'l1_penalty': l1_penalty,
        'l2_penalty': l2_penalty,
        'time_horizon': time_horizon,
        'steady_state_fraction': steady_state_fraction,
        'perturb_betas': perturb_betas,
        'perturb_fold_change': perturb_fold_change,
        'eval_interval': eval_interval,
        'n_param_samples_eval': n_param_samples_eval,
        'initial_differentiable_params': initial_alphas_diff.detach().tolist(),
        'initial_fixed_params': initial_fixed.tolist(),
        'initial_state': initial_state.tolist(),
        'eval_initial_states': [ic.tolist() for ic in eval_initial_states] if eval_initial_states else None,
        'target_b': 1.0,
    }
    logger.log_config(config_dict)

    print("AB Gene Regulatory Network")
    print("="*60)
    print(f"Equations:")
    print(f"  dA/dt = alpha1*beta1/(1 + B^2) - A")
    print(f"  dB/dt = alpha2*A^2/(alpha3 + A^2)*beta2 - B")
    print()
    print(f"Control objective: drive B to 1.0")
    print()

    print("starting training...")
    print(f"initial differentiable params (alpha1, alpha2, alpha3): [1.000, -0.250, -0.250]")
    print(f"fixed params (beta1, beta2): [1.000, 1.000]")
    if perturb_betas:
        print(f"perturbations enabled: beta params will be randomly perturbed by {perturb_fold_change}x each iteration")
    if eval_interval > 0:
        print(f"evaluation enabled: {len(eval_initial_states)} fixed initial conditions")
        if perturb_betas:
            print(f"  eval uses {n_param_samples_eval} parameter samples per IC (total: {len(eval_initial_states) * n_param_samples_eval} evaluations)")
        else:
            print(f"  eval uses nominal parameters (total: {len(eval_initial_states)} evaluations)")
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
    if eval_interval > 0:
        # Find best eval reward from history
        eval_rewards = [r for r in history['eval_reward'] if r is not None]
        best_eval_reward = max(eval_rewards) if eval_rewards else None
        if best_eval_reward is not None:
            print(f"best eval reward: {best_eval_reward:.3f}")
        print(f"best single-iter reward: {history['best_reward']:.3f}")
    else:
        print(f"best reward: {history['best_reward']:.3f}")
    print(f"non-zero params: {history['num_nonzero_params'][-1]}")

    # Format best parameters
    best_params_str = "[" + ", ".join([f"{p.item():.3f}" for p in history['best_params']]) + "]"
    print(f"best params: {best_params_str}")
    print()
    if eval_interval > 0:
        print("Note: ODE parameters have been restored to best eval (not final iteration)")
    else:
        print("Note: ODE parameters have been restored to best (not final iteration)")
    print()

    # Log results
    logger.log_history(history)

    summary = {
        'final_loss': history['loss'][-1],
        'final_reward': history['reward'][-1],
        'best_reward': history['best_reward'],
        'best_eval_reward': max([r for r in history['eval_reward'] if r is not None]) if eval_interval > 0 and any(r is not None for r in history['eval_reward']) else None,
        'num_nonzero_params': history['num_nonzero_params'][-1],
        'best_params': history['best_params'].tolist(),
        'total_iterations': len(history['loss']),
    }

    # Create parameter summary string
    param_names = ['alpha1', 'alpha2', 'alpha3']
    param_summary_lines = []
    param_summary_lines.append("Trained Parameters (alphas):")
    param_summary_lines.append("-"*60)
    for i, (name, val) in enumerate(zip(param_names, history['best_params'])):
        param_summary_lines.append(f"  {name} = {val.item():.6f}")
    param_summary = '\n'.join(param_summary_lines)

    print(param_summary)
    print()

    logger.log_results(summary, param_summary)

    # Create ODE with initial parameters for comparison
    ode_initial = AB(
        differentiable_params=torch.tensor([1.0, -0.25, -0.25]),
        fixed_params=initial_fixed,
    )

    # Plot trajectory comparison and training curves
    # Note: ode now contains best parameters (restored in train_ode_parameters)
    if save_plot:
        # Save plots to experiment log directory
        plot_dir = logger.get_experiment_path()

        plot_training_comparison(
            ode_initial=ode_initial,
            ode_final=ode,
            initial_state=initial_state,
            time_horizon=time_horizon,
            target_var_idx=1,  # B is the second variable
            target_value=1.0,
            perturb_indices=[0, 1],  # Always show perturbations to beta1, beta2
            perturb_fold_change=perturb_fold_change,
            filename=str(plot_dir / 'trajectories')
        )

        plot_training_curves(
            history=history,
            filename=str(plot_dir / 'training')
        )

    # Stop logging and print location
    logger.stop_capture()
    print("="*60)
    print(f"Experiment logs saved to: {logger.get_experiment_path()}")
    print("="*60)


if __name__ == "__main__":
    fire.Fire(train_ab_ode)
