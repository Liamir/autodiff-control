"""Gradient-based optimization for ODE parameters through differentiable simulation."""
import torch
from typing import Callable, Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, field


@dataclass
class TrainingConfig:
    """Configuration for gradient-based training."""
    n_iterations: int = 1000
    learning_rate: float = 1e-3
    l1_penalty: float = 0.0
    l2_penalty: float = 0.0
    log_interval: int = 10
    verbose: bool = True
    steady_state_fraction: float = 0.0  # Fraction of trajectory to skip before computing reward (0.0 = use all, 0.5 = use second half)
    perturb_param_indices: Optional[List[int]] = None  # Indices of fixed_params to perturb (None = no perturbation)
    perturb_fold_change: float = 2.0  # Perturb params by random factor in [1/fold, fold]


def train_ode_parameters(
    env,
    ode,
    optimizer: Optional[torch.optim.Optimizer] = None,
    config: Optional[TrainingConfig] = None,
    callback: Optional[Callable[[int, Dict[str, Any]], None]] = None,
) -> Dict[str, List[float]]:
    """
    Train ODE parameters using gradient descent through differentiable simulation.

    Loss: -reward + l1_penalty * L1(params) + l2_penalty * L2(params)

    Args:
        env: DifferentiableEnv from rpasim
        ode: ODE instance with differentiable parameters to optimize
        optimizer: PyTorch optimizer (default: Adam with config.learning_rate)
        config: Training configuration
        callback: Optional callback function called after each iteration with (iteration, metrics)

    Returns:
        Dictionary containing training history (losses, rewards, etc.)
    """
    if config is None:
        config = TrainingConfig()

    # Get differentiable parameters
    params = ode.differentiable_params
    assert params is not None and params.numel() > 0, "ODE must have differentiable parameters"

    # Setup optimizer (params must be wrapped in a list for PyTorch optimizers)
    if optimizer is None:
        optimizer = torch.optim.Adam([params], lr=config.learning_rate)

    # Training history
    history = {
        'loss': [],
        'reward': [],
        'l1_penalty': [],
        'l2_penalty': [],
        'num_nonzero_params': [],
    }

    # Track best parameters
    best_reward = float('-inf')
    best_params = params.clone().detach()

    for iteration in range(config.n_iterations):
        optimizer.zero_grad()

        # Reset environment
        obs, info = env.reset()
        current_ode, state = obs

        # Save original fixed params (since current_ode is the env's ODE, not a copy)
        original_fixed_params = current_ode.fixed_params.clone() if hasattr(current_ode, 'fixed_params') and current_ode.fixed_params is not None else None

        # Apply perturbations to fixed parameters if requested
        perturbed_factors = None
        if config.perturb_param_indices is not None and len(config.perturb_param_indices) > 0:
            # Create perturbed copy of fixed params
            perturbed_fixed_params = current_ode.fixed_params.clone()
            perturbed_factors = []

            # Perturb specified parameters (sample uniformly in log-scale)
            for idx in config.perturb_param_indices:
                fold = config.perturb_fold_change
                # Sample uniformly in log-space: log(1/fold) to log(fold)
                log_factor = torch.rand(1).item() * 2 * torch.log(torch.tensor(fold)).item() - torch.log(torch.tensor(fold)).item()
                random_factor = torch.exp(torch.tensor(log_factor)).item()
                perturbed_fixed_params[idx] = perturbed_fixed_params[idx] * random_factor
                perturbed_factors.append(random_factor)

            # Update ODE with perturbed fixed params
            current_ode.fixed_params = perturbed_fixed_params

        # Run simulation
        # In rpasim, we step with (ode, time_duration) as action
        time_horizon = env.time_horizon if hasattr(env, 'time_horizon') else 10.0

        obs, reward, terminated, truncated, info = env.step((current_ode, time_horizon))

        # Restore original fixed params (critical since no deepcopy!)
        if original_fixed_params is not None:
            current_ode.fixed_params = original_fixed_params

        # Apply steady state filtering if requested
        if config.steady_state_fraction > 0:
            times, states, rewards = env.get_trajectory()
            start_idx = int(len(times) * config.steady_state_fraction)
            reward = rewards[start_idx:].sum()

        # Compute loss: -reward + penalties
        loss = -reward

        # Add regularization penalties
        l1_reg = torch.tensor(0.0)
        l2_reg = torch.tensor(0.0)

        if config.l1_penalty > 0:
            l1_reg = torch.abs(params).sum()
            loss = loss + config.l1_penalty * l1_reg

        if config.l2_penalty > 0:
            l2_reg = (params ** 2).sum()
            loss = loss + config.l2_penalty * l2_reg

        # Backpropagation
        loss.backward()
        optimizer.step()

        # Count non-zero parameters (for sparsity tracking)
        num_nonzero = (torch.abs(params) > 1e-6).sum().item()

        # Record history
        history['loss'].append(loss.item())
        history['reward'].append(reward.item() if torch.is_tensor(reward) else reward)
        history['l1_penalty'].append(l1_reg.item() if torch.is_tensor(l1_reg) else l1_reg)
        history['l2_penalty'].append(l2_reg.item() if torch.is_tensor(l2_reg) else l2_reg)
        history['num_nonzero_params'].append(num_nonzero)

        # Update best parameters
        current_reward = reward.item() if torch.is_tensor(reward) else reward
        if current_reward > best_reward:
            best_reward = current_reward
            best_params = params.clone().detach()

        # Logging
        if config.verbose and (iteration % config.log_interval == 0 or iteration == config.n_iterations - 1):
            # Format parameters for display
            params_str = "[" + ", ".join([f"{p.item():.3f}" for p in params]) + "]"
            log_msg = (f"Iter {iteration:4d} | Loss: {loss.item():8.3f} | "
                      f"Reward: {history['reward'][-1]:8.3f} | "
                      f"L1: {history['l1_penalty'][-1]:6.3f} | "
                      f"Non-zero params: {num_nonzero:3d} | "
                      f"Params: {params_str}")

            # Add perturbed factors if perturbations are enabled
            if perturbed_factors is not None:
                factors_str = "[" + ", ".join([f"{f:.2f}x" for f in perturbed_factors]) + "]"
                log_msg += f" | β perturbations: {factors_str}"

            print(log_msg)

        # Callback for custom logging/visualization
        if callback is not None:
            metrics = {
                'iteration': iteration,
                'loss': loss.item(),
                'reward': history['reward'][-1],
                'l1_penalty': history['l1_penalty'][-1],
                'l2_penalty': history['l2_penalty'][-1],
                'num_nonzero_params': num_nonzero,
                'ode': current_ode,
            }
            callback(iteration, metrics)

    # Restore best parameters
    params.data.copy_(best_params)
    history['best_reward'] = best_reward
    history['best_params'] = best_params

    return history
