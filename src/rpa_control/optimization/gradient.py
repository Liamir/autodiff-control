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
    scale_aware_regularization: bool = False  # Use scale-aware regularization based on basis function magnitudes
    reg_scale_update_interval: int = 0  # Update regularization scales every N iterations (0 = only compute once at start)


def compute_basis_scales(ode, env) -> torch.Tensor:
    """Compute scale factors for basis functions based on trajectory sampling.

    This enables scale-aware regularization where parameters are penalized
    proportional to the typical magnitude of their basis functions.

    For example, if variable A ~ 1 and B ~ 100, then:
    - Parameter multiplying A should have weight ~ 1
    - Parameter multiplying B should have weight ~ 100
    - Parameter multiplying B² should have weight ~ 10,000

    Args:
        ode: ODE with controller (ControlledODE instance)
        env: Environment for simulation

    Returns:
        Tensor of shape (n_params,) with RMS scale for each basis function
    """
    if not hasattr(ode, 'controller'):
        # No controller, return ones (no scaling)
        if hasattr(ode, 'differentiable_params'):
            return torch.ones(ode.differentiable_params.numel())
        return torch.ones(1)

    # Import basis function generator
    from ..controllers.basis import polynomial_basis

    # Reset environment and simulate trajectory
    obs, info = env.reset()
    current_ode, state = obs

    # Run simulation to collect states
    time_horizon = env.time_horizon if hasattr(env, 'time_horizon') else 10.0
    obs, reward, terminated, truncated, info = env.step((current_ode, time_horizon))

    # Get trajectory states
    times, states, _ = env.get_trajectory()

    # Compute basis function values for each state
    is_dynamic = hasattr(ode, 'is_dynamic') and ode.is_dynamic

    if is_dynamic:
        # Dynamic controller: separate scales for observed and manipulated params
        observed_basis_values = []
        manipulated_basis_values = []

        for state_vec in states:
            # Split state
            base_state = state_vec[:ode.base_state_dim]
            controller_state = state_vec[ode.base_state_dim:]

            # Observed basis: Φ(X, C)
            augmented_state = torch.cat([base_state, controller_state])
            observed_basis = polynomial_basis(
                augmented_state,
                ode.controller.observed_order,
                ode.controller.include_constant
            )
            observed_basis_values.append(observed_basis.detach())

            # Manipulated basis: Ψ(C)
            manipulated_basis = polynomial_basis(
                controller_state,
                ode.controller.manipulated_order,
                ode.controller.include_constant
            )
            manipulated_basis_values.append(manipulated_basis.detach())

        # Stack and compute RMS for each basis function
        observed_basis_values = torch.stack(observed_basis_values)  # (n_timesteps, n_basis_obs)
        manipulated_basis_values = torch.stack(manipulated_basis_values)  # (n_timesteps, n_basis_man)

        observed_scales = torch.sqrt((observed_basis_values ** 2).mean(dim=0))
        manipulated_scales = torch.sqrt((manipulated_basis_values ** 2).mean(dim=0))

        # Flatten and concatenate (matching parameter order in ControlledODE)
        # Order: [observed_params (n_controller_states, n_basis_obs), manipulated_params (n_control_vars, n_basis_man)]
        n_controller_states = ode.controller.n_controller_states
        n_control_vars = ode.controller.n_control_vars

        # Repeat scales for each output dimension
        observed_scales_flat = observed_scales.repeat(n_controller_states)
        manipulated_scales_flat = manipulated_scales.repeat(n_control_vars)

        scales = torch.cat([observed_scales_flat, manipulated_scales_flat])

    else:
        # Static controller: Φ(X) only
        basis_values = []

        for state_vec in states:
            # Extract base state
            base_state = ode.extract_base_state(state_vec) if hasattr(ode, 'extract_base_state') else state_vec

            # Compute basis
            basis = polynomial_basis(
                base_state,
                ode.controller.order,
                ode.controller.include_constant
            )
            basis_values.append(basis.detach())

        # Stack and compute RMS
        basis_values = torch.stack(basis_values)  # (n_timesteps, n_basis)
        basis_scales = torch.sqrt((basis_values ** 2).mean(dim=0))

        # Repeat for each control output
        n_control_vars = ode.controller.n_control_vars
        scales = basis_scales.repeat(n_control_vars)

    # Add small epsilon to avoid division by zero
    scales = scales + 1e-8

    return scales


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
        'control_max': [],      # Max absolute control value
        'control_mean': [],     # Mean absolute control value
        'control_rms': [],      # RMS control value
    }

    # Track best parameters
    best_reward = float('-inf')
    best_params = params.clone().detach()

    # Compute basis scales for scale-aware regularization
    basis_scales = None
    if config.scale_aware_regularization and (config.l1_penalty > 0 or config.l2_penalty > 0):
        if config.verbose:
            print("Computing basis function scales for scale-aware regularization...")
        basis_scales = compute_basis_scales(ode, env)
        if config.verbose:
            print(f"Basis scales (RMS): min={basis_scales.min().item():.2e}, max={basis_scales.max().item():.2e}, mean={basis_scales.mean().item():.2e}")
            print()

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

        # Compute control statistics (for ControlledODE only)
        control_max, control_mean, control_rms = 0.0, 0.0, 0.0
        if hasattr(current_ode, 'controller'):
            # Get trajectory states
            times, states, _ = env.get_trajectory()

            # Compute control for each state
            controls = []
            for state_vec in states:
                # Extract base state (in case of dynamic controller with augmented state)
                if hasattr(current_ode, 'extract_base_state'):
                    base_state = current_ode.extract_base_state(state_vec)
                else:
                    base_state = state_vec

                # Compute control value
                control = current_ode.controller(base_state)
                controls.append(control.detach())

            # Stack and compute statistics
            controls = torch.stack(controls)
            control_max = torch.abs(controls).max().item()
            control_mean = torch.abs(controls).mean().item()
            control_rms = torch.sqrt((controls ** 2).mean()).item()

        # Update basis scales periodically if requested
        if (config.scale_aware_regularization and
            config.reg_scale_update_interval > 0 and
            iteration > 0 and
            iteration % config.reg_scale_update_interval == 0):
            basis_scales = compute_basis_scales(ode, env)
            if config.verbose:
                print(f"[Iteration {iteration}] Updated basis scales: min={basis_scales.min().item():.2e}, max={basis_scales.max().item():.2e}")

        # Compute loss: -reward + penalties
        loss = -reward

        # Add regularization penalties
        l1_reg = torch.tensor(0.0)
        l2_reg = torch.tensor(0.0)

        if config.l1_penalty > 0:
            if basis_scales is not None:
                # Scale-aware regularization: penalize each param proportional to its basis function magnitude
                l1_reg = (basis_scales * torch.abs(params.flatten())).sum()
            else:
                # Standard regularization
                l1_reg = torch.abs(params).sum()
            loss = loss + config.l1_penalty * l1_reg

        if config.l2_penalty > 0:
            if basis_scales is not None:
                # Scale-aware regularization: penalize each param proportional to its basis function magnitude
                l2_reg = (basis_scales * (params.flatten() ** 2)).sum()
            else:
                # Standard regularization
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
        history['control_max'].append(control_max)
        history['control_mean'].append(control_mean)
        history['control_rms'].append(control_rms)

        # Update best parameters
        current_reward = reward.item() if torch.is_tensor(reward) else reward
        if current_reward > best_reward:
            best_reward = current_reward
            best_params = params.clone().detach()

        # Logging
        if config.verbose and (iteration % config.log_interval == 0 or iteration == config.n_iterations - 1):
            # Format parameters for display (flatten if multi-dimensional)
            params_flat = params.flatten()
            params_str = "[" + ", ".join([f"{p.item():.3f}" for p in params_flat]) + "]"
            log_msg = (f"Iter {iteration:4d} | Loss: {loss.item():8.3f} | "
                      f"Reward: {history['reward'][-1]:8.3f} | "
                      f"L1: {history['l1_penalty'][-1]:6.3f} | "
                      f"Non-zero params: {num_nonzero:3d} | "
                      f"Control (max/mean/rms): {control_max:.3f}/{control_mean:.3f}/{control_rms:.3f} | "
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
