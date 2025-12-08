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
    scale_aware_regularization: bool = False  # Normalize basis functions by their RMS for numerical stability
    reg_scale_update_interval: int = 0  # Update basis scales every N iterations (0 = only compute once at start)
    # Three-stage training pipeline (improved alternative to iterative LS)
    use_three_stage: bool = False  # Enable 3-stage: normal → L1 regularization → thresholding
    n_iterations_stage1: int = 1000  # Stage 1: normal training without regularization
    n_iterations_stage2: int = 500   # Stage 2: continued training with L1 regularization
    l1_penalty_stage2: float = 0.01  # L1 penalty coefficient for stage 2
    n_iterations_stage3: int = 500   # Stage 3: continued training after thresholding
    threshold_value_stage3: float = 1e-3  # Absolute threshold for stage 3: |θ| < threshold → 0
    # Deprecated: Iterative least squares with thresholding (replaced by use_three_stage)
    use_thresholding: bool = False  # DEPRECATED: Apply iterative LS: train, threshold, retrain until convergence
    threshold_value: float = 1e-3  # DEPRECATED: Absolute threshold: |θ| < threshold → 0
    max_threshold_rounds: int = 10  # DEPRECATED: Maximum number of threshold-retrain rounds


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
        # Dynamic controller: separate scales for observing and actuating params
        observing_basis_values = []
        actuating_basis_values = []

        for state_vec in states:
            # Split state
            base_state = state_vec[:ode.base_state_dim]
            controller_state = state_vec[ode.base_state_dim:]

            # Observing basis: Φ(X, C)
            augmented_state = torch.cat([base_state, controller_state])
            observing_basis = polynomial_basis(
                augmented_state,
                ode.controller.observing_order,
                ode.controller.include_constant
            )
            observing_basis_values.append(observing_basis.detach())

            # Actuating basis: Ψ(C)
            actuating_basis = polynomial_basis(
                controller_state,
                ode.controller.actuating_order,
                ode.controller.include_constant
            )
            actuating_basis_values.append(actuating_basis.detach())

        # Stack and compute RMS for each basis function
        observing_basis_values = torch.stack(observing_basis_values)  # (n_timesteps, n_basis_obs)
        actuating_basis_values = torch.stack(actuating_basis_values)  # (n_timesteps, n_basis_act)

        observing_scales = torch.sqrt((observing_basis_values ** 2).mean(dim=0))
        actuating_scales = torch.sqrt((actuating_basis_values ** 2).mean(dim=0))

        # Store per-basis scales in controller for basis normalization (Approach 1)
        ode.controller.observing_basis_scales_per_basis = observing_scales + 1e-8
        ode.controller.actuating_basis_scales_per_basis = actuating_scales + 1e-8

        # Flatten and concatenate (matching parameter order in ControlledODE)
        # Order: [observing_params (n_controller_states, n_basis_obs), actuating_params (n_control_vars, n_basis_act)]
        n_controller_states = ode.controller.n_controller_states
        n_control_vars = ode.controller.n_control_vars

        # Repeat scales for each output dimension
        observing_scales_flat = observing_scales.repeat(n_controller_states)
        actuating_scales_flat = actuating_scales.repeat(n_control_vars)

        scales = torch.cat([observing_scales_flat, actuating_scales_flat])

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

        # Store per-basis scales in controller for basis normalization (Approach 1)
        ode.controller.basis_scales_per_basis = basis_scales + 1e-8

        # Repeat for each control output
        n_control_vars = ode.controller.n_control_vars
        scales = basis_scales.repeat(n_control_vars)

    # Add small epsilon to avoid division by zero
    scales = scales + 1e-8

    return scales


def _train_single_round(
    env,
    ode,
    params: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    config: TrainingConfig,
    basis_scales: Optional[torch.Tensor],
    param_mask: Optional[torch.Tensor] = None,
    phase_name: str = "train",
    callback: Optional[Callable[[int, Dict[str, Any]], None]] = None,
    track_lowest_l1: bool = False,
) -> Tuple[Dict[str, List[float]], float, torch.Tensor, Optional[torch.Tensor]]:
    """Execute a single training round of n_iterations.

    Args:
        env: DifferentiableEnv
        ode: ODE with differentiable parameters
        params: Parameter tensor to optimize
        optimizer: PyTorch optimizer
        config: Training configuration
        basis_scales: Basis function scales for scale-aware regularization
        param_mask: Optional mask for parameters (1=active, 0=frozen)
        phase_name: Name for logging (e.g., "train", "round1", "round2")
        callback: Optional callback function
        track_lowest_l1: If True, also track parameters with lowest L1 norm

    Returns:
        history: Dictionary of training metrics
        best_reward: Best reward achieved in this round
        best_params: Best parameters from this round
        lowest_l1_params: Parameters with lowest L1 norm (or None if not tracked)
    """
    history = {
        'loss': [],
        'reward': [],
        'l1_penalty': [],
        'l2_penalty': [],
        'num_nonzero_params': [],
        'control_max': [],
        'control_mean': [],
        'control_rms': [],
    }

    best_reward = float('-inf')
    best_params = params.clone().detach()

    # Track lowest L1 norm parameters if requested (useful for stage 2)
    lowest_l1_norm = float('inf')
    lowest_l1_params = None if not track_lowest_l1 else params.clone().detach()

    for iteration in range(config.n_iterations):
        optimizer.zero_grad()

        # Reset environment
        obs, info = env.reset()
        current_ode, state = obs

        # Save original fixed params
        original_fixed_params = current_ode.fixed_params.clone() if hasattr(current_ode, 'fixed_params') and current_ode.fixed_params is not None else None

        # Apply perturbations if requested
        perturbed_factors = None
        if config.perturb_param_indices is not None and len(config.perturb_param_indices) > 0:
            perturbed_fixed_params = current_ode.fixed_params.clone()
            perturbed_factors = []
            for idx in config.perturb_param_indices:
                fold = config.perturb_fold_change
                log_factor = torch.rand(1).item() * 2 * torch.log(torch.tensor(fold)).item() - torch.log(torch.tensor(fold)).item()
                random_factor = torch.exp(torch.tensor(log_factor)).item()
                perturbed_fixed_params[idx] = perturbed_fixed_params[idx] * random_factor
                perturbed_factors.append(random_factor)
            current_ode.fixed_params = perturbed_fixed_params

        # Run simulation
        time_horizon = env.time_horizon if hasattr(env, 'time_horizon') else 10.0
        obs, reward, terminated, truncated, info = env.step((current_ode, time_horizon))

        # Restore original fixed params
        if original_fixed_params is not None:
            current_ode.fixed_params = original_fixed_params

        # Apply steady state filtering if requested
        if config.steady_state_fraction > 0:
            times, states, rewards = env.get_trajectory()
            start_idx = int(len(times) * config.steady_state_fraction)
            reward = rewards[start_idx:].sum()

        # Compute control statistics
        control_max, control_mean, control_rms = 0.0, 0.0, 0.0
        if hasattr(current_ode, 'controller'):
            times, states, _ = env.get_trajectory()
            controls = []
            is_dynamic = hasattr(current_ode.controller, 'output')
            for state_vec in states:
                if is_dynamic:
                    base_state_dim = current_ode.base_state_dim
                    controller_state = state_vec[base_state_dim:]
                    control = current_ode.controller.output(controller_state)
                else:
                    if hasattr(current_ode, 'extract_base_state'):
                        base_state = current_ode.extract_base_state(state_vec)
                    else:
                        base_state = state_vec
                    control = current_ode.controller(base_state)
                controls.append(control.detach())
            controls = torch.stack(controls)
            control_max = torch.abs(controls).max().item()
            control_mean = torch.abs(controls).mean().item()
            control_rms = torch.sqrt((controls ** 2).mean()).item()

        # Compute loss with regularization
        loss = -reward
        l1_reg = torch.tensor(0.0)
        l2_reg = torch.tensor(0.0)

        # Approach 1: Uniform penalties on scaled parameters
        # (basis normalization happens in controller's forward pass)
        if config.l1_penalty > 0:
            l1_reg = torch.abs(params).sum()
            loss = loss + config.l1_penalty * l1_reg

        if config.l2_penalty > 0:
            l2_reg = (params ** 2).sum()
            loss = loss + config.l2_penalty * l2_reg

        # Update best parameters BEFORE optimizer step
        # (reward was computed with current params, so we need to save these params, not the post-step params)
        current_reward = reward.item() if torch.is_tensor(reward) else reward
        if current_reward > best_reward:
            best_reward = current_reward
            best_params = params.clone().detach()

        # Track lowest L1 norm parameters if requested
        if track_lowest_l1:
            current_l1_norm = torch.abs(params).sum().item()
            if current_l1_norm < lowest_l1_norm:
                lowest_l1_norm = current_l1_norm
                lowest_l1_params = params.clone().detach()

        # Backpropagation
        loss.backward()

        # Apply mask to gradients if provided
        if param_mask is not None and params.grad is not None:
            params.grad.data = params.grad.data * param_mask.float()

        optimizer.step()

        # Apply mask to parameters if provided
        if param_mask is not None:
            params.data = params.data * param_mask.float()

        # Count non-zero parameters
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

        # Logging
        if config.verbose and (iteration % config.log_interval == 0 or iteration == config.n_iterations - 1):
            params_flat = params.flatten()
            params_str = "[" + ", ".join([f"{p.item():.3f}" for p in params_flat]) + "]"
            log_msg = (f"{phase_name:10s} {iteration:4d} | Loss: {loss.item():8.3f} | "
                      f"Reward: {history['reward'][-1]:8.3f} | "
                      f"L1: {history['l1_penalty'][-1]:6.3f} | "
                      f"Non-zero params: {num_nonzero:3d} | "
                      f"Control (max/mean/rms): {control_max:.3f}/{control_mean:.3f}/{control_rms:.3f} | "
                      f"Params: {params_str}")
            if perturbed_factors is not None:
                factors_str = "[" + ", ".join([f"{f:.2f}x" for f in perturbed_factors]) + "]"
                log_msg += f" | β perturbations: {factors_str}"
            print(log_msg)

        # Callback
        if callback is not None:
            metrics = {
                'iteration': iteration,
                'loss': loss.item(),
                'reward': history['reward'][-1],
                'l1_penalty': history['l1_penalty'][-1],
                'l2_penalty': history['l2_penalty'][-1],
                'num_nonzero_params': num_nonzero,
                'ode': current_ode,
                'phase': phase_name,
            }
            callback(iteration, metrics)

    return history, best_reward, best_params, lowest_l1_params


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

    Supports two sparsification approaches:
    1. L1 regularization: Soft thresholding via penalty term
    2. Iterative LS with thresholding: Hard thresholding with retrain until convergence

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

    # Compute basis scales for basis normalization
    basis_scales = None
    if config.scale_aware_regularization:
        if config.verbose:
            print("Computing basis function scales for basis normalization...")
        basis_scales = compute_basis_scales(ode, env)
        if config.verbose:
            print(f"Basis scales (RMS): min={basis_scales.min().item():.2e}, max={basis_scales.max().item():.2e}, mean={basis_scales.mean().item():.2e}")
            print()

    # ========== Three-stage training pipeline ==========
    if config.use_three_stage:
        if config.verbose:
            print(f"{'='*60}")
            print(f"THREE-STAGE TRAINING PIPELINE")
            print(f"{'='*60}")
            print(f"Stage 1: Normal training ({config.n_iterations_stage1} iterations)")
            print(f"Stage 2: L1 regularization (λ={config.l1_penalty_stage2}, {config.n_iterations_stage2} iterations)")
            print(f"Stage 3: Thresholding (threshold={config.threshold_value_stage3:.1e}, {config.n_iterations_stage3} iterations)")
            print()

        # ===== Stage 1: Normal training (no L1 regularization) =====
        if config.verbose:
            print(f"{'='*60}")
            print(f"STAGE 1: Normal Training (no regularization)")
            print(f"{'='*60}")

        # Temporarily disable L1 penalty for stage 1
        original_l1_penalty = config.l1_penalty
        config.l1_penalty = 0.0

        # Create config for stage 1
        stage1_config = TrainingConfig(
            n_iterations=config.n_iterations_stage1,
            learning_rate=config.learning_rate,
            l1_penalty=0.0,  # No L1 in stage 1
            l2_penalty=config.l2_penalty,
            log_interval=config.log_interval,
            verbose=config.verbose,
            steady_state_fraction=config.steady_state_fraction,
            perturb_param_indices=config.perturb_param_indices,
            perturb_fold_change=config.perturb_fold_change,
            scale_aware_regularization=config.scale_aware_regularization,
            reg_scale_update_interval=config.reg_scale_update_interval,
        )

        history, best_reward, best_params, _ = _train_single_round(
            env=env,
            ode=ode,
            params=params,
            optimizer=optimizer,
            config=stage1_config,
            basis_scales=basis_scales,
            param_mask=None,
            phase_name="stage1",
            callback=callback
        )

        # Restore parameters to best from stage 1
        params.data.copy_(best_params)

        # Sync parameters to controller if needed
        if hasattr(ode, 'update_controller_params'):
            ode.update_controller_params()

        if config.verbose:
            print()
            print(f"Stage 1 complete: best_reward={best_reward:.3f}")
            params_str = "[" + ", ".join([f"{p.item():.3f}" for p in params.flatten()]) + "]"
            print(f"Parameters after stage 1: {params_str}")
            print()

        # ===== Stage 2: L1 regularization (continued training) =====
        if config.verbose:
            print(f"{'='*60}")
            print(f"STAGE 2: L1 Regularization (λ={config.l1_penalty_stage2})")
            print(f"{'='*60}")

        # Create fresh optimizer for stage 2 (to reset momentum)
        optimizer = torch.optim.Adam([params], lr=config.learning_rate)

        # Create config for stage 2 with L1 penalty
        stage2_config = TrainingConfig(
            n_iterations=config.n_iterations_stage2,
            learning_rate=config.learning_rate,
            l1_penalty=config.l1_penalty_stage2,
            l2_penalty=config.l2_penalty,
            log_interval=config.log_interval,
            verbose=config.verbose,
            steady_state_fraction=config.steady_state_fraction,
            perturb_param_indices=config.perturb_param_indices,
            perturb_fold_change=config.perturb_fold_change,
            scale_aware_regularization=config.scale_aware_regularization,
            reg_scale_update_interval=config.reg_scale_update_interval,
        )

        stage2_history, stage2_best_reward, stage2_best_params, stage2_lowest_l1_params = _train_single_round(
            env=env,
            ode=ode,
            params=params,
            optimizer=optimizer,
            config=stage2_config,
            basis_scales=basis_scales,
            param_mask=None,
            phase_name="stage2",
            callback=callback,
            track_lowest_l1=True
        )

        # Append stage 2 history
        for key in history.keys():
            if key not in ['best_reward', 'best_params']:
                history[key].extend(stage2_history[key])

        # For stage 2, use parameters with LOWEST L1 NORM for thresholding
        # This gives us the sparsest parameters that still worked reasonably well
        stage2_sparsest_params = stage2_lowest_l1_params

        # Sync parameters to controller if needed
        if hasattr(ode, 'update_controller_params'):
            ode.update_controller_params()

        if config.verbose:
            print()
            print(f"Stage 2 complete")
            print(f"  Best reward during stage 2: {stage2_best_reward:.3f}")
            print(f"  Final reward: {history['reward'][-1]:.3f}")
            sparsest_l1 = torch.abs(stage2_sparsest_params).sum().item()
            params_str = "[" + ", ".join([f"{p.item():.3f}" for p in stage2_sparsest_params.flatten()]) + "]"
            print(f"  Sparsest parameters (L1={sparsest_l1:.3f}, used for thresholding): {params_str}")
            num_small = (torch.abs(stage2_sparsest_params) < config.threshold_value_stage3).sum().item()
            print(f"  Parameters below threshold ({config.threshold_value_stage3:.1e}): {num_small}/{stage2_sparsest_params.numel()}")
            print()

        # ===== Stage 3: Thresholding (continued training) =====
        if config.verbose:
            print(f"{'='*60}")
            print(f"STAGE 3: Thresholding (threshold={config.threshold_value_stage3:.1e})")
            print(f"{'='*60}")

        # Apply threshold to stage 2 sparsest params (lowest L1 norm)
        threshold_mask = torch.abs(stage2_sparsest_params) >= config.threshold_value_stage3
        num_nonzero = threshold_mask.sum().item()

        if config.verbose:
            print(f"Applied threshold: {num_nonzero}/{stage2_sparsest_params.numel()} parameters remain")

        # Set small parameters to zero (using stage 2 sparsest params)
        params.data.copy_(stage2_sparsest_params)
        params.data = params.data * threshold_mask.float()

        # Sync parameters to controller if needed
        if hasattr(ode, 'update_controller_params'):
            ode.update_controller_params()

        if num_nonzero > 0:
            # Create fresh optimizer for stage 3
            optimizer = torch.optim.Adam([params], lr=config.learning_rate)

            # Create config for stage 3 (no L1 penalty)
            stage3_config = TrainingConfig(
                n_iterations=config.n_iterations_stage3,
                learning_rate=config.learning_rate,
                l1_penalty=0.0,  # No L1 in stage 3
                l2_penalty=config.l2_penalty,
                log_interval=config.log_interval,
                verbose=config.verbose,
                steady_state_fraction=config.steady_state_fraction,
                perturb_param_indices=config.perturb_param_indices,
                perturb_fold_change=config.perturb_fold_change,
                scale_aware_regularization=config.scale_aware_regularization,
                reg_scale_update_interval=config.reg_scale_update_interval,
            )

            stage3_history, stage3_best_reward, stage3_best_params, _ = _train_single_round(
                env=env,
                ode=ode,
                params=params,
                optimizer=optimizer,
                config=stage3_config,
                basis_scales=basis_scales,
                param_mask=threshold_mask,
                phase_name="stage3",
                callback=callback
            )

            # Append stage 3 history
            for key in history.keys():
                if key not in ['best_reward', 'best_params']:
                    history[key].extend(stage3_history[key])

            # Save stage 3 best params (best sparse controller after thresholding)
            stage3_best_params_sparse = stage3_best_params.clone()
            stage3_best_reward_sparse = stage3_best_reward

            # Sync parameters to controller if needed
            if hasattr(ode, 'update_controller_params'):
                ode.update_controller_params()

            if config.verbose:
                print()
                print(f"Stage 3 complete")
                print(f"  Best reward: {stage3_best_reward_sparse:.3f}")
                params_str = "[" + ", ".join([f"{p.item():.3f}" for p in stage3_best_params_sparse.flatten()]) + "]"
                print(f"  Best sparse parameters: {params_str}")
        else:
            if config.verbose:
                print("All parameters zeroed. Skipping stage 3 training.")
            stage3_best_params_sparse = params.clone().detach()
            stage3_best_reward_sparse = float('-inf')

        # Restore original L1 penalty
        config.l1_penalty = original_l1_penalty

        # Save both controllers in history
        history['stage1_best_reward'] = best_reward  # Best dense controller (from stage 1)
        history['stage1_best_params'] = best_params.clone()
        history['stage3_best_reward'] = stage3_best_reward_sparse  # Best sparse controller (from stage 3)
        history['stage3_best_params'] = stage3_best_params_sparse.clone()

        if config.verbose:
            print()
            print(f"{'='*60}")
            print(f"THREE-STAGE TRAINING COMPLETE")
            print(f"{'='*60}")
            print(f"Stage 1 best (dense): reward={best_reward:.3f}, params={best_params.numel()}")
            print(f"Stage 3 best (sparse): reward={stage3_best_reward_sparse:.3f}, params={(torch.abs(stage3_best_params_sparse) > 1e-6).sum().item()}/{stage3_best_params_sparse.numel()}")
            print()
            print("Restoring Stage 3 best parameters (sparse controller)")
            print()

        # Restore stage 3 best parameters (sparse controller)
        params.data.copy_(stage3_best_params_sparse)
        best_reward = stage3_best_reward_sparse
        best_params = stage3_best_params_sparse

    # ========== Initial training round (for iterative LS or default) ==========
    elif config.use_thresholding:
        if config.verbose:
            print(f"{'='*60}")
            print(f"INITIAL TRAINING (before thresholding)")
            print(f"{'='*60}")

        history, best_reward, best_params, _ = _train_single_round(
            env=env,
            ode=ode,
            params=params,
            optimizer=optimizer,
            config=config,
            basis_scales=basis_scales,
            param_mask=None,
            phase_name="initial",
            callback=callback
        )

    # ========== Default: Single-stage training ==========
    else:
        history, best_reward, best_params, _ = _train_single_round(
            env=env,
            ode=ode,
            params=params,
            optimizer=optimizer,
            config=config,
            basis_scales=basis_scales,
            param_mask=None,
            phase_name="train",
            callback=callback
        )

    # ========== Iterative thresholding (if enabled) ==========
    if config.use_thresholding:
        if config.verbose:
            print()
            print(f"{'='*60}")
            print(f"ITERATIVE THRESHOLDING")
            print(f"{'='*60}")

        # Track best from each sparsity level
        history['round_best_rewards'] = [best_reward]  # Include initial round
        history['round_best_params'] = [best_params.clone()]
        history['round_num_params'] = [params.numel()]  # Initial: all params

        # Track sparsity pattern for convergence detection
        previous_mask = None

        for round_idx in range(1, config.max_threshold_rounds + 1):
            # Apply absolute threshold to create mask
            current_mask = torch.abs(params) >= config.threshold_value

            # Count non-zero parameters
            num_nonzero = current_mask.sum().item()

            if config.verbose:
                print()
                print(f"Round {round_idx}: Applied threshold {config.threshold_value:.1e}, {num_nonzero}/{params.numel()} params remain")

            # Check convergence: sparsity pattern unchanged
            if previous_mask is not None and torch.equal(current_mask, previous_mask):
                if config.verbose:
                    print(f"Converged! Sparsity pattern unchanged.")
                break

            previous_mask = current_mask.clone()

            # If all parameters are zeroed, stop
            if num_nonzero == 0:
                if config.verbose:
                    print("All parameters zeroed. Stopping.")
                break

            # Reinitialize all parameters to zero (avoiding potentially unstable thresholded values)
            # Active parameters will be learned from scratch with the sparsity constraint
            params.data.zero_()

            # Create fresh optimizer for this round (important: don't carry momentum from previous round)
            optimizer = torch.optim.Adam([params], lr=config.learning_rate)

            # Retrain with mask
            if config.verbose:
                print(f"Retraining with {num_nonzero} active parameters...")

            round_history, round_best_reward, round_best_params, _ = _train_single_round(
                env=env,
                ode=ode,
                params=params,
                optimizer=optimizer,
                config=config,
                basis_scales=basis_scales,
                param_mask=current_mask,
                phase_name=f"round{round_idx}",
                callback=callback
            )

            # Append round history to overall history
            for key in history.keys():
                if key not in ['best_reward', 'best_params', 'round_best_rewards', 'round_best_params', 'round_num_params']:
                    history[key].extend(round_history[key])

            # Store best from this sparsity level
            history['round_best_rewards'].append(round_best_reward)
            history['round_best_params'].append(round_best_params.clone())
            history['round_num_params'].append(num_nonzero)

            # Update best if this round improved
            if round_best_reward > best_reward:
                best_reward = round_best_reward
                best_params = round_best_params.clone()

        if config.verbose:
            print()
            print(f"{'='*60}")
            print(f"THRESHOLDING COMPLETE")
            print(f"{'='*60}")
            print()
            print("Best controllers at each sparsity level:")
            print(f"{'Round':<10} {'Params':<10} {'Reward':<15} {'Parameters'}")
            print("-" * 80)
            # Initial round
            params_str = "[" + ", ".join([f"{p.item():.3f}" for p in history['round_best_params'][0].flatten()]) + "]"
            print(f"{'Initial':<10} {history['round_num_params'][0]:<10} {history['round_best_rewards'][0]:<15.3f} {params_str}")
            # Subsequent rounds
            for i in range(1, len(history['round_best_rewards'])):
                params_str = "[" + ", ".join([f"{p.item():.3f}" for p in history['round_best_params'][i].flatten()]) + "]"
                print(f"{f'Round {i}':<10} {history['round_num_params'][i]:<10} {history['round_best_rewards'][i]:<15.3f} {params_str}")

    # Restore best parameters
    params.data.copy_(best_params)

    # Sync parameters to controller if needed
    if hasattr(ode, 'update_controller_params'):
        ode.update_controller_params()

    history['best_reward'] = best_reward
    history['best_params'] = best_params

    return history
