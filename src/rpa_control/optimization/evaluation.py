"""Evaluation utilities for controllers and basis function analysis."""
import torch
from typing import List, Optional


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


def evaluate_controller(
    env,
    ode,
    initial_states: List[torch.Tensor],
    steady_state_fraction: float = 0.0,
    perturb_param_indices: Optional[List[int]] = None,
    perturb_fold_change: float = 2.0,
    n_param_samples: int = 10,
) -> float:
    """Evaluate controller on multiple fixed initial conditions.

    Args:
        env: DifferentiableEnv (should have initial_state_bounds set to None for evaluation)
        ode: ODE with controller to evaluate
        initial_states: List of initial state tensors to evaluate on
        steady_state_fraction: Fraction of trajectory to skip before computing reward
        perturb_param_indices: Optional indices of fixed_params to perturb (if None, no perturbation)
        perturb_fold_change: Fold change for parameter perturbations
        n_param_samples: Number of parameter samples per initial condition (only used if perturb_param_indices is not None)

    Returns:
        Average reward across all initial conditions (and parameter samples if perturbations enabled)
    """
    total_reward = 0.0
    n_evaluations = 0

    # Temporarily disable randomization in env
    original_bounds = env.initial_state_bounds
    env.initial_state_bounds = None

    # Save original fixed params
    original_fixed_params = ode.fixed_params.clone() if hasattr(ode, 'fixed_params') and ode.fixed_params is not None else None

    for init_state in initial_states:
        # Determine how many parameter samples to use
        n_samples = n_param_samples if (perturb_param_indices is not None and len(perturb_param_indices) > 0) else 1

        for _ in range(n_samples):
            # Apply parameter perturbations if requested
            if perturb_param_indices is not None and len(perturb_param_indices) > 0 and original_fixed_params is not None:
                perturbed_fixed_params = original_fixed_params.clone()
                for idx in perturb_param_indices:
                    fold = perturb_fold_change
                    # Sample uniformly in log space: log(param * factor) where factor ∈ [1/fold, fold]
                    log_factor = torch.rand(1).item() * 2 * torch.log(torch.tensor(fold)).item() - torch.log(torch.tensor(fold)).item()
                    factor = torch.exp(torch.tensor(log_factor)).item()
                    perturbed_fixed_params[idx] = original_fixed_params[idx] * factor
                ode.fixed_params = perturbed_fixed_params

            # Set env initial state
            env.initial_state = init_state

            # Reset and run
            obs, info = env.reset()
            current_ode, state = obs

            time_horizon = env.time_horizon if hasattr(env, 'time_horizon') else 10.0
            obs, reward, terminated, truncated, info = env.step((ode, time_horizon))

            # Apply steady state filtering if requested
            if steady_state_fraction > 0:
                times, states, rewards = env.get_trajectory()
                start_idx = int(len(times) * steady_state_fraction)
                reward = rewards[start_idx:].sum()

            total_reward += reward.item() if torch.is_tensor(reward) else reward
            n_evaluations += 1

    # Restore original params
    if original_fixed_params is not None:
        ode.fixed_params = original_fixed_params

    # Restore original bounds
    env.initial_state_bounds = original_bounds

    return total_reward / n_evaluations
