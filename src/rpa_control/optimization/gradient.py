"""Gradient-based optimization for ODE parameters through differentiable simulation."""
import torch
from typing import Callable, Optional, Dict, Any, List

from .config import TrainingConfig
from .evaluation import compute_basis_scales
from .training_loop import one_stage_training


def train_ode_parameters(
    env,
    ode,
    optimizer: Optional[torch.optim.Optimizer] = None,
    config: Optional[TrainingConfig] = None,
    callback: Optional[Callable[[int, Dict[str, Any]], None]] = None,
    output_dir: Optional[Any] = None,
    full_config: Optional[dict] = None,
) -> Dict[str, List[float]]:
    """
    Train ODE parameters using gradient descent through differentiable simulation.

    Loss: -reward + l1_penalty * L1(params) + l2_penalty * L2(params)

    Supports three-stage training pipeline for sparsification:
    Stage 1: Normal training without regularization
    Stage 2: L1 regularization to encourage sparsity
    Stage 3: Hard thresholding + continued training

    Args:
        env: DifferentiableEnv from rpasim
        ode: ODE instance with differentiable parameters to optimize
        optimizer: PyTorch optimizer (default: Adam with config.learning_rate)
        config: Training configuration
        callback: Optional callback function called after each iteration with (iteration, metrics)
        output_dir: Directory to save trajectory plots (optional)
        full_config: Full configuration dict for trajectory plotting (optional)

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

        history, best_reward, best_params, _ = one_stage_training(
            env=env,
            ode=ode,
            params=params,
            optimizer=optimizer,
            config=stage1_config,
            basis_scales=basis_scales,
            param_mask=None,
            phase_name="stage1",
            callback=callback,
            output_dir=output_dir,
            full_config=full_config,
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

        stage2_history, stage2_best_reward, stage2_best_params, stage2_lowest_l1_params = one_stage_training(
            env=env,
            ode=ode,
            params=params,
            optimizer=optimizer,
            config=stage2_config,
            basis_scales=basis_scales,
            param_mask=None,
            phase_name="stage2",
            callback=callback,
            track_lowest_l1=True,
            output_dir=output_dir,
            full_config=full_config,
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

            stage3_history, stage3_best_reward, stage3_best_params, _ = one_stage_training(
                env=env,
                ode=ode,
                params=params,
                optimizer=optimizer,
                config=stage3_config,
                basis_scales=basis_scales,
                param_mask=threshold_mask,
                phase_name="stage3",
                callback=callback,
                output_dir=output_dir,
                full_config=full_config,
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

    # ========== Default: Single-stage training ==========
    else:
        history, best_reward, best_params, _ = one_stage_training(
            env=env,
            ode=ode,
            params=params,
            optimizer=optimizer,
            config=config,
            basis_scales=basis_scales,
            param_mask=None,
            phase_name="train",
            callback=callback,
            output_dir=output_dir,
            full_config=full_config,
        )

    # Restore best parameters
    params.data.copy_(best_params)

    # Sync parameters to controller if needed
    if hasattr(ode, 'update_controller_params'):
        ode.update_controller_params()

    history['best_reward'] = best_reward
    history['best_params'] = best_params

    return history
