"""Core training loop for one stage of optimization."""
import torch
import time
from typing import Callable, Optional, Dict, Any, List, Tuple
from .config import TrainingConfig
from .evaluation import evaluate_controller


def one_stage_training(
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
    """Execute a single training stage of n_iterations.

    Args:
        env: DifferentiableEnv
        ode: ODE with differentiable parameters
        params: Parameter tensor to optimize
        optimizer: PyTorch optimizer
        config: Training configuration
        basis_scales: Basis function scales for scale-aware regularization
        param_mask: Optional mask for parameters (1=active, 0=frozen)
        phase_name: Name for logging (e.g., "train", "stage1", "stage2")
        callback: Optional callback function
        track_lowest_l1: If True, also track parameters with lowest L1 norm

    Returns:
        history: Dictionary of training metrics
        best_reward: Best reward achieved in this stage
        best_params: Best parameters from this stage
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
        'eval_reward': [],  # Evaluation on fixed ICs
    }

    best_reward = float('-inf')
    best_eval_reward = float('-inf')  # Track best eval score
    best_params = params.clone().detach()

    # Track lowest L1 norm parameters if requested (useful for stage 2)
    lowest_l1_norm = float('inf')
    lowest_l1_params = None if not track_lowest_l1 else params.clone().detach()

    # Check if evaluation is enabled
    use_eval = (config.eval_interval > 0 and
                config.eval_initial_states is not None and
                len(config.eval_initial_states) > 0)

    # Timing accumulators
    timing_stats = {
        'forward': 0.0,
        'control_stats': 0.0,
        'regularization': 0.0,
        'evaluation': 0.0,
        'backward': 0.0,
        'optimizer': 0.0,
        'logging': 0.0,
        'total': 0.0,
    }
    # NFE (Number of Function Evaluations) accumulators
    nfe_stats = {
        'forward': 0,
        'total_calls': 0,
    }
    timing_report_interval = max(50, config.log_interval)

    for iteration in range(config.n_iterations):
        iter_start = time.time()
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

        # Run simulation with optional TBPTT
        time_horizon = env.time_horizon if hasattr(env, 'time_horizon') else 10.0

        forward_start = time.time()
        if config.use_tbptt and config.tbptt_truncation_steps > 0:
            # Truncated BPTT: simulate in chunks and detach between chunks
            # This prevents gradients from flowing too far back in time

            # Get dt from environment (assumes constant dt)
            # We need to infer dt from n_reward_steps
            n_reward_steps = env.n_reward_steps if hasattr(env, 'n_reward_steps') else 100
            dt = time_horizon / n_reward_steps

            # Calculate chunk size
            chunk_duration = config.tbptt_truncation_steps * dt
            num_chunks = int(time_horizon / chunk_duration)
            if num_chunks == 0:
                num_chunks = 1
                chunk_duration = time_horizon

            if iteration == 0 and config.verbose:
                print(f"[DEBUG] TBPTT: time_horizon={time_horizon}, dt={dt:.4f}, chunk_duration={chunk_duration:.4f}, num_chunks={num_chunks}")

            # Accumulate reward and trajectories across chunks
            total_reward = torch.tensor(0.0, requires_grad=True)
            all_times = []
            all_states = []
            all_rewards = []
            total_nfe_forward = 0

            # Initial state
            current_state = state.clone()

            for chunk_idx in range(num_chunks):
                # Detach state to break gradient flow (except for first chunk)
                if chunk_idx > 0:
                    current_state = current_state.detach()

                # Debug: check computational graph setup for first few chunks
                if iteration == 0 and chunk_idx < 3 and config.verbose:
                    print(f"[DEBUG] Chunk {chunk_idx}: START current_state={current_state.detach().numpy()}, requires_grad={current_state.requires_grad}")
                    print(f"[DEBUG] Chunk {chunk_idx}: ODE controller params require_grad={current_ode.controller.params.requires_grad if hasattr(current_ode, 'controller') else 'N/A'}")

                # Set environment state directly (don't call reset() which would go back to initial_state)
                env.current_state = current_state
                env.current_time = chunk_idx * chunk_duration
                # Clear trajectory for this chunk
                env.trajectory_segments = []
                env.time_segments = []
                env.reward_segments = []

                # Restore perturbed params if applicable
                if config.perturb_param_indices is not None and len(config.perturb_param_indices) > 0:
                    current_ode.fixed_params = original_fixed_params.clone()
                    perturbed_fixed_params = current_ode.fixed_params.clone()
                    for idx in config.perturb_param_indices:
                        fold = config.perturb_fold_change
                        log_factor = torch.rand(1).item() * 2 * torch.log(torch.tensor(fold)).item() - torch.log(torch.tensor(fold)).item()
                        random_factor = torch.exp(torch.tensor(log_factor)).item()
                        perturbed_fixed_params[idx] = perturbed_fixed_params[idx] * random_factor
                    current_ode.fixed_params = perturbed_fixed_params

                # Simulate chunk
                chunk_horizon = min(chunk_duration, time_horizon - chunk_idx * chunk_duration)
                obs, chunk_reward, terminated, truncated, info = env.step((current_ode, chunk_horizon))

                # Update current state for next chunk
                current_state = obs[1]

                # Debug: check chunk results
                if iteration == 0 and chunk_idx < 3 and config.verbose:
                    print(f"[DEBUG] Chunk {chunk_idx}: chunk_reward={chunk_reward.item() if torch.is_tensor(chunk_reward) else chunk_reward:.2e}, chunk_reward.requires_grad={chunk_reward.requires_grad if torch.is_tensor(chunk_reward) else 'N/A'}")
                    print(f"[DEBUG] Chunk {chunk_idx}: final_state={current_state.detach().numpy()}, has_nan={torch.isnan(current_state).any().item()}")

                # Accumulate reward and NFE
                total_reward = total_reward + chunk_reward
                total_nfe_forward += info.get('nfe_forward', 0)

                # Collect trajectory for debugging/metrics
                times, states, rewards = env.get_trajectory()
                all_times.extend(times.tolist() if hasattr(times, 'tolist') else times)
                all_states.extend(states)
                all_rewards.extend(rewards)

            reward = total_reward

            # Debug: check forward pass on first iteration
            if iteration == 0 and config.verbose:
                state_has_nan = any(torch.isnan(s).any().item() for s in all_states)
                state_has_inf = any(torch.isinf(s).any().item() for s in all_states)
                state_max = max(torch.abs(s).max().item() for s in all_states)
                reward_has_nan = torch.isnan(reward).any().item() if torch.is_tensor(reward) else False
                print(f"[DEBUG] Iteration 0 forward pass (TBPTT): state_has_nan={state_has_nan}, state_has_inf={state_has_inf}, state_max={state_max:.2e}, reward_has_nan={reward_has_nan}, reward={reward}")
                print(f"[DEBUG] total_reward.requires_grad={reward.requires_grad if torch.is_tensor(reward) else 'N/A'}")

            # Store trajectory in environment for metrics computation
            # (This is a bit hacky but needed for control statistics below)
            env._trajectory = (all_times, all_states, all_rewards)

        else:
            # Standard full BPTT: single simulation through full time horizon
            obs, reward, terminated, truncated, info = env.step((current_ode, time_horizon))
            total_nfe_forward = info.get('nfe_forward', 0)

            # Debug: check forward pass on first iteration
            if iteration == 0 and config.verbose:
                times, states, rewards = env.get_trajectory()
                state_has_nan = any(torch.isnan(s).any().item() for s in states)
                state_has_inf = any(torch.isinf(s).any().item() for s in states)
                state_max = max(torch.abs(s).max().item() for s in states)
                reward_has_nan = torch.isnan(reward).any().item() if torch.is_tensor(reward) else False
                print(f"[DEBUG] Iteration 0 forward pass: state_has_nan={state_has_nan}, state_has_inf={state_has_inf}, state_max={state_max:.2e}, reward_has_nan={reward_has_nan}, reward={reward}")

        # Restore original fixed params
        if original_fixed_params is not None:
            current_ode.fixed_params = original_fixed_params

        forward_time = time.time() - forward_start
        timing_stats['forward'] += forward_time

        # Track NFE stats
        nfe_stats['forward'] += total_nfe_forward
        nfe_stats['total_calls'] += 1

        # Apply steady state filtering if requested
        if config.steady_state_fraction > 0:
            times, states, rewards = env.get_trajectory()
            start_idx = int(len(times) * config.steady_state_fraction)
            reward = rewards[start_idx:].sum()

        # Compute control statistics
        control_stats_start = time.time()
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

        control_stats_time = time.time() - control_stats_start
        timing_stats['control_stats'] += control_stats_time

        # Compute loss with regularization
        reg_start = time.time()
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

        reg_time = time.time() - reg_start
        timing_stats['regularization'] += reg_time

        # Evaluate on fixed ICs if enabled
        eval_start = time.time()
        current_eval_reward = None
        if use_eval and (iteration % config.eval_interval == 0 or iteration == config.n_iterations - 1):
            # Run evaluation (with parameter perturbations if enabled in training)
            current_eval_reward = evaluate_controller(
                env, ode, config.eval_initial_states, config.steady_state_fraction,
                perturb_param_indices=config.perturb_param_indices,
                perturb_fold_change=config.perturb_fold_change,
                n_param_samples=config.n_param_samples_eval,
            )
            # Update best params based on eval score
            if current_eval_reward > best_eval_reward:
                best_eval_reward = current_eval_reward
                best_params = params.clone().detach()

        # Update best parameters BEFORE optimizer step (based on single-iteration reward if no eval)
        # (reward was computed with current params, so we need to save these params, not the post-step params)
        current_reward = reward.item() if torch.is_tensor(reward) else reward
        if not use_eval:  # Only track single-iteration best if not using evaluation
            if current_reward > best_reward:
                best_reward = current_reward
                best_params = params.clone().detach()
        else:
            # Still track single-iteration reward for logging
            if current_reward > best_reward:
                best_reward = current_reward

        # Track lowest L1 norm parameters if requested
        if track_lowest_l1:
            current_l1_norm = torch.abs(params).sum().item()
            if current_l1_norm < lowest_l1_norm:
                lowest_l1_norm = current_l1_norm
                lowest_l1_params = params.clone().detach()

        eval_time = time.time() - eval_start
        timing_stats['evaluation'] += eval_time

        # Backpropagation
        backward_start = time.time()
        loss.backward()
        backward_time = time.time() - backward_start
        timing_stats['backward'] += backward_time

        # Debug: check gradients on first iteration
        if iteration == 0 and config.verbose:
            if params.grad is not None:
                grad_has_nan = torch.isnan(params.grad).any().item()
                grad_has_inf = torch.isinf(params.grad).any().item()
                grad_max = torch.abs(params.grad).max().item()
                grad_mean = torch.abs(params.grad).mean().item()
                print(f"[DEBUG] Iteration 0 gradients (before clipping): has_nan={grad_has_nan}, has_inf={grad_has_inf}, max={grad_max:.2e}, mean={grad_mean:.2e}")
                print(f"[DEBUG] Gradient values: {params.grad.flatten().tolist()}")
            else:
                print(f"[DEBUG] Iteration 0: params.grad is None")

        # Gradient clipping (before applying mask)
        if config.gradient_clip_norm > 0 and params.grad is not None:
            torch.nn.utils.clip_grad_norm_([params], config.gradient_clip_norm)

        # Debug: check gradients after clipping on first iteration
        if iteration == 0 and config.verbose and config.gradient_clip_norm > 0:
            if params.grad is not None:
                grad_has_nan = torch.isnan(params.grad).any().item()
                grad_has_inf = torch.isinf(params.grad).any().item()
                grad_max = torch.abs(params.grad).max().item()
                grad_mean = torch.abs(params.grad).mean().item()
                print(f"[DEBUG] Iteration 0 gradients (after clipping): has_nan={grad_has_nan}, has_inf={grad_has_inf}, max={grad_max:.2e}, mean={grad_mean:.2e}")

        # Apply mask to gradients if provided
        if param_mask is not None and params.grad is not None:
            params.grad.data = params.grad.data * param_mask.float()

        optimizer_start = time.time()
        optimizer.step()
        optimizer_time = time.time() - optimizer_start
        timing_stats['optimizer'] += optimizer_time

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
        history['eval_reward'].append(current_eval_reward if current_eval_reward is not None else float('nan'))

        # Logging
        logging_start = time.time()
        if config.verbose and (iteration % config.log_interval == 0 or iteration == config.n_iterations - 1):
            params_flat = params.flatten()
            params_str = "[" + ", ".join([f"{p.item():.3f}" for p in params_flat]) + "]"
            log_msg = (f"{phase_name:10s} {iteration:4d} | Loss: {loss.item():8.3f} | "
                      f"Reward: {history['reward'][-1]:8.3f} | "
                      f"L1: {history['l1_penalty'][-1]:6.3f} | "
                      f"Non-zero params: {num_nonzero:3d} | "
                      f"Control (max/mean/rms): {control_max:.3f}/{control_mean:.3f}/{control_rms:.3f}")
            # Add eval reward if available
            if current_eval_reward is not None:
                log_msg += f" | Eval: {current_eval_reward:8.3f}"
            log_msg += f" | Params: {params_str}"
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

        logging_time = time.time() - logging_start
        timing_stats['logging'] += logging_time

        # Track total iteration time
        iter_time = time.time() - iter_start
        timing_stats['total'] += iter_time

        # Print timing report periodically
        if config.verbose and (iteration % timing_report_interval == 0 or iteration == config.n_iterations - 1):
            n_iters = iteration + 1
            print(f"\n{'='*80}")
            print(f"TIMING REPORT (averaged over {n_iters} iterations)")
            print(f"{'='*80}")
            total_avg = timing_stats['total'] / n_iters
            print(f"{'Total per iteration:':<30} {total_avg*1000:>8.2f} ms  (100.0%)")
            print(f"{'-'*80}")
            for key in ['forward', 'backward', 'optimizer', 'control_stats', 'regularization', 'evaluation', 'logging']:
                avg_time = timing_stats[key] / n_iters
                pct = (avg_time / total_avg * 100) if total_avg > 0 else 0
                print(f"  {key.capitalize():<28} {avg_time*1000:>8.2f} ms  ({pct:>5.1f}%)")
            print(f"{'-'*80}")
            # Print NFE stats
            avg_nfe_forward = nfe_stats['forward'] / nfe_stats['total_calls'] if nfe_stats['total_calls'] > 0 else 0
            print(f"{'NFE (forward pass avg):':<30} {avg_nfe_forward:>8.0f} calls")
            print(f"{'='*80}\n")

    # Return best eval reward if evaluation was used, otherwise return best single-iteration reward
    final_best_reward = best_eval_reward if use_eval else best_reward
    return history, final_best_reward, best_params, lowest_l1_params
