"""Trajectory plotting during training."""
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional
from ..style import set_style


def plot_training_trajectory(
    ode,
    env,
    params: torch.Tensor,
    config: dict,
    output_dir: Path,
    iteration: int,
    reason: str = "",
):
    """Plot trajectory during training at checkpoints.

    Args:
        ode: ODE with controller
        env: Environment for simulation
        params: Controller parameters to use
        config: Config dict with plotting parameters
        output_dir: Directory to save plots
        iteration: Current training iteration
        reason: Reason for plotting (e.g., "best", "periodic", "final")
    """
    set_style()

    # Set parameters in ODE
    ode.differentiable_params.data.copy_(params)
    if hasattr(ode, 'update_controller_params'):
        ode.update_controller_params()

    # Get initial state (use first eval initial state if available, otherwise config initial_state)
    if config.get('eval_initial_states') and len(config.get('eval_initial_states', [])) > 0:
        initial_state = config['eval_initial_states'][0]
        if not isinstance(initial_state, torch.Tensor):
            initial_state = torch.tensor(initial_state)
    else:
        initial_state = config.get('initial_state')
        if not isinstance(initial_state, torch.Tensor):
            initial_state = torch.tensor(initial_state)

    # Reset environment and simulate
    env.reset()
    time_horizon = config.get('time_horizon', 10.0)

    # Manually set initial state
    with torch.no_grad():
        # Set the state in environment
        if hasattr(env, 'initial_state'):
            env.initial_state = initial_state
        obs, info = env.reset()
        current_ode, state = obs

        # Run forward simulation
        obs, reward, terminated, truncated, info = env.step((ode, time_horizon))

        # Get trajectory
        times, states, rewards = env.get_trajectory()

    # Convert to numpy for plotting
    times_np = np.array([t.item() if torch.is_tensor(t) else t for t in times])
    states_np = np.array([s.detach().numpy() if torch.is_tensor(s) else s for s in states])

    # Compute controls from states
    controls = []
    is_dynamic = hasattr(ode.controller, 'output') if hasattr(ode, 'controller') else False

    with torch.no_grad():
        for state_vec in states[:-1]:  # Exclude last state (no control computed there)
            if not torch.is_tensor(state_vec):
                state_vec = torch.tensor(state_vec)

            if hasattr(ode, 'controller'):
                if is_dynamic:
                    base_state_dim = ode.base_state_dim
                    controller_state = state_vec[base_state_dim:]
                    control = ode.controller.output(controller_state)
                else:
                    if hasattr(ode, 'extract_base_state'):
                        base_state = ode.extract_base_state(state_vec)
                    else:
                        base_state = state_vec
                    control = ode.controller(base_state)
                controls.append(control.detach().numpy())

    controls_np = np.array(controls)

    # Get variable names from config
    state_var_names = config.get('state_var_names', [f'x{i}' for i in range(states_np.shape[1])])
    control_names = config.get('control_names', ['u'])

    # Get reference state (target_vars)
    target_vars = config.get('target_vars', {})
    reference_state = np.zeros(states_np.shape[1])
    for var_name, target_val in target_vars.items():
        if var_name in state_var_names:
            idx = state_var_names.index(var_name)
            reference_state[idx] = target_val

    # Create figure
    n_states = states_np.shape[1]
    n_controls = controls_np.shape[1] if len(controls_np.shape) > 1 else 1

    fig, axes = plt.subplots(n_states + 1, 1, figsize=(10, 3 * (n_states + 1)))
    if n_states == 1:
        axes = [axes[0], axes[1]]

    # Plot states
    for i in range(n_states):
        ax = axes[i]
        ax.plot(times_np, states_np[:, i], 'b-', linewidth=2, label=f'{state_var_names[i]} (trained)')
        if target_vars and state_var_names[i] in target_vars:
            ax.axhline(reference_state[i], color='r', linestyle='--', alpha=0.5, label='target')
        ax.set_xlabel('Time')
        ax.set_ylabel(state_var_names[i])
        title_suffix = f" - iter {iteration}" + (f" ({reason})" if reason else "")
        ax.set_title(f'{state_var_names[i]} Evolution{title_suffix}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

    # Plot control (raw and clamped)
    ax = axes[-1]
    control_times = times_np[:-1]  # Control is one step shorter

    # Get control bounds from config
    control_bounds = config.get('control_bounds', None)

    if n_controls == 1:
        # Flatten if needed
        controls_1d = controls_np.flatten() if len(controls_np.shape) > 1 else controls_np

        # Plot raw controller output
        ax.plot(control_times, controls_1d, 'g--', linewidth=2, alpha=0.6,
               label=f'{control_names[0]} (raw)')

        # Plot clamped control if bounds are available
        if control_bounds is not None:
            controls_clamped = np.clip(controls_1d, control_bounds[0], control_bounds[1])
            ax.plot(control_times, controls_clamped, 'g-', linewidth=2,
                   label=f'{control_names[0]} (clamped)')

            # Add horizontal lines for bounds
            ax.axhline(control_bounds[0], color='r', linestyle=':', alpha=0.5,
                      label=f'lower bound ({control_bounds[0]})')
            ax.axhline(control_bounds[1], color='r', linestyle=':', alpha=0.5,
                      label=f'upper bound ({control_bounds[1]})')
    else:
        for i in range(n_controls):
            # Plot raw controller output
            ax.plot(control_times, controls_np[:, i], linewidth=2, alpha=0.6, linestyle='--',
                   label=f'{control_names[i] if i < len(control_names) else f"u{i}"} (raw)')

            # Plot clamped control if bounds are available
            if control_bounds is not None:
                controls_clamped = np.clip(controls_np[:, i], control_bounds[0], control_bounds[1])
                ax.plot(control_times, controls_clamped, linewidth=2,
                       label=f'{control_names[i] if i < len(control_names) else f"u{i}"} (clamped)')

    ax.axhline(0.0, color='k', linestyle='--', alpha=0.3)
    ax.set_xlabel('Time')
    ax.set_ylabel('Control')
    ax.set_title('Controller Output (Raw MLP output vs Clamped)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    plt.tight_layout()

    # Save figure
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if reason:
        filename = output_dir / f'trajectory_iter{iteration:04d}_{reason}.pdf'
    else:
        filename = output_dir / f'trajectory_iter{iteration:04d}.pdf'

    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close(fig)

    return filename
