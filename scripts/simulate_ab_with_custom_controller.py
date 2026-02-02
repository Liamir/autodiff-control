"""Simulate ABControlled with custom state-dependent controller."""
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchdiffeq import odeint
import sys
from pathlib import Path

# Add configs to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import importlib
config_module = importlib.import_module('configs.ab_controlled')
config = config_module.ENV_CONFIG


def simple_controller(state):
    """Custom state-dependent switching controller.

    Args:
        state: torch.Tensor [A, B]

    Returns:
        u: control signal
    """
    A = float(state[0])
    B = float(state[1])
    Bdot = A - B  # From dB/dt = A - B

    u_hard = 1.0
    u_soft = 0.2

    if B > 1:
        if Bdot < 0:
            u = u_hard
        else:
            u = u_soft
    else:
        if Bdot > 0:
            u = u_hard
        else:
            u = u_soft

    return torch.tensor([u])


class ABWithCustomController:
    """AB ODE with custom controller integrated."""

    def __init__(self, base_ode, controller_fn, control_bounds=None):
        self.base_ode = base_ode
        self.controller_fn = controller_fn
        self.control_bounds = control_bounds
        self.nfe = 0  # Function evaluation counter

    def __call__(self, t, state):
        """Compute derivative with custom controller.

        Args:
            t: time (scalar)
            state: [A, B]

        Returns:
            [dA/dt, dB/dt]
        """
        self.nfe += 1

        # Compute control from custom function
        control = self.controller_fn(state)

        # Clip control if bounds specified
        if self.control_bounds is not None:
            control = torch.clamp(control, min=self.control_bounds[0], max=self.control_bounds[1])

        # Compute ODE dynamics with control
        return self.base_ode(t, state, None, self.base_ode.fixed_params, control=control)


# Setup
initial_state = config['initial_state']
time_horizon = 5.0
n_timesteps = 250  # Fine resolution for rk4
t_span = torch.linspace(0, time_horizon, n_timesteps + 1)

# Create base ODE
base_ode = config['create_base_ode']()

# Create ODE with custom controller
controlled_ode = ABWithCustomController(
    base_ode=base_ode,
    controller_fn=simple_controller,
    control_bounds=(0.0, 1.0)
)

print("="*80)
print("SIMULATING ABControlled WITH CUSTOM CONTROLLER")
print("="*80)
print(f"Initial state: {initial_state.numpy()}")
print(f"Time horizon: {time_horizon}")
print(f"Number of timesteps: {n_timesteps}")
print()
print("Custom Controller Logic:")
print("  If B > 1:")
print("    If dB/dt < 0: u = 1.0 (hard)")
print("    Else:         u = 0.2 (soft)")
print("  Else (B <= 1):")
print("    If dB/dt > 0: u = 1.0 (hard)")
print("    Else:         u = 0.2 (soft)")
print()

# Simulate
print("Running simulation...", end=' ', flush=True)
trajectory = odeint(controlled_ode, initial_state, t_span, method='rk4')
print("done")

# Extract results
times = t_span.numpy()
states = trajectory.detach().numpy()

# Compute control trajectory (for plotting)
controls = []
for state in trajectory:
    u = simple_controller(state)
    controls.append(float(u))
controls = np.array(controls)

print()
print("Results:")
print(f"  Initial state: [A={states[0, 0]:.3f}, B={states[0, 1]:.3f}]")
print(f"  Final state:   [A={states[-1, 0]:.3f}, B={states[-1, 1]:.3f}]")
print(f"  Control range: [{controls.min():.3f}, {controls.max():.3f}]")
print(f"  Control switches: {np.sum(np.abs(np.diff(controls)) > 0.1)}")
print()

# Plot
state_names = config['state_var_names']
n_states = states.shape[1]

fig, axes = plt.subplots(n_states + 1, 1, figsize=(10, 2.5 * (n_states + 1)))

# Plot states
for i in range(n_states):
    ax = axes[i]
    ax.plot(times, states[:, i], 'b-', linewidth=2, label=state_names[i])

    # Mark B=1 threshold on B plot
    if i == 1:
        ax.axhline(1.0, color='r', linestyle='--', alpha=0.5, label='threshold=1.0')

    ax.set_xlabel('Time')
    ax.set_ylabel(state_names[i])
    ax.set_title(f'{state_names[i]} Evolution (Custom Controller)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

# Plot control
ax = axes[-1]
ax.plot(times, controls, 'g-', linewidth=2, label='u (control)')
ax.axhline(1.0, color='gray', linestyle='--', alpha=0.3, label='u_hard')
ax.axhline(0.2, color='gray', linestyle=':', alpha=0.3, label='u_soft')
ax.set_xlabel('Time')
ax.set_ylabel('Control u')
ax.set_title('Custom Controller Output')
ax.set_ylim([0, 1.1])
ax.legend()
ax.grid(True, alpha=0.3)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

plt.tight_layout()

output_file = project_root / 'ab_controlled_custom_controller.pdf'
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"Plot saved to: {output_file}")
print("="*80)
