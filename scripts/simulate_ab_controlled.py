"""Simulate ABControlled for horizon=50 with constant controller."""
import time
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

# Setup
initial_state = config['initial_state']
time_horizon = 50.0
n_timesteps = 1000  # More steps for rk4 stability
t_span = torch.linspace(0, time_horizon, n_timesteps + 1)

# Create ODE with constant controller (u = 1.0)
ode = config['create_ode'](controller_order=2, include_constant=True)

# Set controller to constant 1.0 (only constant term = 1.0, rest = 0)
params = torch.zeros_like(ode.differentiable_params)
params[0, 0] = 1.0  # Constant term for u
ode.differentiable_params.data.copy_(params)
ode.update_controller_params()

print("="*80)
print("SIMULATING ABControlled WITH CONSTANT CONTROLLER")
print("="*80)
print(f"Initial state: {initial_state.numpy()}")
print(f"Time horizon: {time_horizon}")
print(f"Number of timesteps: {n_timesteps}")
print(f"Controller: u = 1.0 (constant)")
print()

# Simulate with timing - dopri5
print("Running simulation with dopri5...", end=' ', flush=True)
start = time.time()
trajectory_dopri5 = odeint(ode, initial_state, t_span, method='dopri5')
elapsed_dopri5 = time.time() - start
print(f"done ({elapsed_dopri5:.3f}s)")

# Simulate with rk4
print("Running simulation with rk4...", end=' ', flush=True)
start = time.time()
trajectory_rk4 = odeint(ode, initial_state, t_span, method='rk4')
elapsed_rk4 = time.time() - start
print(f"done ({elapsed_rk4:.3f}s)")

# Use dopri5 trajectory for plotting
trajectory = trajectory_dopri5
elapsed = elapsed_dopri5

# Extract results
times = t_span.numpy()
states_dopri5 = trajectory_dopri5.detach().numpy()
states_rk4 = trajectory_rk4.detach().numpy()

print()
print("Timing Comparison:")
print(f"  dopri5: {elapsed_dopri5:.3f}s")
print(f"  rk4:    {elapsed_rk4:.3f}s")
print(f"  Speedup: {elapsed_dopri5/elapsed_rk4:.2f}x {'(rk4 faster)' if elapsed_rk4 < elapsed_dopri5 else '(dopri5 faster)'}")
print()

# Compare trajectories
final_diff = np.abs(states_dopri5[-1] - states_rk4[-1])
max_diff = np.max(np.abs(states_dopri5 - states_rk4))
mean_diff = np.mean(np.abs(states_dopri5 - states_rk4))

print("Trajectory Comparison:")
print(f"  Final state (dopri5): {states_dopri5[-1]}")
print(f"  Final state (rk4):    {states_rk4[-1]}")
print(f"  Final state diff:     {final_diff}")
print(f"  Max trajectory diff:  {max_diff:.6e}")
print(f"  Mean trajectory diff: {mean_diff:.6e}")
print(f"  Trajectories match: {np.allclose(states_dopri5, states_rk4, rtol=1e-3, atol=1e-4)}")
print()

states = states_dopri5  # Use dopri5 for plotting

# Plot
state_names = config['state_var_names']
n_states = states.shape[1]

fig, axes = plt.subplots(n_states, 1, figsize=(10, 2.5 * n_states))
if n_states == 1:
    axes = [axes]

for i in range(n_states):
    ax = axes[i]
    ax.plot(times, states_dopri5[:, i], 'b-', linewidth=2, label=f'{state_names[i]} (dopri5)', alpha=0.8)
    ax.plot(times, states_rk4[:, i], 'g--', linewidth=2, label=f'{state_names[i]} (rk4)', alpha=0.8)

    # Mark target if it exists
    if 'target_vars' in config and i in config['target_vars']:
        target = config['target_vars'][i]
        ax.axhline(target, color='r', linestyle='--', alpha=0.5, label=f'target={target}')

    ax.set_xlabel('Time')
    ax.set_ylabel(state_names[i])
    ax.set_title(f'{state_names[i]} Evolution: dopri5 vs rk4 (Constant u=1.0)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

plt.tight_layout()

output_file = project_root / 'ab_controlled_constant_simulation.pdf'
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"Plot saved to: {output_file}")
print("="*80)
