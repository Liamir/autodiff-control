"""Test simulation of iffl2vars with manual control specification."""
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
from torchdiffeq import odeint

# Add rpa-simulators to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "rpa-simulators" / "src"))

from rpasim.ode.rpa.iffl2vars import IFFL2Vars
from rpa_control.style import set_style


def constant_input(t):
    """Constant input: 1.0 for all time."""
    if torch.is_tensor(t):
        return torch.tensor(1.0, dtype=t.dtype, device=t.device)
    else:
        return 1.0


# Fixed parameters
FIXED_PARAMS = torch.tensor([1.0, 0.1, 1.0, 1.0])  # [alpha, delta, beta, gamma]
alpha, delta, beta, gamma = FIXED_PARAMS
Y_SS = (beta * delta) / (alpha * gamma)

# Create ODE with constant input signal
ode = IFFL2Vars(fixed_params=FIXED_PARAMS, input_signal=constant_input)

# Initial conditions: start at steady state
# x_ss = alpha * Input / delta = 1.0 * 1.0 / 0.1 = 10.0
# y_ss = 0.1
x0 = torch.tensor([10.0, 0.1])

# Time span: 40 timesteps (20 + 20)
t_eval = torch.linspace(0.0, 40.0, 400)

print("Simulating iffl2vars with manual control...")
print(f"Initial conditions (steady state): x={x0[0]:.2f}, y={x0[1]:.2f}")
print(f"y_ss = {Y_SS:.4f}")
print(f"Input: constant 1.0")
print(f"Control: 0.5 for t<20, 2.0 for t>=20")
print(f"Time span: [0.0, 40.0]")
print(f"Parameters: alpha={FIXED_PARAMS[0]:.2f}, delta={FIXED_PARAMS[1]:.2f}, "
      f"beta={FIXED_PARAMS[2]:.2f}, gamma={FIXED_PARAMS[3]:.2f}")


# Create ODE function with manual control
def ode_func(t, x):
    # Manual control: 0.5 for t < 20, 2.0 for t >= 20
    t_val = t.item() if torch.is_tensor(t) else t
    if t_val < 20.0:
        u = torch.tensor([0.5], dtype=x.dtype, device=x.device)
    else:
        u = torch.tensor([2.0], dtype=x.dtype, device=x.device)

    return ode.forward(t, x, differentiable_params=None,
                      fixed_params=ode.fixed_params, control=u)


# Test ODE function at initial conditions
print("\nTesting ODE function at initial conditions:")
t0 = torch.tensor(0.0)
dx0 = ode_func(t0, x0)
print(f"  t={t0.item()}, x={x0.numpy()}, dx/dt={dx0.numpy()}")

# Solve using torchdiffeq
print("\nSolving ODE...")
states = odeint(ode_func, x0, t_eval, method='dopri5', rtol=1e-6, atol=1e-8)

times = t_eval.numpy()
states = states.numpy()  # Shape: (n_times, n_states)
x_vals = states[:, 0]
y_vals = states[:, 1]

# Compute control at each time point
controls = np.array([0.5 if t < 20.0 else 2.0 for t in times])

# Compute input signal at each time point (constant 1.0)
inputs = np.ones_like(times)

# Plot
set_style()
fig, axes = plt.subplots(4, 1, figsize=(10, 12))

# Plot x
axes[0].plot(times, x_vals, 'b-', linewidth=2, label='x')
axes[0].axvline(20.0, color='gray', linestyle='--', alpha=0.3, label='Control step')
axes[0].axhline(10.0, color='gray', linestyle=':', alpha=0.3, label='x_ss (u=1)')
axes[0].set_ylabel('x')
axes[0].set_title('State x')
axes[0].legend()
axes[0].grid(True, alpha=0.3)
axes[0].spines['right'].set_visible(False)
axes[0].spines['top'].set_visible(False)

# Plot y
axes[1].plot(times, y_vals, 'r-', linewidth=2, label='y')
axes[1].axvline(20.0, color='gray', linestyle='--', alpha=0.3, label='Control step')
axes[1].axhline(Y_SS, color='gray', linestyle=':', alpha=0.3, label=f'y_ss = {Y_SS:.2f}')
axes[1].set_ylabel('y')
axes[1].set_title('State y')
axes[1].legend()
axes[1].grid(True, alpha=0.3)
axes[1].spines['right'].set_visible(False)
axes[1].spines['top'].set_visible(False)

# Plot Control
axes[2].plot(times, controls, 'g-', linewidth=2, label='Control (u)')
axes[2].axhline(0.5, color='gray', linestyle='--', alpha=0.5)
axes[2].axhline(2.0, color='gray', linestyle='--', alpha=0.5)
axes[2].axhline(1.0, color='gray', linestyle=':', alpha=0.5, label='u=1 (no effect)')
axes[2].axvline(20.0, color='gray', linestyle='--', alpha=0.5)
axes[2].set_ylabel('Control')
axes[2].set_title('Control Signal (u)')
axes[2].legend()
axes[2].grid(True, alpha=0.3)
axes[2].spines['right'].set_visible(False)
axes[2].spines['top'].set_visible(False)

# Plot Input
axes[3].plot(times, inputs, 'purple', linewidth=2, label='Input (constant)')
axes[3].set_xlabel('Time')
axes[3].set_ylabel('Input')
axes[3].set_title('Input Signal')
axes[3].legend()
axes[3].grid(True, alpha=0.3)
axes[3].spines['right'].set_visible(False)
axes[3].spines['top'].set_visible(False)

plt.tight_layout()

# Ensure plots directory exists
plots_dir = Path(__file__).parent.parent / 'plots'
plots_dir.mkdir(exist_ok=True)

output_path = plots_dir / 'iffl2vars_manual_control.pdf'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"\nPlot saved to: {output_path}")

# Print some statistics
print(f"\nResults:")
print(f"  x range: [{x_vals.min():.4f}, {x_vals.max():.4f}]")
print(f"  y range: [{y_vals.min():.4f}, {y_vals.max():.4f}]")
print(f"  State at t=20 (before control step): x={x_vals[200]:.4f}, y={y_vals[200]:.4f}")
print(f"  Final state (t=40): x={x_vals[-1]:.4f}, y={y_vals[-1]:.4f}")
print(f"\n  |y - y_ss| at t=20: {abs(y_vals[200] - Y_SS):.4f}")
print(f"  |y - y_ss| at t=40: {abs(y_vals[-1] - Y_SS):.4f}")
