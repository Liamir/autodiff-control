"""Compare constant vs oscillating control strategies for destabilization."""
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

# Create ODE
ode = IFFL2Vars(fixed_params=FIXED_PARAMS, input_signal=constant_input)

# Initial conditions: steady state
x0 = torch.tensor([10.0, 0.1])

# Time span
t_eval = torch.linspace(0.0, 30.0, 600)
times = t_eval.numpy()

print("Comparing control strategies for destabilization")
print(f"Initial state: {x0.numpy()}")
print(f"y_ss = {Y_SS:.4f}")
print("="*60)


def simulate_with_control(control_func, name):
    """Simulate with given control function."""
    def ode_func(t, x):
        t_val = t.item() if torch.is_tensor(t) else t
        u = control_func(t_val)
        u_tensor = torch.tensor([u], dtype=x.dtype, device=x.device)
        return ode.forward(t, x, differentiable_params=None,
                          fixed_params=ode.fixed_params, control=u_tensor)

    states = odeint(ode_func, x0, t_eval, method='dopri5', rtol=1e-6, atol=1e-8)
    states_np = states.numpy()

    # Compute controls and rewards
    controls = np.array([control_func(t) for t in times])
    y_vals = states_np[:, 1]
    y_ss_val = Y_SS.item() if torch.is_tensor(Y_SS) else Y_SS
    deviations = (y_vals - y_ss_val)**2
    cumulative_reward = deviations.sum()

    print(f"\n{name}:")
    print(f"  Cumulative reward (full 30s): {cumulative_reward:.4f}")
    print(f"  Mean deviation²: {deviations.mean():.6f}")
    print(f"  Max |y - y_ss|: {np.abs(y_vals - y_ss_val).max():.6f}")
    print(f"  Final state: x={states_np[-1, 0]:.4f}, y={states_np[-1, 1]:.4f}")

    return states_np, controls, deviations, cumulative_reward


# Strategy 1: Constant u=0.5
states_low, controls_low, dev_low, reward_low = simulate_with_control(
    lambda t: 0.5, "Constant u=0.5 (MPC solution)"
)

# Strategy 2: Constant u=2.0
states_high, controls_high, dev_high, reward_high = simulate_with_control(
    lambda t: 2.0, "Constant u=2.0"
)

# Strategy 3: Oscillating (period=10s)
def oscillating_10s(t):
    return 2.0 if (t % 20) < 10 else 0.5

states_osc10, controls_osc10, dev_osc10, reward_osc10 = simulate_with_control(
    oscillating_10s, "Oscillating u (period=20s, duty=50%)"
)

# Strategy 4: Oscillating (period=5s)
def oscillating_5s(t):
    return 2.0 if (t % 10) < 5 else 0.5

states_osc5, controls_osc5, dev_osc5, reward_osc5 = simulate_with_control(
    oscillating_5s, "Oscillating u (period=10s, duty=50%)"
)

# Strategy 5: Oscillating (period=2s)
def oscillating_2s(t):
    return 2.0 if (t % 4) < 2 else 0.5

states_osc2, controls_osc2, dev_osc2, reward_osc2 = simulate_with_control(
    oscillating_2s, "Oscillating u (period=4s, duty=50%)"
)

print("\n" + "="*60)
print("WINNER:", end=" ")
rewards = [reward_low, reward_high, reward_osc10, reward_osc5, reward_osc2]
names = ["Constant u=0.5", "Constant u=2.0", "Osc(20s)", "Osc(10s)", "Osc(4s)"]
winner_idx = np.argmax(rewards)
print(f"{names[winner_idx]} with cumulative reward {rewards[winner_idx]:.4f}")
print("="*60)

# Plot comparison
set_style()
fig, axes = plt.subplots(4, 1, figsize=(12, 13))

# Plot y trajectories
ax = axes[0]
ax.plot(times, states_low[:, 1], 'b-', linewidth=2, label='Constant u=0.5', alpha=0.7)
ax.plot(times, states_high[:, 1], 'r-', linewidth=2, label='Constant u=2.0', alpha=0.7)
ax.plot(times, states_osc10[:, 1], 'g-', linewidth=2, label='Osc(20s)', alpha=0.7)
ax.plot(times, states_osc5[:, 1], 'm-', linewidth=2, label='Osc(10s)', alpha=0.7)
ax.plot(times, states_osc2[:, 1], 'c-', linewidth=2, label='Osc(4s)', alpha=0.7)
y_ss_val = Y_SS.item() if torch.is_tensor(Y_SS) else Y_SS
ax.axhline(y_ss_val, color='gray', linestyle='--', alpha=0.5, label=f'y_ss={y_ss_val:.2f}')
ax.set_ylabel('y')
ax.set_title('State y - Different Control Strategies')
ax.legend(ncol=2)
ax.grid(True, alpha=0.3)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

# Plot deviations
ax = axes[1]
ax.plot(times, dev_low, 'b-', linewidth=2, label='Constant u=0.5', alpha=0.7)
ax.plot(times, dev_high, 'r-', linewidth=2, label='Constant u=2.0', alpha=0.7)
ax.plot(times, dev_osc10, 'g-', linewidth=2, label='Osc(20s)', alpha=0.7)
ax.plot(times, dev_osc5, 'm-', linewidth=2, label='Osc(10s)', alpha=0.7)
ax.plot(times, dev_osc2, 'c-', linewidth=2, label='Osc(4s)', alpha=0.7)
ax.set_ylabel('(y - y_ss)²')
ax.set_title('Instantaneous Reward (y - y_ss)²')
ax.legend(ncol=2)
ax.grid(True, alpha=0.3)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

# Plot cumulative deviation
ax = axes[2]
dt = times[1] - times[0]
ax.plot(times, np.cumsum(dev_low) * dt, 'b-', linewidth=2, label='Constant u=0.5', alpha=0.7)
ax.plot(times, np.cumsum(dev_high) * dt, 'r-', linewidth=2, label='Constant u=2.0', alpha=0.7)
ax.plot(times, np.cumsum(dev_osc10) * dt, 'g-', linewidth=2, label='Osc(20s)', alpha=0.7)
ax.plot(times, np.cumsum(dev_osc5) * dt, 'm-', linewidth=2, label='Osc(10s)', alpha=0.7)
ax.plot(times, np.cumsum(dev_osc2) * dt, 'c-', linewidth=2, label='Osc(4s)', alpha=0.7)
ax.set_ylabel(r'$\int (y - y_{ss})^2 \, dt$')
ax.set_title('Cumulative Deviation')
ax.legend(ncol=2)
ax.grid(True, alpha=0.3)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

# Plot control signals
ax = axes[3]
ax.plot(times, controls_low, 'b-', linewidth=2, label='Constant u=0.5', alpha=0.7)
ax.plot(times, controls_high, 'r-', linewidth=2, label='Constant u=2.0', alpha=0.7)
ax.plot(times, controls_osc10, 'g-', linewidth=2, label='Osc(20s)', alpha=0.7)
ax.plot(times, controls_osc5, 'm-', linewidth=2, label='Osc(10s)', alpha=0.7)
ax.plot(times, controls_osc2, 'c-', linewidth=2, label='Osc(4s)', alpha=0.7)
ax.set_xlabel('Time')
ax.set_ylabel('Control (u)')
ax.set_title('Control Signals')
ax.legend(ncol=2)
ax.grid(True, alpha=0.3)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

plt.tight_layout()

# Save
plots_dir = Path(__file__).parent.parent / 'plots'
plots_dir.mkdir(exist_ok=True)
output_path = plots_dir / 'control_strategies_comparison.pdf'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"\nPlot saved to: {output_path}")
