"""Visualize stability of NARROW initial condition range for population dynamics."""
import torch
import matplotlib.pyplot as plt
from torchdiffeq import odeint
from rpasim.ode.classic_control.population import PopulationDynamics
from rpa_control.style import set_style

set_style()

# Create uncontrolled population dynamics
pop_ode = PopulationDynamics()

# Define extreme initial conditions for NARROW range [60-140, 10-30]
extreme_ics = {
    'min-min (60, 10)': torch.tensor([60.0, 10.0]),
    'min-max (60, 30)': torch.tensor([60.0, 30.0]),
    'max-min (140, 10)': torch.tensor([140.0, 10.0]),
    'max-max (140, 30)': torch.tensor([140.0, 30.0]),
    'critical (100, 20)': torch.tensor([100.0, 20.0]),
    'typical (80, 25)': torch.tensor([80.0, 25.0]),
}

# Simulation parameters
time_horizon = 40.0
n_steps = 1000

# Create figure with 2 subplots: phase plot and time series
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#a65628']

for (label, ic), color in zip(extreme_ics.items(), colors):
    # Solve ODE
    times = torch.linspace(0, time_horizon, n_steps)
    states = odeint(pop_ode, ic, times, method='dopri5')
    prey = states[:, 0].detach().numpy()
    predator = states[:, 1].detach().numpy()
    times_np = times.detach().numpy()

    # Check for explosion
    max_prey = prey.max()
    max_predator = predator.max()

    print(f"{label:25s} | Max prey: {max_prey:8.1f} | Max predator: {max_predator:8.1f}")

    # Phase plot
    axes[0].plot(prey, predator, label=label, color=color, linewidth=2, alpha=0.8)
    axes[0].scatter([ic[0].item()], [ic[1].item()], color=color, s=100, marker='o',
                    edgecolors='black', linewidths=1.5, zorder=5)

    # Time series
    axes[1].plot(times_np, prey, color=color, linewidth=2, alpha=0.8, linestyle='-')
    axes[1].plot(times_np, predator, color=color, linewidth=2, alpha=0.8, linestyle='--')

# Mark critical point
axes[0].scatter([100], [20], color='red', s=200, marker='*',
                edgecolors='black', linewidths=2, zorder=10, label='Critical point*')

# Phase plot formatting
axes[0].set_xlabel('Prey')
axes[0].set_ylabel('Predator')
axes[0].set_title('Phase Plot: NARROW Range [60-140, 10-30] (Uncontrolled)')
axes[0].legend(fontsize=9, loc='upper right')
axes[0].grid(True, alpha=0.3)
axes[0].set_xlim(50, 210)
axes[0].set_ylim(5, 45)

# Time series formatting
axes[1].set_xlabel('Time')
axes[1].set_ylabel('Population')
axes[1].set_title('Time Series (Prey=solid, Predator=dashed)')
axes[1].grid(True, alpha=0.3)
axes[1].set_xlim(0, time_horizon)

plt.tight_layout()
plt.savefig('plots/narrow_range_stability.pdf', bbox_inches='tight', dpi=150)
print(f"\nFigure saved to plots/narrow_range_stability.pdf")
