"""Simple test: apply controller to population dynamics."""
import torch
import matplotlib.pyplot as plt
from rpasim.ode.classic_control.population import PopulationDynamics
from rpasim.plot.ode import plot_trajectory
from rpa_control.controllers import StaticController, ControlledODE
from rpa_control.paths import save_fig
from rpa_control.style import set_style

set_style()

# Create population ODE
pop_ode = PopulationDynamics()
print("Population Dynamics (Lotka-Volterra)")
print("="*60)
print(pop_ode)
print()

# Critical point from paper: (c/d, a/b) = (100, 20)
critical_point = torch.tensor([100.0, 20.0])
print(f"Critical point: prey={critical_point[0]:.1f}, predator={critical_point[1]:.1f}")

# Initial state (not at equilibrium)
initial_state = torch.tensor([80.0, 25.0])
print(f"Initial state: prey={initial_state[0]:.1f}, predator={initial_state[1]:.1f}")
print()

# Test 1: Plot uncontrolled system
print("Test 1: Uncontrolled system")
print("-"*60)
time_horizon = 100.0
fig_uncontrolled, axes = plot_trajectory(pop_ode, initial_state, time_horizon)

# Add critical point
axes[0].axhline(y=critical_point[0], color='red', linestyle='--', label='critical', alpha=0.5)
axes[1].axhline(y=critical_point[1], color='red', linestyle='--', label='critical', alpha=0.5)
axes[0].legend()
axes[1].legend()
axes[0].set_title('prey (uncontrolled)')
axes[1].set_title('predator (uncontrolled)')

for ax in axes:
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

plt.tight_layout()
save_fig(fig_uncontrolled, 'population_uncontrolled')
print("Saved: plots/population_uncontrolled.pdf")
print()

# Test 2: Create static controller
print("Test 2: Static controller setup")
print("-"*60)
controller = StaticController(
    n_state_vars=2,      # prey, predator
    n_control_vars=1,    # control affects predator only
    order=2,             # polynomial order 2
    include_constant=True
)

print(f"Controller parameters: {controller.params.numel()}")
print(f"Basis functions: {controller.get_basis_names(['prey', 'predator'])}")
print()

# Test 3: Create controlled ODE
print("Test 3: Controlled ODE")
print("-"*60)
# Control affects predator (index 1) as per paper eq 3.1b
controlled_ode = ControlledODE(
    base_ode=pop_ode,
    controller=controller,
    control_indices=[1]
)

print(f"State dimension: {controlled_ode.state_dim}")
print(f"Differentiable params: {controlled_ode.differentiable_params.numel()}")
print()

# Test 4: Forward pass
print("Test 4: Forward pass test")
print("-"*60)
t = torch.tensor(0.0)
state = initial_state
derivative = controlled_ode(t, state)
control_output = controller(state)

print(f"State: prey={state[0]:.2f}, predator={state[1]:.2f}")
print(f"Control output u: {control_output.item():.4f}")
print(f"Derivative: dprey/dt={derivative[0]:.4f}, dpredator/dt={derivative[1]:.4f}")
print()

# Test 5: Controller summary
print("Test 5: Controller summary")
print("-"*60)
print(controlled_ode.get_controller_summary(['prey', 'predator'], ['u']))
print()

print("="*60)
print("All tests passed! ✓")
print("="*60)
