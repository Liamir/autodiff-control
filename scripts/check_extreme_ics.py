"""Check stability of extreme initial conditions for population dynamics."""
import torch
import matplotlib.pyplot as plt
from torchdiffeq import odeint
from rpasim.ode.classic_control.population import PopulationDynamics
from rpa_control.style import set_style

set_style()

# Create uncontrolled population dynamics
pop_ode = PopulationDynamics()

# Test multiple ranges
# Critical point: (100, 20), typical: (80, 25)
# We want to maintain prey > predator

ranges_to_test = {
    'OLD [10-150, 10-150]': {
        'prey': (10.0, 150.0),
        'predator': (10.0, 150.0),
    },
    'NARROW [60-140, 10-30]': {
        'prey': (60.0, 140.0),
        'predator': (10.0, 30.0),
    },
    'MEDIUM [50-150, 10-30]': {
        'prey': (50.0, 150.0),
        'predator': (10.0, 30.0),
    },
}

print("Testing different initial condition ranges:")
print("=" * 80)

for range_name, bounds in ranges_to_test.items():
    print(f"\n{range_name}")
    print("-" * 80)

    # Define extreme initial conditions for this range
    prey_min, prey_max = bounds['prey']
    pred_min, pred_max = bounds['predator']

    extreme_ics = {
        f'min-min ({prey_min:.0f}, {pred_min:.0f})': torch.tensor([prey_min, pred_min]),
        f'min-max ({prey_min:.0f}, {pred_max:.0f})': torch.tensor([prey_min, pred_max]),
        f'max-min ({prey_max:.0f}, {pred_min:.0f})': torch.tensor([prey_max, pred_min]),
        f'max-max ({prey_max:.0f}, {pred_max:.0f})': torch.tensor([prey_max, pred_max]),
        'critical (100, 20)': torch.tensor([100.0, 20.0]),
        'typical (80, 25)': torch.tensor([80.0, 25.0]),
    }

    # Simulation parameters
    time_horizon = 40.0
    n_steps = 1000

    # Test each extreme IC and compute max values
    max_values = []
    for label, ic in extreme_ics.items():
        # Solve ODE
        times = torch.linspace(0, time_horizon, n_steps)
        states = odeint(pop_ode, ic, times, method='dopri5')
        prey = states[:, 0].detach().numpy()
        predator = states[:, 1].detach().numpy()

        # Check for explosion
        max_prey = prey.max()
        max_predator = predator.max()
        max_values.append((max_prey, max_predator))

        print(f"  {label:30s} | Max prey: {max_prey:8.1f} | Max predator: {max_predator:8.1f}")

    # Summary statistics for this range
    max_preys = [v[0] for v in max_values[:4]]  # Only extreme corners
    max_predators = [v[1] for v in max_values[:4]]
    print(f"\n  Summary (extreme corners only):")
    print(f"    Max prey across corners:     {max(max_preys):8.1f}")
    print(f"    Max predator across corners: {max(max_predators):8.1f}")
    print(f"    Avg max prey:                {sum(max_preys)/len(max_preys):8.1f}")
    print(f"    Avg max predator:            {sum(max_predators)/len(max_predators):8.1f}")

print("\n" + "=" * 80)
print("RECOMMENDATION:")
print("  Choose the range with lowest max values for more stable training.")
print("=" * 80)
