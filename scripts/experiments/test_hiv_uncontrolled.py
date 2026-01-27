"""Test uncontrolled HIV system dynamics with different solvers."""
import torch
from torchdiffeq import odeint
from rpasim.ode.classic_control.hiv import HIVTreatment

# Create HIV ODE
ode = HIVTreatment()

# Initial state
initial_state = torch.tensor([10.0, 0.1, 0.1, 0.1, 0.1])

# Time points
time_horizon = 50.0
n_steps = 100
t = torch.linspace(0, time_horizon, n_steps)

print(f"Testing uncontrolled HIV system")
print(f"Initial state: {initial_state.numpy()}")
print(f"Time horizon: {time_horizon}")
print(f"Time steps: {n_steps}")
print(f"dt: {time_horizon / (n_steps - 1):.4f}")
print()

# Simulate with zero control (u=0)
def ode_func(t, x):
    return ode(t, x, fixed_params=ode.fixed_params, control=torch.tensor([0.0]))

# Test both solvers
for method in ['dopri5', 'rk4']:
    print("=" * 60)
    print(f"Testing with method: {method}")
    print("=" * 60)

    try:
        solution = odeint(ode_func, initial_state, t, method=method)

        print("Simulation successful!")
        print(f"Final state: {solution[-1].detach().numpy()}")
        print()
        print("State trajectory:")
        print(f"  t={0:6.2f}: {solution[0].detach().numpy()}")
        print(f"  t={10:6.2f}: {solution[20].detach().numpy()}")
        print(f"  t={20:6.2f}: {solution[40].detach().numpy()}")
        print(f"  t={30:6.2f}: {solution[60].detach().numpy()}")
        print(f"  t={40:6.2f}: {solution[80].detach().numpy()}")
        print(f"  t={50:6.2f}: {solution[-1].detach().numpy()}")
        print()

        # Check for negative values
        min_vals = solution.min(dim=0).values
        print(f"Minimum values across trajectory:")
        for i, (name, val) in enumerate(zip(ode.variable_names, min_vals)):
            print(f"  {name}: {val.item():.4f}")

        if (min_vals < 0).any():
            print(f"\n❌ WARNING: Negative values detected with {method} (unphysical)")
        else:
            print(f"\n✅ All values remain positive with {method}")

    except Exception as e:
        print(f"❌ Simulation failed with {method}: {e}")
        import traceback
        traceback.print_exc()

    print()
