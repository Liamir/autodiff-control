"""Debug MPC implementation to find why it outputs zero control."""
import torch
import numpy as np
from rpasim.ode.classic_control.population import PopulationDynamics
from rpa_control.mpc import MPCController, MPCConfig

# Setup
pop_ode = PopulationDynamics()
critical_point = torch.tensor([100.0, 20.0])
x0 = torch.tensor([80.0, 25.0], dtype=torch.float32)

print("=" * 60)
print("MPC Debugging")
print("=" * 60)
print()

# Create MPC controller
mpc_config = MPCConfig(
    prediction_horizon=10,
    dt=0.1,
    Q=torch.tensor([1.0, 1.0]),
    Ru=0.5,
    R_deltau=0.5,
    u_min=-20.0,
    u_max=20.0,
    ftol=1e-3,
    disp=True,  # Show optimization output
)

mpc = MPCController(
    ode=pop_ode,
    config=mpc_config,
    reference_state=critical_point,
    control_indices=[1],
)

print("Test 1: Check ODE dynamics method")
print("-" * 60)
x_test = np.array([80.0, 25.0])
dx_no_control = mpc._ode_dynamics(x_test, 0.0)
dx_with_control = mpc._ode_dynamics(x_test, 10.0)
print(f"State: {x_test}")
print(f"dx with u=0.0: {dx_no_control}")
print(f"dx with u=10.0: {dx_with_control}")
print(f"Difference: {dx_with_control - dx_no_control}")
print(f"Expected difference: [0, 10] (control on index 1)")
print()

print("Test 2: Check RK4 integration")
print("-" * 60)
x_next_no_control = mpc._rk4_step(x_test, 0.0, 0.1)
x_next_with_control = mpc._rk4_step(x_test, 10.0, 0.1)
print(f"x_next with u=0.0: {x_next_no_control}")
print(f"x_next with u=10.0: {x_next_with_control}")
print(f"Difference: {x_next_with_control - x_next_no_control}")
print()

print("Test 3: Evaluate cost function")
print("-" * 60)
u_seq_zero = np.zeros(10)
u_seq_ones = np.ones(10)
u_seq_optimal = np.array([-5.0] * 10)  # Reasonable control

cost_zero = mpc._mpc_objective(u_seq_zero, x_test)
cost_ones = mpc._mpc_objective(u_seq_ones, x_test)
cost_optimal = mpc._mpc_objective(u_seq_optimal, x_test)

print(f"Cost with u=0: {cost_zero:.4f}")
print(f"Cost with u=1: {cost_ones:.4f}")
print(f"Cost with u=-5: {cost_optimal:.4f}")
print()

if cost_zero < cost_ones and cost_zero < cost_optimal:
    print("WARNING: Zero control has lowest cost! This explains why MPC outputs zero.")
    print("The cost function may be misconfigured.")
else:
    print("Cost function favors non-zero control, which is correct.")
print()

print("Test 4: Run one MPC step with verbose output")
print("-" * 60)
u_mpc, info = mpc.step(x0)
print(f"\nMPC output:")
print(f"  Control: {u_mpc.item():.4f}")
print(f"  Success: {info['success']}")
print(f"  Cost: {info['cost']:.4f}")
print(f"  Message: {info['message']}")
print(f"  Iterations: {info['nit']}")
print()

print("Test 5: Manual optimization check")
print("-" * 60)
from scipy.optimize import minimize

def test_objective(u_seq):
    return mpc._mpc_objective(u_seq, x0.numpy())

# Test with different initial guesses
for u_init_val in [0.0, -5.0, 5.0]:
    u_init = np.full(10, u_init_val)
    res = minimize(
        test_objective,
        u_init,
        method='SLSQP',
        bounds=[(-20.0, 20.0) for _ in range(10)],
        options={'ftol': 1e-3, 'disp': False}
    )
    print(f"Initial guess u={u_init_val}: optimal u[0]={res.x[0]:.4f}, cost={res.fun:.4f}")
