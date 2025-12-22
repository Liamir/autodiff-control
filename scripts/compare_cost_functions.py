"""Compare cost function between standalone and mpc.py implementation."""
import numpy as np
import torch
from rpasim.ode.classic_control.population import PopulationDynamics
from rpa_control.mpc import MPCController, MPCConfig

# ==========================================
# Standalone implementation
# ==========================================
def lotka_volterra_standalone(x, u, params):
    a, b, d, g = params['a'], params['b'], params['d'], params['g']
    x1, x2 = x
    dx1 = a*x1 - b*x1*x2
    dx2 = d*x1*x2 - g*x2 + u
    return np.array([dx1, dx2])

def rk4_step_standalone(x, u, dt, params):
    k1 = lotka_volterra_standalone(x, u, params)
    k2 = lotka_volterra_standalone(x + 0.5*dt*k1, u, params)
    k3 = lotka_volterra_standalone(x + 0.5*dt*k2, u, params)
    k4 = lotka_volterra_standalone(x + dt*k3, u, params)
    return x + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

def mpc_objective_standalone(u_seq, x0, N, dt, xref, Q, R, params):
    cost = 0.0
    x_curr = x0.copy()
    u_seq = u_seq.reshape(N)

    for k in range(N):
        u_k = u_seq[k]
        x_next = rk4_step_standalone(x_curr, u_k, dt, params)

        error = x_next - xref
        state_cost = error.T @ Q @ error
        control_cost = R * (u_k**2)

        cost += state_cost + control_cost
        x_curr = x_next

    return cost

# ==========================================
# Test parameters
# ==========================================
params = {'a': 0.5, 'b': 0.025, 'd': 0.005, 'g': 0.5}
dt = 0.1
N = 10
Q = np.diag([1.0, 1.0])
R_standalone = 0.1  # From standalone script
xref = np.array([100.0, 20.0])
x0 = np.array([80.0, 25.0])

u_seq_zero = np.zeros(N)
u_seq_minus5 = np.full(N, -5.0)

print("=" * 60)
print("Cost Function Comparison")
print("=" * 60)
print()

print("Standalone Implementation (R = 0.1)")
print("-" * 60)
cost_standalone_zero = mpc_objective_standalone(u_seq_zero, x0, N, dt, xref, Q, R_standalone, params)
cost_standalone_minus5 = mpc_objective_standalone(u_seq_minus5, x0, N, dt, xref, Q, R_standalone, params)
print(f"Cost with u=0:    {cost_standalone_zero:.4f}")
print(f"Cost with u=-5:   {cost_standalone_minus5:.4f}")
print(f"Improvement:      {cost_standalone_zero - cost_standalone_minus5:.4f}")
print()

print("MPC.py Implementation (Ru = 0.5, R_deltau = 0.5)")
print("-" * 60)
pop_ode = PopulationDynamics()
critical_point = torch.tensor([100.0, 20.0])

mpc_config = MPCConfig(
    prediction_horizon=N,
    dt=dt,
    Q=torch.tensor([1.0, 1.0]),
    Ru=0.5,
    R_deltau=0.5,
    u_min=-20.0,
    u_max=20.0,
)

mpc = MPCController(
    ode=pop_ode,
    config=mpc_config,
    reference_state=critical_point,
    control_indices=[1],
)

cost_mpc_zero = mpc._mpc_objective(u_seq_zero, x0)
cost_mpc_minus5 = mpc._mpc_objective(u_seq_minus5, x0)
print(f"Cost with u=0:    {cost_mpc_zero:.4f}")
print(f"Cost with u=-5:   {cost_mpc_minus5:.4f}")
print(f"Improvement:      {cost_mpc_zero - cost_mpc_minus5:.4f}")
print()

print("Analysis of First Step")
print("-" * 60)
print("\nFor u=0:")
print("  Standalone: state_cost + 0.1 * 0^2 = state_cost + 0")
print("  MPC.py:     state_cost + 0.5 * 0^2 + 0.5 * (0-0)^2 = state_cost + 0")
print()
print("For u=-5:")
print("  Standalone: state_cost + 0.1 * 25 = state_cost + 2.5")
print("  MPC.py:     state_cost + 0.5 * 25 + 0.5 * 25 = state_cost + 25")
print()
print(f"Control penalty ratio: {(0.5 + 0.5) / 0.1:.1f}x higher in MPC.py")
print()

print("Testing with Same R Value (R = Ru = R_deltau = 0.1)")
print("-" * 60)
mpc_config_fixed = MPCConfig(
    prediction_horizon=N,
    dt=dt,
    Q=torch.tensor([1.0, 1.0]),
    Ru=0.1,  # Match standalone
    R_deltau=0.0,  # Disable rate-of-change penalty
    u_min=-20.0,
    u_max=20.0,
)

mpc_fixed = MPCController(
    ode=pop_ode,
    config=mpc_config_fixed,
    reference_state=critical_point,
    control_indices=[1],
)

cost_fixed_zero = mpc_fixed._mpc_objective(u_seq_zero, x0)
cost_fixed_minus5 = mpc_fixed._mpc_objective(u_seq_minus5, x0)
print(f"Cost with u=0:    {cost_fixed_zero:.4f}")
print(f"Cost with u=-5:   {cost_fixed_minus5:.4f}")
print(f"Improvement:      {cost_fixed_zero - cost_fixed_minus5:.4f}")
print()

if abs(cost_fixed_zero - cost_standalone_zero) < 0.1:
    print("✓ Costs match when using same R values!")
else:
    print(f"✗ Costs still differ: {abs(cost_fixed_zero - cost_standalone_zero):.4f}")
    print("  There must be another difference...")
