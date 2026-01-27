"""Direct comparison of cost function outputs."""
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

def mpc_objective_standalone(u_seq, x0, N, dt, xref, Q, Ru, R_deltau, u_prev, params):
    cost = 0.0
    x_curr = x0.copy()
    u_seq = u_seq.reshape(N)

    for k in range(N):
        u_k = u_seq[k]
        x_next = rk4_step_standalone(x_curr, u_k, dt, params)

        error = x_next - xref
        state_cost = error.T @ Q @ error

        control_magnitude_cost = Ru * (u_k**2)

        if k == 0:
            du = u_k - u_prev
        else:
            du = u_k - u_seq[k-1]
        control_change_cost = R_deltau * (du**2)

        cost += state_cost + control_magnitude_cost + control_change_cost
        x_curr = x_next

    return cost

# ==========================================
# Setup both implementations
# ==========================================
params = {'a': 0.5, 'b': 0.025, 'd': 0.005, 'g': 0.5}
dt = 0.1
N = 10
Q = np.diag([1.0, 1.0])
Ru = 0.5
R_deltau = 0.5
xref = np.array([100.0, 20.0])
x0 = np.array([80.0, 25.0])
u_prev = 0.0

# Create MPC controller
pop_ode = PopulationDynamics()
critical_point = torch.tensor([100.0, 20.0])

mpc_config = MPCConfig(
    prediction_horizon=N,
    dt=dt,
    Q=torch.tensor([1.0, 1.0]),
    Ru=Ru,
    R_deltau=R_deltau,
)

mpc = MPCController(
    ode=pop_ode,
    config=mpc_config,
    reference_state=critical_point,
    control_indices=[1],
)

# Test with same control sequences
print("=" * 60)
print("Direct Cost Function Comparison")
print("=" * 60)
print()

test_sequences = [
    ("All zeros", np.zeros(N)),
    ("All ones", np.ones(N)),
    ("All -5", np.full(N, -5.0)),
    ("Optimal from standalone", np.array([-10.1653, -9.5, -9.0, -8.5, -8.0, -7.5, -7.0, -6.5, -6.0, -5.5])),
]

for name, u_seq in test_sequences:
    cost_standalone = mpc_objective_standalone(u_seq.copy(), x0, N, dt, xref, Q, Ru, R_deltau, u_prev, params)
    cost_mpc = mpc._mpc_objective(u_seq.copy(), x0)

    diff = abs(cost_standalone - cost_mpc)

    print(f"{name}:")
    print(f"  Standalone: {cost_standalone:.6f}")
    print(f"  MPC.py:     {cost_mpc:.6f}")
    print(f"  Difference: {diff:.6f}")

    if diff > 0.001:
        print(f"  ❌ MISMATCH!")
    else:
        print(f"  ✓ Match")
    print()
