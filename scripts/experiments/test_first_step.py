"""Test just the first optimization step to see why standalone works."""
import numpy as np
from scipy.optimize import minimize

# Standalone cost function
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

# Setup
params = {'a': 0.5, 'b': 0.025, 'd': 0.005, 'g': 0.5}
dt = 0.1
N = 10
Q = np.diag([1.0, 1.0])
Ru = 0.5
R_deltau = 0.5
xref = np.array([100.0, 20.0])
x_current = np.array([70.0, 15.0])
u_guess = np.zeros(N)
u_prev = 0.0
bounds = [(-20, 20) for _ in range(N)]

print("=" * 60)
print("Testing First MPC Step (Standalone Implementation)")
print("=" * 60)
print()

print(f"Initial state: {x_current}")
print(f"Target state: {xref}")
print(f"Initial guess: all zeros")
print()

# Run optimization exactly as standalone script does
res = minimize(
    mpc_objective_standalone,
    u_guess,
    args=(x_current, N, dt, xref, Q, Ru, R_deltau, u_prev, params),
    method='SLSQP',
    bounds=bounds,
    options={'ftol': 1e-3, 'disp': True}
)

print()
print(f"Optimal control: u[0] = {res.x[0]:.4f}")
print(f"Optimal cost: {res.fun:.4f}")
print(f"Success: {res.success}")
print(f"Iterations: {res.nit}")
print(f"Function evaluations: {res.nfev}")
print()

# Apply control and see where we end up
x_next = rk4_step_standalone(x_current, res.x[0], dt, params)
print(f"Next state after one step: {x_next}")
