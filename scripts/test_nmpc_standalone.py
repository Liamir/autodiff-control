"""Standalone NMPC test using scipy.optimize directly."""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# ==========================================
# 1. THE MODEL (Assumed Known)
# ==========================================

def lotka_volterra(x, u, params):
    """
    State derivatives: dx/dt = f(x, u)
    x: [Prey, Predator]
    u: Control input (forcing on Prey)
    """
    a, b, d, g = params['a'], params['b'], params['d'], params['g']
    x1, x2 = x

    # Dynamics from your MATLAB code
    dx1 = a*x1 - b*x1*x2
    dx2 = d*x1*x2 - g*x2 + u

    return np.array([dx1, dx2])

def rk4_step(x, u, dt, params):
    """
    Explicit Runge-Kutta 4 discrete stepper.
    Essential for MPC predictions inside the optimization loop
    because it is faster and more stable than ode45 for fixed steps.
    """
    k1 = lotka_volterra(x, u, params)
    k2 = lotka_volterra(x + 0.5*dt*k1, u, params)
    k3 = lotka_volterra(x + 0.5*dt*k2, u, params)
    k4 = lotka_volterra(x + dt*k3, u, params)
    return x + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

# ==========================================
# 2. MPC COST FUNCTION
# ==========================================

def mpc_objective(u_seq, x0, N, dt, xref, Q, Ru, R_deltau, u_prev, params):
    """
    Calculates the cost J for a candidate control sequence u_seq.
    J = Sum( (x - xref)^T Q (x - xref) + Ru * u^2 + R_deltau * (u - u_prev)^2 )
    """
    cost = 0.0
    x_curr = x0.copy()

    # Reshape input because 'minimize' flattens it
    u_seq = u_seq.reshape(N)

    for k in range(N):
        u_k = u_seq[k]

        # Predict next state using our internal model
        x_next = rk4_step(x_curr, u_k, dt, params)

        # Calculate State Error Cost
        error = x_next - xref
        state_cost = error.T @ Q @ error

        # Calculate Control Magnitude Cost
        control_magnitude_cost = Ru * (u_k**2)

        # Calculate Control Rate-of-Change Cost
        if k == 0:
            du = u_k - u_prev
        else:
            du = u_k - u_seq[k-1]
        control_change_cost = R_deltau * (du**2)

        cost += state_cost + control_magnitude_cost + control_change_cost

        # Update for next step
        x_curr = x_next

    return cost

# ==========================================
# 3. MAIN SIMULATION LOOP
# ==========================================

# Parameters
params = {'a': 0.5, 'b': 0.025, 'd': 0.005, 'g': 0.5}
dt = 0.1
T_final = 20  # Simulate for 20 time units
time_steps = int(T_final / dt)

# MPC Tuning (Paper Parameters)
N = 10                  # Horizon length (m_p = m_c = 10)
Q = np.diag([1.0, 1.0]) # Q = I_2x2
Ru = 0.5                # Control magnitude penalty
R_deltau = 0.5          # Control rate-of-change penalty
xref = np.array([params['g']/params['d'], params['a']/params['b']]) # [100, 20] Target

# Initial Setup
x_current = np.array([70.0, 15.0]) # Initial state
u_guess = np.zeros(N)              # Initial guess for the optimizer
u_prev = 0.0                       # Previous control for rate-of-change penalty

# History storage
x_history = [x_current]
u_history = []
t_history = [0]

print("Starting MPC Loop...")
print(f"Target state: Prey={xref[0]:.1f}, Predator={xref[1]:.1f}")
print(f"Initial state: Prey={x_current[0]:.1f}, Predator={x_current[1]:.1f}")
print()

for i in range(time_steps):

    # --- A. OPTIMIZATION STEP ---
    # We define bounds to ensure realistic control inputs (u ∈ [-20, 20] from paper)
    bounds = [(-20, 20) for _ in range(N)]

    # Minimize the cost function
    res = minimize(
        mpc_objective,
        u_guess,
        args=(x_current, N, dt, xref, Q, Ru, R_deltau, u_prev, params),
        method='SLSQP',       # SQP is standard for nonlinear MPC
        bounds=bounds,
        options={'ftol': 1e-3, 'disp': False}
    )

    u_optimal_sequence = res.x
    u_control = u_optimal_sequence[0] # Receding Horizon: Take only the first step

    # --- B. APPLY CONTROL ---
    # Apply u_control to the "Real" Plant (which is just our model here)
    x_next_real = rk4_step(x_current, u_control, dt, params)

    # --- C. UPDATE & STORE ---
    x_current = x_next_real

    x_history.append(x_current)
    u_history.append(u_control)
    t_history.append((i+1)*dt)

    # --- D. WARM START ---
    # Shift the found solution to be the guess for the next step.
    # This significantly speeds up convergence.
    u_prev = u_control  # Update previous control
    u_guess = np.roll(u_optimal_sequence, -1)
    u_guess[-1] = u_optimal_sequence[-1] # Duplicate last element

    # Print progress
    if (i+1) % 50 == 0:
        print(f"Step {i+1}/{time_steps}: Prey={x_current[0]:.2f}, Predator={x_current[1]:.2f}, u={u_control:.2f}")

print()
print("MPC Loop Complete!")
print(f"Final state: Prey={x_current[0]:.2f}, Predator={x_current[1]:.2f}")
print(f"Target state: Prey={xref[0]:.1f}, Predator={xref[1]:.1f}")

# ==========================================
# 4. VISUALIZATION
# ==========================================
x_history = np.array(x_history)
u_history = np.array(u_history)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

# State Plot
ax1.plot(t_history, x_history[:, 0], label='Prey (x1)', linewidth=2)
ax1.plot(t_history, x_history[:, 1], label='Predator (x2)', linewidth=2)
ax1.axhline(xref[0], color='r', linestyle='--', alpha=0.5, label='Ref x1')
ax1.axhline(xref[1], color='g', linestyle='--', alpha=0.5, label='Ref x2')
ax1.set_ylabel('Population')
ax1.set_title('MPC Control of Lotka-Volterra System')
ax1.legend()
ax1.grid(True)
ax1.spines['right'].set_visible(False)
ax1.spines['top'].set_visible(False)

# Control Input Plot
ax2.plot(t_history[:-1], u_history, 'k-', label='Control Input (u)')
ax2.set_ylabel('Control Input')
ax2.set_xlabel('Time')
ax2.legend()
ax2.grid(True)
ax2.spines['right'].set_visible(False)
ax2.spines['top'].set_visible(False)

plt.tight_layout()
plt.savefig('plots/nmpc_standalone_test.pdf', bbox_inches='tight')
print()
print("Saved plot to: plots/nmpc_standalone_test.pdf")
plt.show()
