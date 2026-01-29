"""Test if Diffrax works at all with a simple ODE."""
import jax
import jax.numpy as jnp
import diffrax

# Simple exponential decay: dy/dt = -0.1*y
def simple_ode(t, y, args):
    return -0.1 * y

y0 = jnp.array([1.0])
t_span = jnp.linspace(0, 10, 11)

print("Testing Diffrax with simple ODE (dy/dt = -0.1*y)...")
print()

# Try explicit solver first
print("1. Testing explicit solver (Dopri5)...")
try:
    solver = diffrax.Dopri5()
    sol = diffrax.diffeqsolve(
        diffrax.ODETerm(simple_ode),
        solver,
        t0=0.0,
        t1=10.0,
        dt0=0.1,
        y0=y0,
        saveat=diffrax.SaveAt(ts=t_span),
    )
    print(f"   ✓ Success! Final value: {sol.ys[-1]}")
except Exception as e:
    print(f"   ✗ Failed: {e}")

# Try implicit solver
print("\n2. Testing implicit solver (Kvaerno5)...")
try:
    solver = diffrax.Kvaerno5()
    stepsize_controller = diffrax.PIDController(rtol=1e-7, atol=1e-9)
    sol = diffrax.diffeqsolve(
        diffrax.ODETerm(simple_ode),
        solver,
        t0=0.0,
        t1=10.0,
        dt0=0.1,
        y0=y0,
        saveat=diffrax.SaveAt(ts=t_span),
        stepsize_controller=stepsize_controller,
        max_steps=100000,  # Increase max_steps
    )
    print(f"   ✓ Success! Final value: {sol.ys[-1]}")
    print(f"     Steps taken: {sol.stats['num_steps']}")
except Exception as e:
    print(f"   ✗ Failed: {e}")

# Test gradients
print("\n3. Testing gradients with Kvaerno5...")
try:
    solver = diffrax.Kvaerno5()
    stepsize_controller = diffrax.PIDController(rtol=1e-7, atol=1e-9)

    def loss_fn(y0_param):
        sol = diffrax.diffeqsolve(
            diffrax.ODETerm(simple_ode),
            solver,
            t0=0.0,
            t1=10.0,
            dt0=0.1,
            y0=y0_param,
            saveat=diffrax.SaveAt(ts=t_span),
            stepsize_controller=stepsize_controller,
            max_steps=100000,  # Increase max_steps
        )
        return jnp.sum(sol.ys[-1])

    grad_fn = jax.grad(loss_fn)
    grad = grad_fn(y0)
    print(f"   ✓ Gradients work! Gradient: {grad}")
except Exception as e:
    print(f"   ✗ Failed: {e}")

print("\n" + "="*60)
print("CONCLUSION")
print("="*60)
print("If all tests passed, Diffrax is working correctly.")
print("The issue might be with the HPA ODE specifically.")
