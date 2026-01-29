"""Debug Diffrax implicit solvers step by step."""
import jax
import jax.numpy as jnp
import diffrax
import time

print("="*70)
print("DEBUGGING DIFFRAX IMPLICIT SOLVERS")
print("="*70)

# Simple exponential decay: dy/dt = -0.1*y
def simple_ode(t, y, args):
    return -0.1 * y

y0 = jnp.array([1.0])
t0, t1 = 0.0, 10.0
t_span = jnp.linspace(t0, t1, 11)

print("\nTest ODE: dy/dt = -0.1*y, y(0) = 1.0")
print(f"Expected solution: y(10) ≈ 0.368")
print()

# Test 1: Explicit solver as baseline
print("-"*70)
print("TEST 1: Explicit solver (Dopri5) - Should work")
print("-"*70)
try:
    sol = diffrax.diffeqsolve(
        diffrax.ODETerm(simple_ode),
        diffrax.Dopri5(),
        t0=t0,
        t1=t1,
        dt0=0.1,
        y0=y0,
        saveat=diffrax.SaveAt(ts=t_span),
    )
    print(f"✓ Success!")
    print(f"  Final value: {sol.ys[-1][0]:.6f}")
    print(f"  Steps: {sol.stats['num_steps']}")
except Exception as e:
    print(f"✗ Failed: {str(e)[:100]}")

# Test 2: Kvaerno5 with default settings
print("\n" + "-"*70)
print("TEST 2: Kvaerno5 with default settings")
print("-"*70)
try:
    sol = diffrax.diffeqsolve(
        diffrax.ODETerm(simple_ode),
        diffrax.Kvaerno5(),
        t0=t0,
        t1=t1,
        dt0=0.1,
        y0=y0,
        saveat=diffrax.SaveAt(ts=t_span),
    )
    print(f"✓ Success!")
    print(f"  Final value: {sol.ys[-1][0]:.6f}")
    print(f"  Steps: {sol.stats['num_steps']}")
except Exception as e:
    print(f"✗ Failed: {str(e)[:200]}")

# Test 3: Kvaerno5 with fixed step size (no adaptive control)
print("\n" + "-"*70)
print("TEST 3: Kvaerno5 with FIXED step size (no adaptive)")
print("-"*70)
try:
    sol = diffrax.diffeqsolve(
        diffrax.ODETerm(simple_ode),
        diffrax.Kvaerno5(),
        t0=t0,
        t1=t1,
        dt0=0.1,
        y0=y0,
        saveat=diffrax.SaveAt(ts=t_span),
        stepsize_controller=diffrax.ConstantStepSize(),
    )
    print(f"✓ Success!")
    print(f"  Final value: {sol.ys[-1][0]:.6f}")
    print(f"  Steps: {sol.stats['num_steps']}")
except Exception as e:
    print(f"✗ Failed: {str(e)[:200]}")

# Test 4: Kvaerno5 with very relaxed tolerances
print("\n" + "-"*70)
print("TEST 4: Kvaerno5 with RELAXED tolerances")
print("-"*70)
try:
    sol = diffrax.diffeqsolve(
        diffrax.ODETerm(simple_ode),
        diffrax.Kvaerno5(),
        t0=t0,
        t1=t1,
        dt0=0.1,
        y0=y0,
        saveat=diffrax.SaveAt(ts=t_span),
        stepsize_controller=diffrax.PIDController(rtol=1e-3, atol=1e-3),
        max_steps=100000,
    )
    print(f"✓ Success!")
    print(f"  Final value: {sol.ys[-1][0]:.6f}")
    print(f"  Steps: {sol.stats['num_steps']}")
except Exception as e:
    print(f"✗ Failed: {str(e)[:200]}")

# Test 5: Try other implicit solvers
print("\n" + "-"*70)
print("TEST 5: Other implicit solvers")
print("-"*70)

implicit_solvers = [
    ('Kvaerno3', diffrax.Kvaerno3()),
    ('Kvaerno4', diffrax.Kvaerno4()),
]

for name, solver in implicit_solvers:
    print(f"\n  {name}:", end=" ")
    try:
        sol = diffrax.diffeqsolve(
            diffrax.ODETerm(simple_ode),
            solver,
            t0=t0,
            t1=t1,
            dt0=0.1,
            y0=y0,
            saveat=diffrax.SaveAt(ts=t_span),
            stepsize_controller=diffrax.ConstantStepSize(),
        )
        print(f"✓ y(10)={sol.ys[-1][0]:.6f}, steps={sol.stats['num_steps']}")
    except Exception as e:
        print(f"✗ {str(e)[:80]}")

# Test 6: Semi-implicit solver
print("\n" + "-"*70)
print("TEST 6: Semi-implicit solver (ImplicitEuler)")
print("-"*70)
try:
    sol = diffrax.diffeqsolve(
        diffrax.ODETerm(simple_ode),
        diffrax.ImplicitEuler(),
        t0=t0,
        t1=t1,
        dt0=0.1,
        y0=y0,
        saveat=diffrax.SaveAt(ts=t_span),
        stepsize_controller=diffrax.ConstantStepSize(),
    )
    print(f"✓ Success!")
    print(f"  Final value: {sol.ys[-1][0]:.6f}")
    print(f"  Steps: {sol.stats['num_steps']}")
except Exception as e:
    print(f"✗ Failed: {str(e)[:200]}")

# Test 7: Test with HalfSolver (adaptive switching)
print("\n" + "-"*70)
print("TEST 7: HalfSolver (adaptive implicit/explicit)")
print("-"*70)
try:
    sol = diffrax.diffeqsolve(
        diffrax.ODETerm(simple_ode),
        diffrax.HalfSolver(diffrax.Kvaerno5()),
        t0=t0,
        t1=t1,
        dt0=0.1,
        y0=y0,
        saveat=diffrax.SaveAt(ts=t_span),
        stepsize_controller=diffrax.ConstantStepSize(),
    )
    print(f"✓ Success!")
    print(f"  Final value: {sol.ys[-1][0]:.6f}")
    print(f"  Steps: {sol.stats['num_steps']}")
except Exception as e:
    print(f"✗ Failed: {str(e)[:200]}")

print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print("""
If implicit solvers are failing, possible issues:
1. Diffrax version incompatibility
2. JAX version incompatibility
3. Implicit solvers need special configuration
4. Bug in Diffrax implicit solver implementation

Next steps:
- Check Diffrax version and examples
- Try downgrading/upgrading Diffrax
- Look at Diffrax documentation for implicit solver usage
""")

# Print versions
import diffrax as dfx
print(f"\nVersions:")
print(f"  Diffrax: {dfx.__version__}")
print(f"  JAX: {jax.__version__}")
