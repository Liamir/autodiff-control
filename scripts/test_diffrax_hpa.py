"""Test Diffrax with HPA ODE (properly configured)."""
import jax
import jax.numpy as jnp
import diffrax
import time
import sys
from pathlib import Path

# Add src and configs to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root))


def create_hpa_jax(fixed_params_np, stressor_val=2.0):
    """Create native JAX version of HPA ODE."""
    gamma_x1, gamma_x2, gamma_x3, gamma_P, gamma_A, KGR, nGR, KP, KA = fixed_params_np

    def hpa_ode_jax(t, x, args):
        """HPA ODE in JAX."""
        # Unpack state
        x1, x2, x3, P, A = x[0], x[1], x[2], x[3], x[4]

        # Control inputs (all 1.0 for baseline)
        I1 = I2 = I3 = 1.0
        C1 = C2 = C3 = 1.0
        A1 = A2 = A3 = 1.0
        u = stressor_val

        # Receptor functions
        def MR(x_val):
            return 1.0 / x_val

        def GR(x_val):
            return 1.0 / (jnp.pow(x_val / KGR, nGR) + 1.0)

        # Effective cortisol
        x3_eff = C3 * x3

        # Derivatives
        dx1_dt = gamma_x1 * (I1 * u * MR(x3_eff) * GR(x3_eff) - A1 * x1)
        dx2_dt = gamma_x2 * (I2 * (C1 * x1) * P * GR(x3_eff) - A2 * x2)
        dx3_dt = gamma_x3 * (I3 * (C2 * x2) * A - A3 * x3)
        dP_dt = gamma_P * P * ((C1 * x1) * (1.0 - P / KP) - 1.0)
        dA_dt = gamma_A * A * ((C2 * x2) * (1.0 - A / KA) - 1.0)

        return jnp.array([dx1_dt, dx2_dt, dx3_dt, dP_dt, dA_dt])

    return hpa_ode_jax


def test_diffrax_hpa(solver_name, rtol, atol):
    """Test a Diffrax solver with HPA."""
    # Import config
    import importlib
    config_module = importlib.import_module(f"configs.hpa_i1")
    config = config_module.ENV_CONFIG

    # Setup
    import torch
    base_ode = config['create_base_ode']()
    initial_state = config['initial_state']
    time_horizon = 20.0
    n_reward_steps = 100

    # Get fixed params and create JAX ODE
    fixed_params_np = base_ode.fixed_params.detach().cpu().numpy()
    ode_jax = create_hpa_jax(fixed_params_np, stressor_val=2.0)

    # Convert to JAX
    y0 = jnp.array(initial_state.detach().cpu().numpy())
    t_span = jnp.linspace(0, time_horizon, n_reward_steps + 1)

    # Choose solver
    if solver_name == 'Kvaerno5':
        solver = diffrax.Kvaerno5()
    elif solver_name == 'Kvaerno3':
        solver = diffrax.Kvaerno3()
    elif solver_name == 'Kvaerno4':
        solver = diffrax.Kvaerno4()
    elif solver_name == 'Dopri5':
        solver = diffrax.Dopri5()
    else:
        raise ValueError(f"Unknown solver: {solver_name}")

    # Configure stepsize controller
    stepsize_controller = diffrax.PIDController(rtol=rtol, atol=atol)

    print(f"\n{solver_name} (rtol={rtol}, atol={atol}):")
    print(f"  JIT compiling...", end=' ', flush=True)

    try:
        # Warm up (JIT compile)
        _ = diffrax.diffeqsolve(
            diffrax.ODETerm(ode_jax),
            solver,
            t0=0.0,
            t1=time_horizon,
            dt0=0.1,
            y0=y0,
            saveat=diffrax.SaveAt(ts=t_span),
            stepsize_controller=stepsize_controller,
            max_steps=100000,
        )
        print("done")

        # Time it
        print(f"  Timing...", end=' ', flush=True)
        start = time.time()
        sol = diffrax.diffeqsolve(
            diffrax.ODETerm(ode_jax),
            solver,
            t0=0.0,
            t1=time_horizon,
            dt0=0.1,
            y0=y0,
            saveat=diffrax.SaveAt(ts=t_span),
            stepsize_controller=stepsize_controller,
            max_steps=100000,
        )
        elapsed = time.time() - start
        print(f"done")

        print(f"  ✓ Success!")
        print(f"    Time: {elapsed:.3f}s")
        print(f"    Steps: {sol.stats['num_steps']}")
        print(f"    Final state: {sol.ys[-1]}")

        return {
            'solver': solver_name,
            'time': elapsed,
            'steps': sol.stats['num_steps'],
            'success': True,
            'trajectory': sol.ys,
        }

    except Exception as e:
        print(f"FAILED")
        print(f"  ✗ Error: {str(e)[:150]}")
        return {
            'solver': solver_name,
            'time': float('inf'),
            'success': False,
            'error': str(e)
        }


def main():
    """Test Diffrax with HPA."""
    print("="*70)
    print("DIFFRAX with HPA - Properly Configured")
    print("="*70)
    print("Horizon: 20 days, 101 time points")
    print()

    results = []

    # Test explicit solver baseline
    print("-"*70)
    print("BASELINE: Explicit Solver")
    print("-"*70)
    result = test_diffrax_hpa('Dopri5', rtol=1e-7, atol=1e-9)
    results.append(result)

    # Test implicit solvers with different tolerances
    print("\n" + "-"*70)
    print("IMPLICIT SOLVERS: Kvaerno Family")
    print("-"*70)

    # Try different tolerance levels
    tolerance_configs = [
        ('Relaxed', 1e-3, 1e-3),
        ('Medium', 1e-5, 1e-7),
        ('Tight', 1e-7, 1e-9),
    ]

    for tol_name, rtol, atol in tolerance_configs:
        print(f"\n{tol_name} tolerances (rtol={rtol}, atol={atol}):")

        for solver_name in ['Kvaerno5', 'Kvaerno3']:
            result = test_diffrax_hpa(solver_name, rtol, atol)
            results.append(result)

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"\n{'Solver':<15} {'Tolerances':<15} {'Time (s)':<12} {'Steps':<10} {'Status':<10}")
    print("-"*70)

    for r in results:
        if r['success']:
            # Extract tolerance from result (approximation for display)
            tol_str = "default" if r['solver'] == 'Dopri5' else "varies"
            print(f"{r['solver']:<15} {tol_str:<15} {r['time']:>8.3f}    {r.get('steps', 'N/A'):<10} {'✓':<10}")
        else:
            print(f"{r['solver']:<15} {'N/A':<15} {'FAILED':<12} {'N/A':<10} {'✗':<10}")

    # Find fastest successful solver
    successful = [r for r in results if r['success']]
    if successful:
        fastest = min(successful, key=lambda x: x['time'])
        dopri5 = next((r for r in results if r['solver'] == 'Dopri5'), None)

        print("\n" + "="*70)
        print("BEST RESULT")
        print("="*70)
        print(f"\nFastest: {fastest['solver']}")
        print(f"  Time: {fastest['time']:.3f}s")
        print(f"  Steps: {fastest['steps']}")

        if dopri5 and dopri5['success']:
            speedup = dopri5['time'] / fastest['time']
            print(f"  Speedup vs Dopri5: {speedup:.2f}x")


if __name__ == "__main__":
    main()
