"""Compare stiff ODE solvers with native JAX implementation of HPA."""
import time
import torch
import jax
import jax.numpy as jnp
import diffrax
import numpy as np
from scipy.integrate import solve_ivp
from torchdiffeq import odeint
import sys
from pathlib import Path

# Add src and configs to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root))


def create_hpa_jax(fixed_params_np, stressor_val=2.0):
    """Create native JAX version of HPA ODE.

    Args:
        fixed_params_np: Fixed parameters as numpy array
        stressor_val: Stressor value (constant for this test)

    Returns:
        JAX ODE function
    """
    gamma_x1, gamma_x2, gamma_x3, gamma_P, gamma_A, KGR, nGR, KP, KA = fixed_params_np

    def hpa_ode_jax(t, x, args):
        """HPA ODE in JAX.

        Args:
            t: Time
            x: State [CRH, ACTH, Cortisol, P, A]
            args: Additional arguments (unused)

        Returns:
            dx/dt
        """
        # Unpack state
        x1, x2, x3, P, A = x[0], x[1], x[2], x[3], x[4]

        # Control inputs (all 1.0 for baseline)
        I1 = I2 = I3 = 1.0
        C1 = C2 = C3 = 1.0
        A1 = A2 = A3 = 1.0

        # Stressor
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


def test_pytorch_solver(ode, initial_state, t_span, solver='dopri5'):
    """Test PyTorch/torchdiffeq solver."""
    try:
        # Warm up
        _ = odeint(ode, initial_state, t_span, method=solver)

        # Time it
        start = time.time()
        traj = odeint(ode, initial_state, t_span, method=solver)
        elapsed = time.time() - start

        return {
            'solver': solver,
            'backend': 'pytorch',
            'time': elapsed,
            'success': True,
            'trajectory': traj.detach().cpu().numpy()
        }
    except Exception as e:
        return {
            'solver': solver,
            'backend': 'pytorch',
            'time': float('inf'),
            'success': False,
            'error': str(e)
        }


def test_diffrax_solver(ode_jax, initial_state, t_span, solver_name='Kvaerno5'):
    """Test JAX/Diffrax solver."""
    # Convert inputs to JAX
    y0 = jnp.array(initial_state.detach().cpu().numpy())
    t_eval = jnp.array(t_span.detach().cpu().numpy())
    t0 = float(t_eval[0])
    t1 = float(t_eval[-1])

    # Choose solver
    if solver_name == 'Kvaerno5':
        solver = diffrax.Kvaerno5()
    elif solver_name == 'Kvaerno3':
        solver = diffrax.Kvaerno3()
    elif solver_name == 'Kvaerno4':
        solver = diffrax.Kvaerno4()
    else:
        raise ValueError(f"Unknown solver: {solver_name}")

    # Step size controller
    stepsize_controller = diffrax.PIDController(rtol=1e-7, atol=1e-9)

    # Warm up (JIT compile)
    print(f"    JIT compiling...", end=' ', flush=True)
    try:
        _ = diffrax.diffeqsolve(
            diffrax.ODETerm(ode_jax),
            solver,
            t0=t0,
            t1=t1,
            dt0=0.1,
            y0=y0,
            saveat=diffrax.SaveAt(ts=t_eval),
            stepsize_controller=stepsize_controller,
        )
        print("done")
    except Exception as e:
        print(f"FAILED during warmup")
        return {
            'solver': solver_name,
            'backend': 'diffrax',
            'time': float('inf'),
            'success': False,
            'error': str(e)
        }

    # Time it
    try:
        start = time.time()
        sol = diffrax.diffeqsolve(
            diffrax.ODETerm(ode_jax),
            solver,
            t0=t0,
            t1=t1,
            dt0=0.1,
            y0=y0,
            saveat=diffrax.SaveAt(ts=t_eval),
            stepsize_controller=stepsize_controller,
        )
        elapsed = time.time() - start

        return {
            'solver': solver_name,
            'backend': 'diffrax',
            'time': elapsed,
            'success': True,
            'trajectory': np.array(sol.ys),
            'nsteps': sol.stats['num_steps']
        }
    except Exception as e:
        return {
            'solver': solver_name,
            'backend': 'diffrax',
            'time': float('inf'),
            'success': False,
            'error': str(e)
        }


def test_scipy_solver(ode, initial_state, t_span, method='BDF'):
    """Test scipy solver."""
    y0 = initial_state.detach().cpu().numpy()
    t_eval = t_span.detach().cpu().numpy()
    t_span_scipy = [t_eval[0], t_eval[-1]]

    def ode_func_numpy(t, y):
        y_torch = torch.tensor(y, dtype=torch.float32)
        t_torch = torch.tensor(t, dtype=torch.float32)
        dydt = ode(t_torch, y_torch)
        return dydt.detach().cpu().numpy()

    try:
        # Warm up
        _ = solve_ivp(ode_func_numpy, t_span_scipy, y0, method=method, t_eval=t_eval)

        # Time it
        start = time.time()
        sol = solve_ivp(ode_func_numpy, t_span_scipy, y0, method=method, t_eval=t_eval)
        elapsed = time.time() - start

        return {
            'solver': method,
            'backend': 'scipy',
            'time': elapsed,
            'success': sol.success,
            'trajectory': sol.y.T,
            'nfev': sol.nfev
        }
    except Exception as e:
        return {
            'solver': method,
            'backend': 'scipy',
            'time': float('inf'),
            'success': False,
            'error': str(e)
        }


def test_gradients_jax(ode_jax, initial_state, t_span, solver_name='Kvaerno5'):
    """Test if JAX/Diffrax solver supports gradients."""
    try:
        y0 = jnp.array(initial_state.detach().cpu().numpy())
        t_eval = jnp.array(t_span[:11].detach().cpu().numpy())  # Short trajectory
        t0 = float(t_eval[0])
        t1 = float(t_eval[-1])

        if solver_name == 'Kvaerno5':
            solver = diffrax.Kvaerno5()
        elif solver_name == 'Kvaerno3':
            solver = diffrax.Kvaerno3()
        else:
            solver = diffrax.Kvaerno4()

        stepsize_controller = diffrax.PIDController(rtol=1e-7, atol=1e-9)

        # Define function to differentiate
        def loss_fn(y0_param):
            sol = diffrax.diffeqsolve(
                diffrax.ODETerm(ode_jax),
                solver,
                t0=t0,
                t1=t1,
                dt0=0.1,
                y0=y0_param,
                saveat=diffrax.SaveAt(ts=t_eval),
                stepsize_controller=stepsize_controller,
            )
            return jnp.sum(sol.ys[-1])

        # Compute gradient
        grad_fn = jax.grad(loss_fn)
        grad = grad_fn(y0)
        grad_norm = float(jnp.linalg.norm(grad))

        return True, grad_norm
    except Exception as e:
        return False, str(e)[:100]


def main():
    """Compare stiff ODE solvers."""
    print("="*80)
    print("Stiff ODE Solver Comparison: PyTorch vs JAX/Diffrax vs scipy")
    print("="*80)
    print()
    print("Testing HPA system (horizon=20 days, 101 time points)")
    print("Goal: Find a solver that is BOTH fast AND differentiable")
    print()

    # Import config
    import importlib
    config_module = importlib.import_module(f"configs.hpa_i1")
    config = config_module.ENV_CONFIG

    # Setup
    base_ode = config['create_base_ode']()
    initial_state = config['initial_state']
    time_horizon = 20.0
    n_reward_steps = 100
    t_span = torch.linspace(0, time_horizon, n_reward_steps + 1)

    # Get fixed params
    fixed_params_np = base_ode.fixed_params.detach().cpu().numpy()

    # Create native JAX ODE
    print("Creating native JAX HPA ODE...")
    ode_jax = create_hpa_jax(fixed_params_np, stressor_val=2.0)
    print()

    results = []

    # Test PyTorch solvers
    print("-"*80)
    print("PYTORCH/TORCHDIFFEQ SOLVERS")
    print("-"*80)
    for solver in ['dopri5', 'bosh3']:
        print(f"  {solver}...", end=' ', flush=True)
        result = test_pytorch_solver(base_ode, initial_state, t_span, solver)
        results.append(result)
        if result['success']:
            print(f"{result['time']:.3f}s")
        else:
            print(f"FAILED")

    # Test Diffrax solvers
    print("\n" + "-"*80)
    print("JAX/DIFFRAX SOLVERS (Implicit, Stiff-Aware)")
    print("-"*80)
    for solver in ['Kvaerno5', 'Kvaerno3']:
        print(f"  {solver}...")
        result = test_diffrax_solver(ode_jax, initial_state, t_span, solver)
        results.append(result)
        if result['success']:
            print(f"    Time: {result['time']:.3f}s, Steps: {result.get('nsteps', 'N/A')}")
        else:
            print(f"    FAILED: {result.get('error', 'Unknown')[:80]}...")

    # Test scipy solvers
    print("\n" + "-"*80)
    print("SCIPY SOLVERS (Implicit, but not differentiable)")
    print("-"*80)
    for method in ['BDF', 'Radau']:
        print(f"  {method}...", end=' ', flush=True)
        result = test_scipy_solver(base_ode, initial_state, t_span, method)
        results.append(result)
        if result['success']:
            print(f"{result['time']:.3f}s (nfev: {result.get('nfev', 'N/A')})")
        else:
            print(f"FAILED")

    # Test gradients for successful diffrax solvers
    print("\n" + "="*80)
    print("GRADIENT SUPPORT TEST (JAX/Diffrax only)")
    print("="*80)

    gradient_results = {}

    for solver in ['Kvaerno5', 'Kvaerno3']:
        # Check if solver succeeded
        solver_result = next((r for r in results if r['solver'] == solver and r['backend'] == 'diffrax'), None)
        if solver_result and solver_result['success']:
            print(f"\n  Testing {solver}...", end=' ', flush=True)
            supports_grad, grad_info = test_gradients_jax(ode_jax, initial_state, t_span, solver)
            gradient_results[f'diffrax-{solver}'] = (supports_grad, grad_info)
            status = "✓" if supports_grad else "✗"
            print(f"{status} (grad norm: {grad_info})")
        else:
            print(f"\n  {solver}: ✗ (solver failed)")
            gradient_results[f'diffrax-{solver}'] = (False, 'solver failed')

    # PyTorch gradient support (already known)
    gradient_results['pytorch-dopri5'] = (True, 'stable')
    gradient_results['pytorch-bosh3'] = (True, 'inf warning')

    # Summary table
    print("\n" + "="*80)
    print("COMPLETE COMPARISON")
    print("="*80)
    print(f"\n{'Solver':<20} {'Backend':<12} {'Time (s)':<12} {'Gradients':<15} {'Training?':<10}")
    print("-"*80)

    for result in results:
        if result['success']:
            solver_key = f"{result['backend']}-{result['solver']}"
            has_grad = gradient_results.get(solver_key, (False, 0))[0]
            grad_str = "✓" if has_grad else "✗"
            train_str = "YES" if has_grad else "NO"

            print(f"{result['solver']:<20} {result['backend']:<12} {result['time']:>8.3f}    {grad_str:<15} {train_str:<10}")
        else:
            print(f"{result['solver']:<20} {result['backend']:<12} {'FAILED':<12} {'-':<15} {'-':<10}")

    # Find best trainable solver
    print("\n" + "="*80)
    print("RECOMMENDATION FOR TRAINING")
    print("="*80)

    trainable = []
    for result in results:
        if result['success']:
            solver_key = f"{result['backend']}-{result['solver']}"
            has_grad = gradient_results.get(solver_key, (False, 0))[0]
            if has_grad:
                trainable.append(result)

    if trainable:
        fastest = min(trainable, key=lambda x: x['time'])
        print(f"\nFASTEST TRAINABLE SOLVER: {fastest['solver']} ({fastest['backend']})")
        print(f"  Time: {fastest['time']:.3f}s")

        # Compare to dopri5
        dopri5_result = next((r for r in results if r['solver'] == 'dopri5'), None)
        if dopri5_result and dopri5_result['success']:
            speedup = dopri5_result['time'] / fastest['time']
            print(f"  Speedup vs dopri5: {speedup:.2f}x")

        print(f"\n{'='*80}")
        if fastest['backend'] == 'diffrax':
            print("✓ JAX/Diffrax implicit solvers are BOTH fast AND differentiable!")
            print("  Recommendation: Switch to JAX for significant speedup in HPA training")
        else:
            print("  PyTorch remains the best option for now")
    else:
        print("\nNo trainable solvers found!")


if __name__ == "__main__":
    main()
