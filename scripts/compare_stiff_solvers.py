"""Compare stiff ODE solvers: PyTorch, JAX/Diffrax, and scipy."""
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


def convert_ode_to_jax(torch_ode):
    """Convert PyTorch ODE to JAX-compatible function.

    Args:
        torch_ode: PyTorch ODE instance

    Returns:
        JAX-compatible ODE function
    """
    # Get fixed params as numpy
    if hasattr(torch_ode, 'fixed_params') and torch_ode.fixed_params is not None:
        fixed_params_np = torch_ode.fixed_params.detach().cpu().numpy()
    else:
        fixed_params_np = None

    # Create JAX version
    def ode_jax(t, y, args):
        """JAX ODE function."""
        # Convert to torch
        t_torch = torch.tensor(float(t), dtype=torch.float32)
        y_torch = torch.tensor(np.array(y), dtype=torch.float32)

        # Call original ODE
        with torch.no_grad():
            dydt_torch = torch_ode(t_torch, y_torch)

        # Convert back to jax
        dydt_jax = jnp.array(dydt_torch.detach().cpu().numpy())
        return dydt_jax

    return ode_jax


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
    print(f"    JIT compiling {solver_name}...", end=' ', flush=True)
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
        print(f"FAILED: {e}")
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


def test_gradients_pytorch(ode, initial_state, t_span, solver):
    """Test if PyTorch solver supports gradients."""
    try:
        # Make params require grad
        if hasattr(ode, 'fixed_params') and ode.fixed_params is not None:
            ode.fixed_params = ode.fixed_params.clone().requires_grad_(True)
            param = ode.fixed_params
        else:
            initial_state = initial_state.clone().requires_grad_(True)
            param = initial_state

        # Forward
        traj = odeint(ode, initial_state, t_span[:11], method=solver)  # Short trajectory
        loss = traj[-1].sum()

        # Backward
        loss.backward()

        if param.grad is not None and param.grad.abs().sum() > 0:
            return True, param.grad.norm().item()
        return False, 0.0
    except Exception as e:
        return False, str(e)


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
        return False, str(e)


def main():
    """Compare stiff ODE solvers across backends."""
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

    # Convert ODE to JAX
    print("Converting ODE to JAX...")
    ode_jax = convert_ode_to_jax(base_ode)
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
    for solver in ['Kvaerno5', 'Kvaerno3', 'Kvaerno4']:
        print(f"  {solver}...")
        result = test_diffrax_solver(ode_jax, initial_state, t_span, solver)
        results.append(result)
        if result['success']:
            print(f"    Time: {result['time']:.3f}s, Steps: {result.get('nsteps', 'N/A')}")
        else:
            print(f"    FAILED: {result.get('error', 'Unknown')}")

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

    # Test gradients
    print("\n" + "="*80)
    print("GRADIENT SUPPORT TEST")
    print("="*80)

    gradient_results = {}

    # PyTorch
    print("\nPyTorch solvers:")
    for solver in ['dopri5', 'bosh3']:
        supports_grad, grad_info = test_gradients_pytorch(
            config['create_base_ode'](),
            initial_state,
            t_span,
            solver
        )
        gradient_results[f'pytorch-{solver}'] = (supports_grad, grad_info)
        status = "✓" if supports_grad else "✗"
        print(f"  {solver}: {status} (grad norm: {grad_info})")

    # Diffrax
    print("\nDiffrax solvers:")
    for solver in ['Kvaerno5', 'Kvaerno3']:
        supports_grad, grad_info = test_gradients_jax(ode_jax, initial_state, t_span, solver)
        gradient_results[f'diffrax-{solver}'] = (supports_grad, grad_info)
        status = "✓" if supports_grad else "✗"
        print(f"  {solver}: {status} (grad norm: {grad_info})")

    # Summary table
    print("\n" + "="*80)
    print("COMPLETE COMPARISON")
    print("="*80)
    print(f"\n{'Solver':<20} {'Backend':<12} {'Time (s)':<12} {'Gradients':<12} {'Training?':<10}")
    print("-"*80)

    for result in results:
        if result['success']:
            solver_key = f"{result['backend']}-{result['solver']}"
            has_grad = gradient_results.get(solver_key, (False, 0))[0]
            grad_str = "✓" if has_grad else "✗"
            train_str = "YES" if has_grad else "NO"

            print(f"{result['solver']:<20} {result['backend']:<12} {result['time']:>8.3f}    {grad_str:<12} {train_str:<10}")
        else:
            print(f"{result['solver']:<20} {result['backend']:<12} {'FAILED':<12} {'-':<12} {'-':<10}")

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
            print("  Recommendation: Switch to JAX for HPA training")
        else:
            print("  PyTorch remains the best option")
    else:
        print("\nNo trainable solvers found!")


if __name__ == "__main__":
    main()
