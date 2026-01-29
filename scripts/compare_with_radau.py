"""Compare torchdiffeq solvers with scipy's Radau (implicit, good for stiff ODEs)."""
import torch
import time
import importlib
import sys
from pathlib import Path
from torchdiffeq import odeint
import numpy as np
from scipy.integrate import solve_ivp

# Add src and configs to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root))


def test_torchdiffeq_solver(ode, initial_state, t_span, solver):
    """Test a torchdiffeq solver.

    Args:
        ode: ODE instance
        initial_state: Initial state tensor
        t_span: Time points tensor
        solver: Solver name

    Returns:
        dict with timing info
    """
    try:
        # Warm up
        _ = odeint(ode, initial_state, t_span, method=solver)

        # Time it
        start = time.time()
        traj = odeint(ode, initial_state, t_span, method=solver)
        elapsed = time.time() - start

        return {
            'solver': solver,
            'backend': 'torchdiffeq',
            'time': elapsed,
            'success': True,
            'trajectory': traj
        }
    except Exception as e:
        return {
            'solver': solver,
            'backend': 'torchdiffeq',
            'time': float('inf'),
            'success': False,
            'error': str(e)
        }


def test_scipy_solver(ode, initial_state, t_span, solver):
    """Test a scipy solver (Radau).

    Args:
        ode: ODE instance (PyTorch-based)
        initial_state: Initial state tensor
        t_span: Time points tensor
        solver: Solver name ('Radau', 'BDF', 'LSODA')

    Returns:
        dict with timing info
    """
    # Convert to numpy
    y0 = initial_state.detach().cpu().numpy()
    t_eval = t_span.detach().cpu().numpy()
    t_span_scipy = [t_eval[0], t_eval[-1]]

    # Create numpy-compatible ODE function
    def ode_func_numpy(t, y):
        """Wrapper to call PyTorch ODE with numpy arrays."""
        y_torch = torch.tensor(y, dtype=torch.float32)
        t_torch = torch.tensor(t, dtype=torch.float32)

        # Call ODE
        dydt = ode(t_torch, y_torch)

        # Convert back to numpy
        return dydt.detach().cpu().numpy()

    try:
        # Warm up
        _ = solve_ivp(ode_func_numpy, t_span_scipy, y0, method=solver, t_eval=t_eval)

        # Time it
        start = time.time()
        sol = solve_ivp(ode_func_numpy, t_span_scipy, y0, method=solver, t_eval=t_eval)
        elapsed = time.time() - start

        # Convert trajectory back to torch for comparison
        trajectory = torch.tensor(sol.y.T, dtype=torch.float32)

        return {
            'solver': solver,
            'backend': 'scipy',
            'time': elapsed,
            'success': sol.success,
            'trajectory': trajectory,
            'nfev': sol.nfev,  # Number of function evaluations
            'message': sol.message
        }
    except Exception as e:
        return {
            'solver': solver,
            'backend': 'scipy',
            'time': float('inf'),
            'success': False,
            'error': str(e)
        }


def compare_trajectories(traj1, traj2, names):
    """Compare two trajectories and report differences.

    Args:
        traj1: First trajectory tensor
        traj2: Second trajectory tensor
        names: Tuple of (name1, name2) for display

    Returns:
        dict with error metrics
    """
    # Convert to numpy for comparison
    t1 = traj1.detach().cpu().numpy()
    t2 = traj2.detach().cpu().numpy()

    abs_error = np.abs(t1 - t2)
    rel_error = abs_error / (np.abs(t1) + 1e-8)

    return {
        'max_abs_error': abs_error.max(),
        'mean_abs_error': abs_error.mean(),
        'max_rel_error': rel_error.max(),
        'mean_rel_error': rel_error.mean()
    }


def main():
    """Compare bosh3, adaptive_heun, and Radau for HPA."""
    print("="*70)
    print("Solver Comparison: torchdiffeq vs scipy.Radau")
    print("="*70)
    print("Environment: HPA_i1")
    print("Time horizon: 20 days")
    print("Time steps: 101")
    print()
    print("NOTE: Radau is an implicit solver designed for stiff ODEs")
    print("      It cannot be used for training (no PyTorch autodiff)")
    print("      But useful to compare performance and accuracy")
    print()

    # Import config
    config_module = importlib.import_module(f"configs.hpa_i1")
    config = config_module.ENV_CONFIG

    # Create ODE
    base_ode = config['create_base_ode']()
    initial_state = config['initial_state']
    time_horizon = 20.0
    n_reward_steps = 100
    t_span = torch.linspace(0, time_horizon, n_reward_steps + 1)

    # Test solvers
    solvers_torch = ['bosh3', 'adaptive_heun', 'dopri5', 'implicit_adams']
    solvers_scipy = ['Radau', 'BDF', 'LSODA']

    results = []

    # Test torchdiffeq solvers
    print("Testing torchdiffeq solvers:")
    for solver in solvers_torch:
        print(f"  {solver}...", end=' ', flush=True)
        result = test_torchdiffeq_solver(base_ode, initial_state, t_span, solver)
        results.append(result)
        if result['success']:
            print(f"{result['time']:.3f}s")
        else:
            print(f"FAILED: {result.get('error', 'Unknown')}")

    # Test scipy solvers
    print("\nTesting scipy solvers:")
    for solver in solvers_scipy:
        print(f"  {solver}...", end=' ', flush=True)
        result = test_scipy_solver(base_ode, initial_state, t_span, solver)
        results.append(result)
        if result['success']:
            nfev = result.get('nfev', 'N/A')
            print(f"{result['time']:.3f}s (nfev: {nfev})")
        else:
            print(f"FAILED: {result.get('error', 'Unknown')}")

    # Summary table
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)
    print(f"\n{'Solver':<20} {'Backend':<15} {'Time (s)':<12} {'NFE':<10}")
    print("-"*70)

    for result in results:
        if result['success']:
            nfev = result.get('nfev', '-')
            print(f"{result['solver']:<20} {result['backend']:<15} {result['time']:>8.3f}    {str(nfev):<10}")
        else:
            print(f"{result['solver']:<20} {result['backend']:<15} {'FAILED':<12} {'-':<10}")

    # Find fastest
    successful_results = [r for r in results if r['success']]
    if successful_results:
        fastest = min(successful_results, key=lambda x: x['time'])
        print(f"\n{'='*70}")
        print(f"FASTEST: {fastest['solver']} ({fastest['backend']})")
        print(f"  Time: {fastest['time']:.3f}s")
        if 'nfev' in fastest:
            print(f"  Function evaluations: {fastest['nfev']}")

        # Compare speedup vs bosh3
        bosh3_result = next((r for r in results if r['solver'] == 'bosh3'), None)
        if bosh3_result and bosh3_result['success'] and fastest['solver'] != 'bosh3':
            speedup = bosh3_result['time'] / fastest['time']
            print(f"  Speedup vs bosh3: {speedup:.2f}x")
        print("="*70)

    # Trajectory comparison
    print("\n" + "="*70)
    print("TRAJECTORY ACCURACY (vs bosh3 reference)")
    print("="*70)

    bosh3_result = next((r for r in results if r['solver'] == 'bosh3' and r['success']), None)
    if bosh3_result:
        print(f"\nUsing bosh3 as reference trajectory\n")
        for result in results:
            if result['success'] and result['solver'] != 'bosh3':
                errors = compare_trajectories(
                    bosh3_result['trajectory'],
                    result['trajectory'],
                    ('bosh3', result['solver'])
                )
                print(f"{result['solver']}:")
                print(f"  Max abs error: {errors['max_abs_error']:.2e}")
                print(f"  Mean abs error: {errors['mean_abs_error']:.2e}")
                print(f"  Max rel error: {errors['max_rel_error']:.2e}")
                print()

    # Note about training
    print("="*70)
    print("IMPORTANT NOTE")
    print("="*70)
    print("""
For TRAINING with gradient-based optimization:
  ✓ Use: bosh3 or adaptive_heun (support PyTorch autodiff)
  ✗ Cannot use: Radau, BDF, LSODA (scipy only, no autodiff)

If Radau is fastest, it suggests the HPA system is very stiff.
Consider using torchdiffeq's implicit_adams or writing a custom
PyTorch-compatible implicit solver.
""")


if __name__ == "__main__":
    main()
