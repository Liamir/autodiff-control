"""Time forward pass (ODE integration) for different environments to compare simulation speed."""
import time
import torch
import importlib
import sys
from pathlib import Path
from torchdiffeq import odeint

# Add src and configs to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root))


def time_ode_integration(env_name, solver='dopri5', n_runs=10):
    """Time ODE integration for a given environment (no controller).

    Args:
        env_name: Name of the config module (e.g., 'population', 'hpa_i1')
        solver: ODE solver method (e.g., 'dopri5', 'euler', 'rk4')
        n_runs: Number of runs to average over

    Returns:
        dict: Timing statistics and environment info
    """
    # Import the config
    config_module = importlib.import_module(f"configs.{env_name}")
    config = config_module.ENV_CONFIG

    # Create uncontrolled base ODE
    base_ode = config['create_base_ode']()

    # Get initial state and time horizon
    initial_state = config['initial_state']
    time_horizon = config['defaults']['time_horizon']
    n_reward_steps = config['defaults']['n_reward_steps']

    # Create time points
    t_span = torch.linspace(0, time_horizon, n_reward_steps + 1)

    # Warm up (compile, cache, etc.)
    for _ in range(2):
        try:
            _ = odeint(base_ode, initial_state, t_span, method=solver)
        except Exception as e:
            return {
                'env_name': env_name,
                'solver': solver,
                'error': str(e),
                'avg_time': float('inf'),
                'n_states': len(initial_state),
                'time_horizon': time_horizon,
                'n_time_steps': len(t_span),
            }

    # Time the forward passes
    times = []
    for i in range(n_runs):
        start = time.time()
        try:
            trajectory = odeint(base_ode, initial_state, t_span, method=solver)
            elapsed = time.time() - start
            times.append(elapsed)
        except Exception as e:
            return {
                'env_name': env_name,
                'solver': solver,
                'error': str(e),
                'avg_time': float('inf'),
                'n_states': len(initial_state),
                'time_horizon': time_horizon,
                'n_time_steps': len(t_span),
            }

    avg_time = sum(times) / len(times)
    std_time = (sum((t - avg_time)**2 for t in times) / len(times)) ** 0.5

    return {
        'env_name': env_name,
        'solver': solver,
        'avg_time': avg_time,
        'std_time': std_time,
        'min_time': min(times),
        'max_time': max(times),
        'n_states': len(initial_state),
        'time_horizon': time_horizon,
        'n_time_steps': len(t_span),
        'ode_name': base_ode.name if hasattr(base_ode, 'name') else str(type(base_ode).__name__),
    }


def main():
    """Compare forward pass times for population and HPA environments with different solvers."""
    print("="*80)
    print("ODE Integration Timing Comparison - Multiple Solvers")
    print("="*80)

    envs = ['population', 'hpa_i1']
    # Test different solvers
    solvers = ['dopri5', 'euler', 'rk4', 'bosh3', 'adaptive_heun']

    n_runs = 1  # Just one run per solver for quick testing
    results = []

    for env_name in envs:
        print(f"\n{'='*80}")
        print(f"Environment: {env_name.upper()}")
        print(f"{'='*80}")

        for solver in solvers:
            print(f"\n{env_name} with {solver}:")
            try:
                result = time_ode_integration(env_name, solver=solver, n_runs=n_runs)
                results.append(result)

                if 'error' in result:
                    print(f"  ERROR: {result['error']}")
                else:
                    print(f"  State dimension: {result['n_states']}")
                    print(f"  Time steps: {result['n_time_steps']}")
                    print(f"  Average time: {result['avg_time']:.4f} ± {result['std_time']:.4f} s")
                    print(f"  Time per step: {result['avg_time']/result['n_time_steps']*1000:.2f} ms")

            except Exception as e:
                print(f"  EXCEPTION: {e}")
                import traceback
                traceback.print_exc()

    # Print comparison table
    print("\n" + "="*80)
    print("SUMMARY TABLE")
    print("="*80)
    print(f"\n{'Solver':<15} {'Population (s)':<18} {'HPA_i1 (s)':<18} {'Speedup':<10}")
    print("-"*80)

    # Group by solver
    solver_results = {}
    for result in results:
        if 'error' not in result:
            solver = result['solver']
            if solver not in solver_results:
                solver_results[solver] = {}
            solver_results[solver][result['env_name']] = result['avg_time']

    for solver in solvers:
        if solver in solver_results:
            pop_time = solver_results[solver].get('population', float('inf'))
            hpa_time = solver_results[solver].get('hpa_i1', float('inf'))
            if pop_time != float('inf') and hpa_time != float('inf'):
                speedup = hpa_time / pop_time
                print(f"{solver:<15} {pop_time:>8.4f}s         {hpa_time:>8.4f}s         {speedup:>6.2f}x")
            else:
                print(f"{solver:<15} {'FAILED':<18} {'FAILED':<18} {'-':<10}")

    # Find fastest solver for each env
    print("\n" + "="*80)
    print("FASTEST SOLVERS")
    print("="*80)

    for env_name in envs:
        env_results = [r for r in results if r['env_name'] == env_name and 'error' not in r]
        if env_results:
            fastest = min(env_results, key=lambda x: x['avg_time'])
            print(f"\n{env_name.upper()}:")
            print(f"  Fastest: {fastest['solver']}")
            print(f"  Time: {fastest['avg_time']:.4f}s ({fastest['avg_time']/fastest['n_time_steps']*1000:.2f} ms/step)")

            # Show speedup vs dopri5
            dopri5_result = next((r for r in env_results if r['solver'] == 'dopri5'), None)
            if dopri5_result and fastest['solver'] != 'dopri5':
                speedup = dopri5_result['avg_time'] / fastest['avg_time']
                print(f"  Speedup vs dopri5: {speedup:.2f}x faster")

    print("\n" + "="*80)


if __name__ == "__main__":
    main()
