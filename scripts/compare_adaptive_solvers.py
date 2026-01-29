"""Compare adaptive ODE solvers on CPU and GPU with reduced time horizon."""
import torch
import time
import importlib
import sys
from pathlib import Path
from torchdiffeq import odeint

# Add src and configs to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root))


def test_solver(env_name, solver, device, time_horizon=20, n_reward_steps=100):
    """Test a single solver configuration.

    Args:
        env_name: Config name (e.g., 'hpa_i1')
        solver: Solver name
        device: 'cpu' or 'cuda'
        time_horizon: Simulation time
        n_reward_steps: Number of time points

    Returns:
        dict with timing info or error
    """
    # Import config
    config_module = importlib.import_module(f"configs.{env_name}")
    config = config_module.ENV_CONFIG

    # Create ODE
    base_ode = config['create_base_ode']()

    # Move to device
    if device == 'cuda':
        if hasattr(base_ode, 'fixed_params') and base_ode.fixed_params is not None:
            base_ode.fixed_params = base_ode.fixed_params.cuda()

    # Get initial state and time points
    initial_state = config['initial_state'].to(device)
    t_span = torch.linspace(0, time_horizon, n_reward_steps + 1).to(device)

    try:
        # Warm up
        _ = odeint(base_ode, initial_state, t_span, method=solver)
        if device == 'cuda':
            torch.cuda.synchronize()

        # Time it
        if device == 'cuda':
            torch.cuda.synchronize()
        start = time.time()
        traj = odeint(base_ode, initial_state, t_span, method=solver)
        if device == 'cuda':
            torch.cuda.synchronize()
        elapsed = time.time() - start

        return {
            'solver': solver,
            'device': device,
            'time': elapsed,
            'success': True
        }
    except Exception as e:
        return {
            'solver': solver,
            'device': device,
            'time': float('inf'),
            'success': False,
            'error': str(e)
        }


def main():
    """Compare all adaptive solvers on CPU and GPU."""
    if not torch.cuda.is_available():
        print("ERROR: No GPU available!")
        return

    print("="*70)
    print("Adaptive Solver Comparison: CPU vs GPU")
    print("="*70)
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Environment: HPA_i1")
    print(f"Time horizon: 20 days (reduced from 50)")
    print(f"Time steps: 101")
    print()

    solvers = ['dopri5', 'dopri8', 'bosh3', 'adaptive_heun']
    devices = ['cpu', 'cuda']

    results = []

    # Test each combination
    for solver in solvers:
        print(f"\n{solver}:")
        for device in devices:
            print(f"  Testing on {device.upper()}...", end=' ', flush=True)
            result = test_solver('hpa_i1', solver, device, time_horizon=20)
            results.append(result)

            if result['success']:
                print(f"{result['time']:.3f}s")
            else:
                print(f"FAILED: {result.get('error', 'Unknown error')}")

    # Print summary table
    print("\n" + "="*70)
    print("SUMMARY TABLE")
    print("="*70)
    print(f"\n{'Solver':<15} {'CPU (s)':<12} {'GPU (s)':<12} {'GPU Speedup':<15}")
    print("-"*70)

    for solver in solvers:
        cpu_result = next((r for r in results if r['solver'] == solver and r['device'] == 'cpu'), None)
        gpu_result = next((r for r in results if r['solver'] == solver and r['device'] == 'cuda'), None)

        if cpu_result and gpu_result and cpu_result['success'] and gpu_result['success']:
            cpu_time = cpu_result['time']
            gpu_time = gpu_result['time']
            speedup = cpu_time / gpu_time

            if speedup > 1:
                speedup_str = f"{speedup:.2f}x faster"
            else:
                speedup_str = f"{1/speedup:.2f}x slower"

            print(f"{solver:<15} {cpu_time:>8.3f}    {gpu_time:>8.3f}    {speedup_str:<15}")
        else:
            cpu_str = f"{cpu_result['time']:.3f}" if cpu_result and cpu_result['success'] else "FAILED"
            gpu_str = f"{gpu_result['time']:.3f}" if gpu_result and gpu_result['success'] else "FAILED"
            print(f"{solver:<15} {cpu_str:>8}    {gpu_str:>8}    {'N/A':<15}")

    # Find best solver for each device
    print("\n" + "="*70)
    print("BEST SOLVERS")
    print("="*70)

    for device in devices:
        device_results = [r for r in results if r['device'] == device and r['success']]
        if device_results:
            best = min(device_results, key=lambda x: x['time'])
            print(f"\n{device.upper()}: {best['solver']}")
            print(f"  Time: {best['time']:.3f}s")

            # Compare to dopri5 baseline
            dopri5_result = next((r for r in results if r['solver'] == 'dopri5' and r['device'] == device), None)
            if dopri5_result and dopri5_result['success'] and best['solver'] != 'dopri5':
                speedup = dopri5_result['time'] / best['time']
                print(f"  Speedup vs dopri5: {speedup:.2f}x")

    # Overall winner
    all_successful = [r for r in results if r['success']]
    if all_successful:
        winner = min(all_successful, key=lambda x: x['time'])
        print(f"\n{'='*70}")
        print(f"OVERALL WINNER: {winner['solver']} on {winner['device'].upper()}")
        print(f"  Time: {winner['time']:.3f}s")
        print(f"{'='*70}")


if __name__ == "__main__":
    main()
