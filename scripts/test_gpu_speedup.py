"""Test whether GPU acceleration helps with ODE integration."""
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


def test_gpu_speedup(env_name='hpa_i1', solver='dopri5', n_runs=3):
    """Test CPU vs GPU performance for ODE integration.

    Args:
        env_name: Name of the config module
        solver: ODE solver to use
        n_runs: Number of runs to average
    """
    # Check GPU availability
    if not torch.cuda.is_available():
        print("WARNING: CUDA not available. This test requires a GPU.")
        print("You're running on CPU only.")
        return

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA version: {torch.version.cuda}")
    print()

    # Import config
    config_module = importlib.import_module(f"configs.{env_name}")
    config = config_module.ENV_CONFIG

    # Get parameters
    initial_state = config['initial_state']
    time_horizon = config['defaults']['time_horizon']
    n_reward_steps = config['defaults']['n_reward_steps']
    t_span = torch.linspace(0, time_horizon, n_reward_steps + 1)

    print(f"Environment: {env_name}")
    print(f"Solver: {solver}")
    print(f"State dimension: {len(initial_state)}")
    print(f"Time steps: {len(t_span)}")
    print()

    results = {}

    # Test CPU
    print("="*60)
    print("CPU TEST")
    print("="*60)

    base_ode_cpu = config['create_base_ode']()
    initial_state_cpu = initial_state.cpu()
    t_span_cpu = t_span.cpu()

    # Warm up
    for _ in range(2):
        _ = odeint(base_ode_cpu, initial_state_cpu, t_span_cpu, method=solver)

    # Time
    cpu_times = []
    for i in range(n_runs):
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        start = time.time()
        traj_cpu = odeint(base_ode_cpu, initial_state_cpu, t_span_cpu, method=solver)
        elapsed = time.time() - start
        cpu_times.append(elapsed)
        print(f"  Run {i+1}/{n_runs}: {elapsed:.4f}s")

    cpu_avg = sum(cpu_times) / len(cpu_times)
    print(f"\nCPU average: {cpu_avg:.4f}s")
    results['cpu'] = cpu_avg

    # Test GPU
    print("\n" + "="*60)
    print("GPU TEST")
    print("="*60)

    try:
        base_ode_gpu = config['create_base_ode']()

        # Move ODE parameters to GPU
        if hasattr(base_ode_gpu, 'fixed_params') and base_ode_gpu.fixed_params is not None:
            base_ode_gpu.fixed_params = base_ode_gpu.fixed_params.cuda()

        initial_state_gpu = initial_state.cuda()
        t_span_gpu = t_span.cuda()

        # Warm up
        for _ in range(2):
            _ = odeint(base_ode_gpu, initial_state_gpu, t_span_gpu, method=solver)
            torch.cuda.synchronize()

        # Time
        gpu_times = []
        for i in range(n_runs):
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            start = time.time()
            traj_gpu = odeint(base_ode_gpu, initial_state_gpu, t_span_gpu, method=solver)
            torch.cuda.synchronize()
            elapsed = time.time() - start
            gpu_times.append(elapsed)
            print(f"  Run {i+1}/{n_runs}: {elapsed:.4f}s")

        gpu_avg = sum(gpu_times) / len(gpu_times)
        print(f"\nGPU average: {gpu_avg:.4f}s")
        results['gpu'] = gpu_avg

        # Speedup
        speedup = cpu_avg / gpu_avg
        print("\n" + "="*60)
        print("RESULTS")
        print("="*60)
        print(f"CPU: {cpu_avg:.4f}s")
        print(f"GPU: {gpu_avg:.4f}s")
        if speedup > 1:
            print(f"GPU is {speedup:.2f}x FASTER")
        else:
            print(f"GPU is {1/speedup:.2f}x SLOWER (CPU better)")

    except Exception as e:
        print(f"\nGPU test failed: {e}")
        import traceback
        traceback.print_exc()

    return results


def explain_gpu_tradeoffs():
    """Explain when GPU helps with ODE solving."""
    print("""
GPU ACCELERATION FOR ODE SOLVING - KEY POINTS
================================================================

1. ODE SOLVING IS SEQUENTIAL
   - Can't parallelize time dimension (step t depends on step t-1)
   - GPU doesn't help with "more time steps"

2. GPU HELPS WITH:
   ✓ Complex ODE functions (many operations per evaluation)
   ✓ High-dimensional state spaces (large matrix operations)
   ✓ Batch processing (multiple trajectories in parallel)
   ✓ Large controllers (polynomial basis, neural networks)

3. GPU OVERHEAD:
   ✗ Data transfer CPU↔GPU
   ✗ Kernel launch overhead
   ✗ For simple/small ODEs, overhead > benefit

4. FOR HPA:
   - Complex ODE: ✓ (many operations per step)
   - 5D state: ~ (medium dimensionality)
   - Expected: GPU might help 2-5x, not 100x

5. BEST USE CASE:
   - Training with batched initial conditions
   - Multiple trajectories in parallel
   - Large polynomial controllers (high order)

RECOMMENDATION:
   Try GPU, but don't expect miracles for single trajectory.
   Real speedup comes from batching multiple ICs in parallel.
================================================================
""")


if __name__ == "__main__":
    explain_gpu_tradeoffs()

    if torch.cuda.is_available():
        print("\nRunning GPU vs CPU comparison...\n")
        test_gpu_speedup('hpa_i1', solver='dopri5', n_runs=3)
    else:
        print("\nNo GPU available - skipping performance test.")
        print("This is running on WSL2/CPU only.")
