"""Quick GPU vs CPU test with single run."""
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


def quick_test():
    """Quick single-run GPU vs CPU test."""
    if not torch.cuda.is_available():
        print("No GPU available!")
        return

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()

    # Import config
    config_module = importlib.import_module(f"configs.hpa_i1")
    config = config_module.ENV_CONFIG

    # Get parameters
    initial_state = config['initial_state']
    time_horizon = config['defaults']['time_horizon']
    n_reward_steps = config['defaults']['n_reward_steps']
    t_span = torch.linspace(0, time_horizon, n_reward_steps + 1)

    print(f"HPA environment: {len(initial_state)}D, {len(t_span)} time steps")
    print()

    # CPU Test
    print("CPU Test (dopri5)...")
    base_ode_cpu = config['create_base_ode']()
    initial_state_cpu = initial_state.cpu()
    t_span_cpu = t_span.cpu()

    start = time.time()
    traj_cpu = odeint(base_ode_cpu, initial_state_cpu, t_span_cpu, method='dopri5')
    cpu_time = time.time() - start
    print(f"  CPU time: {cpu_time:.3f}s")

    # GPU Test
    print("\nGPU Test (dopri5)...")
    base_ode_gpu = config['create_base_ode']()

    # Move ODE parameters to GPU
    if hasattr(base_ode_gpu, 'fixed_params') and base_ode_gpu.fixed_params is not None:
        base_ode_gpu.fixed_params = base_ode_gpu.fixed_params.cuda()

    initial_state_gpu = initial_state.cuda()
    t_span_gpu = t_span.cuda()

    torch.cuda.synchronize()
    start = time.time()
    traj_gpu = odeint(base_ode_gpu, initial_state_gpu, t_span_gpu, method='dopri5')
    torch.cuda.synchronize()
    gpu_time = time.time() - start
    print(f"  GPU time: {gpu_time:.3f}s")

    # Results
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"CPU: {cpu_time:.3f}s")
    print(f"GPU: {gpu_time:.3f}s")

    if gpu_time < cpu_time:
        speedup = cpu_time / gpu_time
        print(f"\n✓ GPU is {speedup:.2f}x FASTER!")
    else:
        slowdown = gpu_time / cpu_time
        print(f"\n✗ GPU is {slowdown:.2f}x SLOWER (CPU better for this problem)")


if __name__ == "__main__":
    quick_test()
