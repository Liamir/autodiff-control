"""Test PyTorch odeint vs odeint_adjoint vs JAX Diffrax."""
import time
import torch
import jax
import jax.numpy as jnp
import diffrax
from torchdiffeq import odeint, odeint_adjoint
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
        x1, x2, x3, P, A = x[0], x[1], x[2], x[3], x[4]
        I1 = I2 = I3 = 1.0
        C1 = C2 = C3 = 1.0
        A1 = A2 = A3 = 1.0
        u = stressor_val

        def MR(x_val):
            return 1.0 / x_val

        def GR(x_val):
            return 1.0 / (jnp.pow(x_val / KGR, nGR) + 1.0)

        x3_eff = C3 * x3
        dx1_dt = gamma_x1 * (I1 * u * MR(x3_eff) * GR(x3_eff) - A1 * x1)
        dx2_dt = gamma_x2 * (I2 * (C1 * x1) * P * GR(x3_eff) - A2 * x2)
        dx3_dt = gamma_x3 * (I3 * (C2 * x2) * A - A3 * x3)
        dP_dt = gamma_P * P * ((C1 * x1) * (1.0 - P / KP) - 1.0)
        dA_dt = gamma_A * A * ((C2 * x2) * (1.0 - A / KA) - 1.0)

        return jnp.array([dx1_dt, dx2_dt, dx3_dt, dP_dt, dA_dt])

    return hpa_ode_jax


def test_pytorch_method(ode, initial_state, t_span, use_adjoint=False):
    """Test PyTorch with odeint or odeint_adjoint."""
    method_name = "odeint_adjoint" if use_adjoint else "odeint"
    ode_func = odeint_adjoint if use_adjoint else odeint

    print(f"\nPyTorch {method_name}:")

    try:
        # Warm up
        print(f"  Warming up...", end=' ', flush=True)
        _ = ode_func(ode, initial_state, t_span, method='dopri5')
        print("done")

        # Forward pass
        print(f"  Forward pass...", end=' ', flush=True)
        start = time.time()
        traj = ode_func(ode, initial_state, t_span, method='dopri5')
        forward_time = time.time() - start
        print(f"{forward_time:.3f}s")

        # Backward pass
        print(f"  Backward pass...", end=' ', flush=True)
        if hasattr(ode, 'fixed_params') and ode.fixed_params is not None:
            ode.fixed_params = ode.fixed_params.clone().requires_grad_(True)
            param = ode.fixed_params
        else:
            initial_state_grad = initial_state.clone().requires_grad_(True)
            param = initial_state_grad

        start = time.time()
        traj = ode_func(ode, initial_state if param is ode.fixed_params else initial_state_grad, t_span, method='dopri5')
        loss = traj[-1].sum()
        loss.backward()
        backward_time = time.time() - start - forward_time
        print(f"{backward_time:.3f}s")

        total_time = forward_time + backward_time
        has_grad = param.grad is not None and param.grad.abs().sum() > 0
        grad_norm = param.grad.norm().item() if has_grad else 0.0

        print(f"  Total: {total_time:.3f}s")
        print(f"  Gradients: {'✓' if has_grad else '✗'} (norm: {grad_norm:.2e})")

        return {
            'name': f"PyTorch {method_name}",
            'forward_time': forward_time,
            'backward_time': backward_time,
            'total_time': total_time,
            'gradients': has_grad,
            'grad_norm': grad_norm,
            'success': True,
        }

    except Exception as e:
        print(f"\n  ✗ FAILED: {str(e)[:150]}")
        return {'name': f"PyTorch {method_name}", 'success': False, 'error': str(e)}


def test_jax_adjoint_mode(ode_jax, initial_state, t_span, adjoint_mode):
    """Test JAX with different adjoint modes."""
    print(f"\nJAX Diffrax ({adjoint_mode}):")

    y0 = jnp.array(initial_state.detach().cpu().numpy())
    t_eval = jnp.array(t_span.detach().cpu().numpy())

    # Choose adjoint controller
    if adjoint_mode == 'RecursiveCheckpoint':
        adjoint = diffrax.RecursiveCheckpointAdjoint()
    elif adjoint_mode == 'BacksolveAdjoint':
        adjoint = diffrax.BacksolveAdjoint()
    elif adjoint_mode == 'DirectAdjoint':
        adjoint = diffrax.DirectAdjoint()
    else:
        raise ValueError(f"Unknown adjoint mode: {adjoint_mode}")

    solver = diffrax.Dopri5()
    stepsize_controller = diffrax.PIDController(rtol=1e-7, atol=1e-9)

    try:
        # Warm up (JIT compile)
        print(f"  JIT compiling...", end=' ', flush=True)
        _ = diffrax.diffeqsolve(
            diffrax.ODETerm(ode_jax),
            solver,
            t0=0.0,
            t1=float(t_eval[-1]),
            dt0=0.1,
            y0=y0,
            saveat=diffrax.SaveAt(ts=t_eval),
            stepsize_controller=stepsize_controller,
            adjoint=adjoint,
        )
        print("done")

        # Forward pass
        print(f"  Forward pass...", end=' ', flush=True)
        start = time.time()
        sol = diffrax.diffeqsolve(
            diffrax.ODETerm(ode_jax),
            solver,
            t0=0.0,
            t1=float(t_eval[-1]),
            dt0=0.1,
            y0=y0,
            saveat=diffrax.SaveAt(ts=t_eval),
            stepsize_controller=stepsize_controller,
            adjoint=adjoint,
        )
        forward_time = time.time() - start
        steps = sol.stats['num_steps']
        print(f"{forward_time:.3f}s ({steps} steps)")

        # Backward pass
        print(f"  Backward pass...", end=' ', flush=True)

        def loss_fn(y0_param):
            sol = diffrax.diffeqsolve(
                diffrax.ODETerm(ode_jax),
                solver,
                t0=0.0,
                t1=float(t_eval[-1]),
                dt0=0.1,
                y0=y0_param,
                saveat=diffrax.SaveAt(ts=t_eval),
                stepsize_controller=stepsize_controller,
                adjoint=adjoint,
            )
            return jnp.sum(sol.ys[-1])

        start = time.time()
        grad_fn = jax.grad(loss_fn)
        grad = grad_fn(y0)
        backward_time = time.time() - start - forward_time
        print(f"{backward_time:.3f}s")

        grad_norm = float(jnp.linalg.norm(grad))
        total_time = forward_time + backward_time

        print(f"  Total: {total_time:.3f}s")
        print(f"  Gradients: ✓ (norm: {grad_norm:.2e})")

        return {
            'name': f"JAX {adjoint_mode}",
            'forward_time': forward_time,
            'backward_time': backward_time,
            'total_time': total_time,
            'steps': steps,
            'gradients': True,
            'grad_norm': grad_norm,
            'success': True,
        }

    except Exception as e:
        print(f"\n  ✗ FAILED: {str(e)[:150]}")
        return {'name': f"JAX {adjoint_mode}", 'success': False, 'error': str(e)}


def main():
    """Compare different backpropagation methods."""
    print("="*80)
    print("BACKPROPAGATION METHOD COMPARISON")
    print("="*80)
    print("Testing: odeint vs odeint_adjoint (PyTorch)")
    print("         RecursiveCheckpoint vs BacksolveAdjoint (JAX)")
    print()
    print("HPA System: horizon=20 days, 101 time points")
    print()

    # Setup
    import importlib
    config_module = importlib.import_module(f"configs.hpa_i1")
    config = config_module.ENV_CONFIG

    base_ode = config['create_base_ode']()
    initial_state = config['initial_state']
    time_horizon = 20.0
    n_reward_steps = 100
    t_span = torch.linspace(0, time_horizon, n_reward_steps + 1)

    # Create JAX ODE
    fixed_params_np = base_ode.fixed_params.detach().cpu().numpy()
    ode_jax = create_hpa_jax(fixed_params_np, stressor_val=2.0)

    results = []

    # Test PyTorch methods
    print("-"*80)
    print("PYTORCH METHODS")
    print("-"*80)

    result = test_pytorch_method(config['create_base_ode'](), initial_state, t_span, use_adjoint=False)
    results.append(result)

    result = test_pytorch_method(config['create_base_ode'](), initial_state, t_span, use_adjoint=True)
    results.append(result)

    # Test JAX methods
    print("\n" + "-"*80)
    print("JAX/DIFFRAX METHODS")
    print("-"*80)

    for adjoint_mode in ['RecursiveCheckpoint', 'BacksolveAdjoint']:
        result = test_jax_adjoint_mode(ode_jax, initial_state, t_span, adjoint_mode)
        results.append(result)

    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"\n{'Method':<30} {'Fwd (s)':<12} {'Bwd (s)':<12} {'Total (s)':<12} {'Speedup':<10}")
    print("-"*80)

    baseline_time = None
    for r in results:
        if r['success']:
            fwd = r['forward_time']
            bwd = r['backward_time']
            total = r['total_time']

            if baseline_time is None:
                baseline_time = total
                speedup_str = "baseline"
            else:
                speedup = baseline_time / total
                speedup_str = f"{speedup:.2f}x"

            print(f"{r['name']:<30} {fwd:>8.3f}    {bwd:>8.3f}    {total:>8.3f}    {speedup_str:<10}")
        else:
            print(f"{r['name']:<30} {'FAILED':<12}")

    # Analysis
    print("\n" + "="*80)
    print("ANALYSIS")
    print("="*80)

    pt_odeint = next((r for r in results if r['name'] == 'PyTorch odeint'), None)
    pt_adjoint = next((r for r in results if r['name'] == 'PyTorch odeint_adjoint'), None)
    jax_recursive = next((r for r in results if 'RecursiveCheckpoint' in r['name']), None)
    jax_backsolve = next((r for r in results if 'BacksolveAdjoint' in r['name']), None)

    if pt_odeint and pt_adjoint and pt_odeint['success'] and pt_adjoint['success']:
        speedup = pt_odeint['total_time'] / pt_adjoint['total_time']
        print(f"\nPyTorch: odeint_adjoint is {speedup:.2f}x vs odeint")
        if speedup > 1.5:
            print("  → Adjoint method provides significant speedup in PyTorch!")
        else:
            print("  → Adjoint method provides modest benefit in PyTorch")

    if jax_recursive and jax_backsolve and jax_recursive['success'] and jax_backsolve['success']:
        speedup = jax_recursive['total_time'] / jax_backsolve['total_time']
        print(f"\nJAX: BacksolveAdjoint is {speedup:.2f}x vs RecursiveCheckpoint")
        if speedup > 1.5:
            print("  → BacksolveAdjoint provides significant speedup!")
            print("  ⚠️  But gradients may be approximate (see Diffrax docs)")
        else:
            print("  → RecursiveCheckpoint (default) is competitive")

    # Best option
    successful = [r for r in results if r['success']]
    if successful:
        fastest = min(successful, key=lambda x: x['total_time'])
        print(f"\n{'='*80}")
        print(f"FASTEST METHOD: {fastest['name']}")
        print(f"  Total time: {fastest['total_time']:.3f}s")
        print(f"={'='*80}")


if __name__ == "__main__":
    main()
