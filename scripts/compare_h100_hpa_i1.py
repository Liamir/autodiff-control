"""Compare dopri5 (PyTorch vs JAX) vs Kvaerno5 with horizon=100 on hpa_i1."""
import time
import torch
import jax
import jax.numpy as jnp
import diffrax
import numpy as np
from torchdiffeq import odeint
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


def test_pytorch(ode, initial_state, t_span, name="PyTorch dopri5"):
    """Test PyTorch dopri5."""
    print(f"\n{name}:")
    print(f"  Warming up...", end=' ', flush=True)

    try:
        # Warm up
        _ = odeint(ode, initial_state, t_span, method='dopri5')
        print("done")

        # Forward pass
        print(f"  Forward pass...", end=' ', flush=True)
        start = time.time()
        traj = odeint(ode, initial_state, t_span, method='dopri5')
        forward_time = time.time() - start
        print(f"{forward_time:.3f}s")

        # Backward pass
        print(f"  Backward pass...", end=' ', flush=True)
        ode_grad = ode  # Use same ODE instance
        if hasattr(ode_grad, 'fixed_params') and ode_grad.fixed_params is not None:
            ode_grad.fixed_params = ode_grad.fixed_params.clone().requires_grad_(True)
            param = ode_grad.fixed_params
        else:
            initial_state_grad = initial_state.clone().requires_grad_(True)
            param = initial_state_grad

        start = time.time()
        traj = odeint(ode_grad, initial_state if param is ode_grad.fixed_params else initial_state_grad, t_span, method='dopri5')
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
            'name': name,
            'forward_time': forward_time,
            'backward_time': backward_time,
            'total_time': total_time,
            'gradients': has_grad,
            'grad_norm': grad_norm,
            'success': True,
        }

    except Exception as e:
        print(f"\n  ✗ FAILED: {str(e)[:100]}")
        return {'name': name, 'success': False, 'error': str(e)}


def test_jax(ode_jax, initial_state, t_span, solver_name, rtol, atol, name):
    """Test JAX/Diffrax solver."""
    print(f"\n{name}:")

    y0 = jnp.array(initial_state.detach().cpu().numpy())
    t_eval = jnp.array(t_span.detach().cpu().numpy())

    if solver_name == 'Dopri5':
        solver = diffrax.Dopri5()
    elif solver_name == 'Kvaerno5':
        solver = diffrax.Kvaerno5()
    else:
        raise ValueError(f"Unknown solver: {solver_name}")

    stepsize_controller = diffrax.PIDController(rtol=rtol, atol=atol)

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
            max_steps=100000,
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
            max_steps=100000,
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
                max_steps=100000,
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
            'name': name,
            'forward_time': forward_time,
            'backward_time': backward_time,
            'total_time': total_time,
            'steps': steps,
            'gradients': True,
            'grad_norm': grad_norm,
            'success': True,
        }

    except Exception as e:
        print(f"\n  ✗ FAILED: {str(e)[:100]}")
        return {'name': name, 'success': False, 'error': str(e)}


def main():
    """Compare dopri5 (PyTorch vs JAX) and Kvaerno5 with horizon=100."""
    print("="*80)
    print("COMPARISON: dopri5 (PyTorch vs JAX) vs Kvaerno5")
    print("="*80)
    print("HPA_I1 System: horizon=100 days, 101 time points")
    print()

    # Setup
    import importlib
    config_module = importlib.import_module(f"configs.hpa_i1")
    config = config_module.ENV_CONFIG

    base_ode = config['create_base_ode']()
    initial_state = config['initial_state']
    time_horizon = 100.0  # EXTENDED HORIZON
    n_reward_steps = 100
    t_span = torch.linspace(0, time_horizon, n_reward_steps + 1)

    # Create JAX ODE
    fixed_params_np = base_ode.fixed_params.detach().cpu().numpy()
    ode_jax = create_hpa_jax(fixed_params_np, stressor_val=2.0)

    print("Testing 3 configurations:")
    print("  1. PyTorch dopri5")
    print("  2. JAX Dopri5")
    print("  3. JAX Kvaerno5 (implicit, relaxed tolerances)")
    print()

    results = []

    # Test 1: PyTorch dopri5
    print("-"*80)
    print("TEST 1: PyTorch dopri5")
    print("-"*80)
    result = test_pytorch(base_ode, initial_state, t_span, "PyTorch dopri5")
    results.append(result)

    # Test 2: JAX Dopri5
    print("\n" + "-"*80)
    print("TEST 2: JAX Dopri5 (explicit)")
    print("-"*80)
    result = test_jax(ode_jax, initial_state, t_span, 'Dopri5', 1e-7, 1e-9, "JAX Dopri5")
    results.append(result)

    # Test 3: JAX Kvaerno5
    print("\n" + "-"*80)
    print("TEST 3: JAX Kvaerno5 (implicit, stiff-aware)")
    print("-"*80)
    result = test_jax(ode_jax, initial_state, t_span, 'Kvaerno5', 1e-3, 1e-3, "JAX Kvaerno5")
    results.append(result)

    # Summary
    print("\n" + "="*80)
    print("SUMMARY TABLE (horizon=100 days)")
    print("="*80)
    print(f"\n{'Solver':<25} {'Fwd (s)':<12} {'Bwd (s)':<12} {'Total (s)':<12} {'Speedup':<10}")
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

            print(f"{r['name']:<25} {fwd:>8.3f}    {bwd:>8.3f}    {total:>8.3f}    {speedup_str:<10}")
        else:
            print(f"{r['name']:<25} {'FAILED':<12}")

    # Best option
    print("\n" + "="*80)
    print("RECOMMENDATION")
    print("="*80)

    successful = [r for r in results if r['success']]
    if successful:
        fastest = min(successful, key=lambda x: x['total_time'])
        pytorch_result = next((r for r in results if 'PyTorch' in r['name']), None)

        print(f"\nFASTEST: {fastest['name']}")
        print(f"  Forward: {fastest['forward_time']:.3f}s")
        print(f"  Backward: {fastest['backward_time']:.3f}s")
        print(f"  Total: {fastest['total_time']:.3f}s")

        if 'steps' in fastest:
            print(f"  Steps: {fastest['steps']}")

        if pytorch_result and pytorch_result['success']:
            speedup = pytorch_result['total_time'] / fastest['total_time']
            print(f"\n  Speedup vs PyTorch: {speedup:.1f}x FASTER!")

            # Calculate training time savings
            iterations = 500
            pt_time = pytorch_result['total_time'] * iterations
            jax_time = fastest['total_time'] * iterations
            savings = pt_time - jax_time

            print(f"\n  Training time for {iterations} iterations:")
            print(f"    PyTorch: {pt_time/60:.1f} minutes ({pt_time/3600:.1f} hours)")
            print(f"    {fastest['name']}: {jax_time/60:.1f} minutes ({jax_time/3600:.1f} hours)")
            print(f"    Time saved: {savings/60:.1f} minutes ({savings/3600:.1f} hours)")


if __name__ == "__main__":
    main()
