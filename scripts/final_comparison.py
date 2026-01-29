"""Final comprehensive comparison: PyTorch vs JAX/Diffrax with gradients."""
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


def test_pytorch(ode, initial_state, t_span, solver='dopri5'):
    """Test PyTorch solver (forward only)."""
    try:
        # Warm up
        _ = odeint(ode, initial_state, t_span, method=solver)

        # Time forward pass
        start = time.time()
        traj = odeint(ode, initial_state, t_span, method=solver)
        forward_time = time.time() - start

        return {
            'solver': solver,
            'backend': 'pytorch',
            'forward_time': forward_time,
            'success': True,
            'trajectory': traj.detach().cpu().numpy()
        }
    except Exception as e:
        return {'solver': solver, 'backend': 'pytorch', 'success': False, 'error': str(e)}


def test_pytorch_gradients(ode, initial_state, t_span, solver='dopri5'):
    """Test PyTorch solver with backward pass."""
    try:
        # Make params require grad
        if hasattr(ode, 'fixed_params') and ode.fixed_params is not None:
            ode.fixed_params = ode.fixed_params.clone().requires_grad_(True)
            param = ode.fixed_params
        else:
            initial_state = initial_state.clone().requires_grad_(True)
            param = initial_state

        # Forward + backward
        start = time.time()
        traj = odeint(ode, initial_state, t_span, method=solver)
        loss = traj[-1].sum()
        loss.backward()
        total_time = time.time() - start

        if param.grad is not None and param.grad.abs().sum() > 0:
            return True, param.grad.norm().item(), total_time
        return False, 0.0, total_time
    except Exception as e:
        return False, str(e), 0.0


def test_jax(ode_jax, initial_state, t_span, solver_name, rtol, atol):
    """Test JAX/Diffrax solver (forward only)."""
    y0 = jnp.array(initial_state.detach().cpu().numpy())
    t_eval = jnp.array(t_span.detach().cpu().numpy())

    if solver_name == 'Dopri5':
        solver = diffrax.Dopri5()
    elif solver_name == 'Kvaerno5':
        solver = diffrax.Kvaerno5()
    elif solver_name == 'Kvaerno3':
        solver = diffrax.Kvaerno3()
    else:
        raise ValueError(f"Unknown solver: {solver_name}")

    stepsize_controller = diffrax.PIDController(rtol=rtol, atol=atol)

    try:
        # Warm up (JIT compile)
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

        # Time forward pass
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

        return {
            'solver': solver_name,
            'backend': 'jax',
            'forward_time': forward_time,
            'steps': sol.stats['num_steps'],
            'success': True,
            'trajectory': np.array(sol.ys)
        }
    except Exception as e:
        return {'solver': solver_name, 'backend': 'jax', 'success': False, 'error': str(e)[:100]}


def test_jax_gradients(ode_jax, initial_state, t_span, solver_name, rtol, atol):
    """Test JAX/Diffrax solver with gradients."""
    y0 = jnp.array(initial_state.detach().cpu().numpy())
    t_eval = jnp.array(t_span.detach().cpu().numpy())

    if solver_name == 'Dopri5':
        solver = diffrax.Dopri5()
    elif solver_name == 'Kvaerno5':
        solver = diffrax.Kvaerno5()
    elif solver_name == 'Kvaerno3':
        solver = diffrax.Kvaerno3()
    else:
        raise ValueError(f"Unknown solver: {solver_name}")

    stepsize_controller = diffrax.PIDController(rtol=rtol, atol=atol)

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

    try:
        # Time forward + backward
        start = time.time()
        grad_fn = jax.grad(loss_fn)
        grad = grad_fn(y0)
        total_time = time.time() - start

        grad_norm = float(jnp.linalg.norm(grad))
        return True, grad_norm, total_time
    except Exception as e:
        return False, str(e)[:100], 0.0


def main():
    """Complete comparison with gradients."""
    print("="*80)
    print("FINAL COMPARISON: PyTorch vs JAX/Diffrax")
    print("="*80)
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

    # Test PyTorch
    print("-"*80)
    print("PYTORCH/TORCHDIFFEQ")
    print("-"*80)

    for solver in ['dopri5', 'bosh3']:
        print(f"\n{solver}:")
        # Forward only
        result = test_pytorch(base_ode, initial_state, t_span, solver)
        if result['success']:
            print(f"  Forward: {result['forward_time']:.3f}s")

            # Test gradients
            grad_ok, grad_info, total_time = test_pytorch_gradients(
                config['create_base_ode'](),  # Fresh ODE
                initial_state,
                t_span,
                solver
            )
            backward_time = total_time - result['forward_time'] if grad_ok else 0
            print(f"  Backward: {backward_time:.3f}s")
            print(f"  Total: {total_time:.3f}s")
            print(f"  Gradients: {'✓' if grad_ok else '✗'} (norm: {grad_info})")

            results.append({
                **result,
                'backward_time': backward_time,
                'total_time': total_time,
                'gradients': grad_ok,
                'grad_norm': grad_info if grad_ok else None
            })
        else:
            print(f"  FAILED: {result.get('error', '')[:80]}")

    # Test JAX
    print("\n" + "-"*80)
    print("JAX/DIFFRAX")
    print("-"*80)

    jax_configs = [
        ('Dopri5', 1e-7, 1e-9),  # Explicit baseline
        ('Kvaerno5', 1e-3, 1e-3),  # Implicit, relaxed
        ('Kvaerno3', 1e-3, 1e-3),  # Implicit, relaxed
    ]

    for solver_name, rtol, atol in jax_configs:
        tol_str = f"rtol={rtol}, atol={atol}"
        print(f"\n{solver_name} ({tol_str}):")

        # Forward only
        result = test_jax(ode_jax, initial_state, t_span, solver_name, rtol, atol)
        if result['success']:
            print(f"  Forward: {result['forward_time']:.3f}s ({result['steps']} steps)")

            # Test gradients
            grad_ok, grad_info, total_time = test_jax_gradients(
                ode_jax, initial_state, t_span, solver_name, rtol, atol
            )
            backward_time = total_time - result['forward_time'] if grad_ok else 0
            print(f"  Backward: {backward_time:.3f}s")
            print(f"  Total: {total_time:.3f}s")
            print(f"  Gradients: {'✓' if grad_ok else '✗'} (norm: {grad_info})")

            results.append({
                **result,
                'backward_time': backward_time,
                'total_time': total_time,
                'gradients': grad_ok,
                'grad_norm': grad_info if grad_ok else None,
                'tol': tol_str
            })
        else:
            print(f"  FAILED: {result.get('error', '')[:80]}")

    # Summary table
    print("\n" + "="*80)
    print("COMPLETE RESULTS")
    print("="*80)
    print(f"\n{'Solver':<20} {'Backend':<10} {'Fwd (s)':<10} {'Bwd (s)':<10} {'Total (s)':<10} {'Grads':<8}")
    print("-"*80)

    for r in results:
        if r['success']:
            grad_str = "✓" if r.get('gradients') else "✗"
            print(f"{r['solver']:<20} {r['backend']:<10} {r['forward_time']:>8.3f}  {r.get('backward_time', 0):>8.3f}  {r.get('total_time', 0):>8.3f}  {grad_str:<8}")

    # Best for training
    print("\n" + "="*80)
    print("RECOMMENDATION FOR TRAINING")
    print("="*80)

    trainable = [r for r in results if r['success'] and r.get('gradients')]
    if trainable:
        fastest = min(trainable, key=lambda x: x.get('total_time', float('inf')))
        print(f"\nFASTEST TRAINABLE: {fastest['solver']} ({fastest['backend']})")
        print(f"  Forward: {fastest['forward_time']:.3f}s")
        print(f"  Backward: {fastest.get('backward_time', 0):.3f}s")
        print(f"  Total: {fastest.get('total_time', 0):.3f}s")

        # Compare to PyTorch dopri5
        pt_dopri5 = next((r for r in results if r['solver'] == 'dopri5' and r['backend'] == 'pytorch'), None)
        if pt_dopri5:
            speedup = pt_dopri5.get('total_time', 0) / fastest.get('total_time', 1)
            print(f"\n  Speedup vs PyTorch dopri5: {speedup:.1f}x FASTER!")

        if fastest['backend'] == 'jax':
            print(f"\n  ✓ Switch to JAX/Diffrax for massive speedup!")


if __name__ == "__main__":
    main()
