"""Verify that PyTorch and JAX gradients match when computing w.r.t. same variable."""
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


def test_pytorch_grad_wrt_y0(ode, initial_state, t_span):
    """Compute PyTorch gradient w.r.t. initial state."""
    print("\nPyTorch (gradient w.r.t. initial state):")
    print("  Computing...", end=' ', flush=True)

    try:
        # Make initial state require grad
        y0 = initial_state.clone().requires_grad_(True)

        # Forward pass
        start = time.time()
        traj = odeint(ode, y0, t_span, method='dopri5')

        # Loss = sum of final state
        loss = traj[-1].sum()

        # Backward pass
        loss.backward()
        elapsed = time.time() - start

        grad = y0.grad.detach().cpu().numpy()
        grad_norm = float(np.linalg.norm(grad))

        print(f"done ({elapsed:.3f}s)")
        print(f"  Loss: {loss.item():.6e}")
        print(f"  Gradient norm: {grad_norm:.6e}")
        print(f"  Gradient: {grad}")

        return {
            'loss': loss.item(),
            'grad': grad,
            'grad_norm': grad_norm,
            'time': elapsed,
            'success': True
        }

    except Exception as e:
        print(f"FAILED: {str(e)[:100]}")
        return {'success': False, 'error': str(e)}


def test_jax_grad_wrt_y0(ode_jax, initial_state, t_span):
    """Compute JAX gradient w.r.t. initial state."""
    print("\nJAX Dopri5 (gradient w.r.t. initial state):")
    print("  Computing...", end=' ', flush=True)

    y0 = jnp.array(initial_state.detach().cpu().numpy())
    t_eval = jnp.array(t_span.detach().cpu().numpy())

    solver = diffrax.Dopri5()
    stepsize_controller = diffrax.PIDController(rtol=1e-7, atol=1e-9)

    try:
        # Warm up (JIT compile)
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

        # JIT compile
        _ = jax.grad(loss_fn)(y0)

        # Time it
        start = time.time()
        loss = loss_fn(y0)
        grad = jax.grad(loss_fn)(y0)
        elapsed = time.time() - start

        grad_np = np.array(grad)
        grad_norm = float(jnp.linalg.norm(grad))

        print(f"done ({elapsed:.3f}s)")
        print(f"  Loss: {float(loss):.6e}")
        print(f"  Gradient norm: {grad_norm:.6e}")
        print(f"  Gradient: {grad_np}")

        return {
            'loss': float(loss),
            'grad': grad_np,
            'grad_norm': grad_norm,
            'time': elapsed,
            'success': True
        }

    except Exception as e:
        print(f"FAILED: {str(e)[:100]}")
        return {'success': False, 'error': str(e)}


def main():
    """Verify gradient parity between PyTorch and JAX."""
    print("="*80)
    print("GRADIENT PARITY TEST")
    print("="*80)
    print("Computing ∂loss/∂y0 for both PyTorch and JAX")
    print("Loss = sum(final_state)")
    print()

    # Setup
    import importlib
    config_module = importlib.import_module(f"configs.hpa_i1")
    config = config_module.ENV_CONFIG

    base_ode = config['create_base_ode']()
    initial_state = config['initial_state']

    # Use shorter horizon for faster testing
    time_horizon = 20.0
    n_reward_steps = 100
    t_span = torch.linspace(0, time_horizon, n_reward_steps + 1)

    print(f"Configuration:")
    print(f"  Environment: hpa_i1")
    print(f"  Horizon: {time_horizon} days")
    print(f"  Time points: {n_reward_steps + 1}")
    print(f"  Initial state: {initial_state.numpy()}")
    print()

    # Create JAX ODE
    fixed_params_np = base_ode.fixed_params.detach().cpu().numpy()
    ode_jax = create_hpa_jax(fixed_params_np, stressor_val=2.0)

    print("-"*80)
    print("COMPUTING GRADIENTS")
    print("-"*80)

    # Test PyTorch
    pt_result = test_pytorch_grad_wrt_y0(base_ode, initial_state, t_span)

    # Test JAX
    jax_result = test_jax_grad_wrt_y0(ode_jax, initial_state, t_span)

    # Compare
    print("\n" + "="*80)
    print("COMPARISON")
    print("="*80)

    if pt_result['success'] and jax_result['success']:
        # Compare losses
        loss_diff = abs(pt_result['loss'] - jax_result['loss'])
        loss_rel_diff = loss_diff / abs(pt_result['loss'])

        print(f"\nLoss values:")
        print(f"  PyTorch: {pt_result['loss']:.6e}")
        print(f"  JAX:     {jax_result['loss']:.6e}")
        print(f"  Absolute diff: {loss_diff:.6e}")
        print(f"  Relative diff: {loss_rel_diff:.6e}")

        # Compare gradient norms
        grad_norm_diff = abs(pt_result['grad_norm'] - jax_result['grad_norm'])
        grad_norm_rel_diff = grad_norm_diff / pt_result['grad_norm']

        print(f"\nGradient norms:")
        print(f"  PyTorch: {pt_result['grad_norm']:.6e}")
        print(f"  JAX:     {jax_result['grad_norm']:.6e}")
        print(f"  Absolute diff: {grad_norm_diff:.6e}")
        print(f"  Relative diff: {grad_norm_rel_diff:.6e}")

        # Element-wise gradient comparison
        grad_diff = np.abs(pt_result['grad'] - jax_result['grad'])
        grad_rel_diff = grad_diff / (np.abs(pt_result['grad']) + 1e-10)

        print(f"\nElement-wise gradient comparison:")
        print(f"  Max absolute diff: {np.max(grad_diff):.6e}")
        print(f"  Max relative diff: {np.max(grad_rel_diff):.6e}")
        print(f"  Mean absolute diff: {np.mean(grad_diff):.6e}")
        print(f"  Mean relative diff: {np.mean(grad_rel_diff):.6e}")

        # Check if gradients match within tolerance
        rtol = 1e-4  # 0.01% relative tolerance
        atol = 1e-6  # Absolute tolerance

        matches = np.allclose(pt_result['grad'], jax_result['grad'], rtol=rtol, atol=atol)

        print(f"\nGradients match (rtol={rtol}, atol={atol}): {matches}")

        if matches:
            print("\n✓ SUCCESS: PyTorch and JAX gradients match!")
            print("  Both frameworks compute identical gradients w.r.t. initial state.")
            print("  The difference in the previous comparison was because we were")
            print("  computing gradients w.r.t. different variables (params vs y0).")
        else:
            print("\n✗ WARNING: Gradients don't match within tolerance")
            print("  This could be due to:")
            print("  - Numerical differences in ODE solvers")
            print("  - Different adaptive step sizes")
            print("  - Accumulation of floating point errors")

            # Show which components differ most
            print(f"\n  Component-wise differences:")
            for i, (pt_g, jax_g, diff) in enumerate(zip(pt_result['grad'], jax_result['grad'], grad_diff)):
                rel_diff = diff / (abs(pt_g) + 1e-10)
                print(f"    y0[{i}]: PyTorch={pt_g:.6e}, JAX={jax_g:.6e}, diff={diff:.6e} ({rel_diff:.2%})")

        # Performance comparison
        print(f"\nPerformance:")
        print(f"  PyTorch: {pt_result['time']:.3f}s")
        print(f"  JAX:     {jax_result['time']:.3f}s")
        speedup = pt_result['time'] / jax_result['time']
        print(f"  Speedup: {speedup:.1f}x")

    else:
        print("\n✗ One or both methods failed")
        if not pt_result['success']:
            print(f"  PyTorch error: {pt_result.get('error', 'Unknown')}")
        if not jax_result['success']:
            print(f"  JAX error: {jax_result.get('error', 'Unknown')}")


if __name__ == "__main__":
    main()
