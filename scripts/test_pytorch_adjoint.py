"""Test PyTorch odeint_adjoint by wrapping ODE as nn.Module."""
import time
import torch
import torch.nn as nn
from torchdiffeq import odeint, odeint_adjoint
import sys
from pathlib import Path

# Add src and configs to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root))


class ODEFuncWrapper(nn.Module):
    """Wrap ODE as nn.Module for odeint_adjoint."""

    def __init__(self, base_ode):
        super().__init__()
        self.base_ode = base_ode
        # Register fixed_params as a parameter so adjoint can track it
        if hasattr(base_ode, 'fixed_params') and base_ode.fixed_params is not None:
            self.fixed_params = nn.Parameter(base_ode.fixed_params.clone())
        else:
            self.fixed_params = None

    def forward(self, t, y):
        """Forward pass through ODE."""
        # Temporarily replace base_ode's fixed_params with our parameter
        if self.fixed_params is not None:
            original_params = self.base_ode.fixed_params
            self.base_ode.fixed_params = self.fixed_params
            result = self.base_ode(t, y)
            self.base_ode.fixed_params = original_params
            return result
        else:
            return self.base_ode(t, y)


def test_pytorch_odeint(ode, initial_state, t_span):
    """Test PyTorch odeint (backprop through solver)."""
    print("\nPyTorch odeint (backprop through solver):")

    try:
        # Warm up
        print(f"  Warming up...", end=' ', flush=True)
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
        if hasattr(ode, 'fixed_params') and ode.fixed_params is not None:
            ode.fixed_params = ode.fixed_params.clone().requires_grad_(True)
            param = ode.fixed_params
        else:
            initial_state_grad = initial_state.clone().requires_grad_(True)
            param = initial_state_grad

        start = time.time()
        traj = odeint(ode, initial_state if param is ode.fixed_params else initial_state_grad, t_span, method='dopri5')
        loss = traj[-1].sum()
        loss.backward()
        backward_time = time.time() - start - forward_time
        print(f"{backward_time:.3f}s")

        total_time = forward_time + backward_time
        has_grad = param.grad is not None and param.grad.abs().sum() > 0
        grad_norm = param.grad.norm().item() if has_grad else 0.0

        print(f"  Total: {total_time:.3f}s")
        print(f"  Gradients: {'✓' if has_grad else '✗'} (norm: {grad_norm:.2e})")

        return total_time, has_grad

    except Exception as e:
        print(f"\n  ✗ FAILED: {str(e)[:150]}")
        return None, False


def test_pytorch_odeint_adjoint(ode_module, initial_state, t_span):
    """Test PyTorch odeint_adjoint (adjoint method)."""
    print("\nPyTorch odeint_adjoint (adjoint method):")

    try:
        # Warm up
        print(f"  Warming up...", end=' ', flush=True)
        _ = odeint_adjoint(ode_module, initial_state, t_span, method='dopri5')
        print("done")

        # Forward pass
        print(f"  Forward pass...", end=' ', flush=True)
        start = time.time()
        traj = odeint_adjoint(ode_module, initial_state, t_span, method='dopri5')
        forward_time = time.time() - start
        print(f"{forward_time:.3f}s")

        # Backward pass
        print(f"  Backward pass...", end=' ', flush=True)
        start = time.time()
        traj = odeint_adjoint(ode_module, initial_state, t_span, method='dopri5')
        loss = traj[-1].sum()
        loss.backward()
        backward_time = time.time() - start - forward_time
        print(f"{backward_time:.3f}s")

        total_time = forward_time + backward_time
        has_grad = ode_module.fixed_params.grad is not None and ode_module.fixed_params.grad.abs().sum() > 0
        grad_norm = ode_module.fixed_params.grad.norm().item() if has_grad else 0.0

        print(f"  Total: {total_time:.3f}s")
        print(f"  Gradients: {'✓' if has_grad else '✗'} (norm: {grad_norm:.2e})")

        return total_time, has_grad

    except Exception as e:
        print(f"\n  ✗ FAILED: {str(e)[:150]}")
        return None, False


def main():
    """Compare odeint vs odeint_adjoint in PyTorch."""
    print("="*80)
    print("PYTORCH: odeint vs odeint_adjoint")
    print("="*80)
    print("HPA System: horizon=20 days, 101 time points")
    print()

    # Setup
    import importlib
    config_module = importlib.import_module(f"configs.hpa_i1")
    config = config_module.ENV_CONFIG

    initial_state = config['initial_state']
    time_horizon = 20.0
    n_reward_steps = 100
    t_span = torch.linspace(0, time_horizon, n_reward_steps + 1)

    print("-"*80)
    print("TEST 1: odeint (backprop through solver)")
    print("-"*80)

    base_ode = config['create_base_ode']()
    time_odeint, grad_ok_odeint = test_pytorch_odeint(base_ode, initial_state, t_span)

    print("\n" + "-"*80)
    print("TEST 2: odeint_adjoint (adjoint method)")
    print("-"*80)

    base_ode2 = config['create_base_ode']()
    ode_module = ODEFuncWrapper(base_ode2)
    time_odeint_adjoint, grad_ok_adjoint = test_pytorch_odeint_adjoint(ode_module, initial_state, t_span)

    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)

    if time_odeint and time_odeint_adjoint:
        speedup = time_odeint / time_odeint_adjoint
        print(f"\nPyTorch odeint: {time_odeint:.3f}s")
        print(f"PyTorch odeint_adjoint: {time_odeint_adjoint:.3f}s")
        print(f"\nSpeedup: {speedup:.2f}x")

        if speedup > 1.2:
            print("\n✓ odeint_adjoint provides significant speedup!")
            print("  → Use odeint_adjoint for training to save memory and time")
        elif speedup < 0.8:
            print("\n✗ odeint_adjoint is SLOWER")
            print("  → Stick with regular odeint")
        else:
            print("\n≈ Similar performance")
            print("  → odeint_adjoint main benefit is memory, not speed")
    else:
        print("\nOne or both methods failed - cannot compare")


if __name__ == "__main__":
    main()
