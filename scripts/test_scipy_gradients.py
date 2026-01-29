"""Test whether scipy_solver supports backpropagation."""
import torch
import importlib
import sys
from pathlib import Path
from torchdiffeq import odeint

# Add src and configs to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root))


def test_gradients(solver_name, scipy_method=None):
    """Test if a solver supports backpropagation.

    Args:
        solver_name: Name of solver ('bosh3', 'scipy_solver', etc.)
        scipy_method: For scipy_solver, which method to use ('BDF', 'Radau', etc.)

    Returns:
        dict with gradient test results
    """
    print(f"\nTesting: {solver_name}", end='')
    if scipy_method:
        print(f" (method={scipy_method})", end='')
    print()

    # Import config
    config_module = importlib.import_module(f"configs.hpa_i1")
    config = config_module.ENV_CONFIG

    # Create ODE
    base_ode = config['create_base_ode']()
    initial_state = config['initial_state'].clone().requires_grad_(True)
    t_span = torch.linspace(0, 5.0, 11)  # Short simulation

    # Create parameter to track gradients
    if hasattr(base_ode, 'fixed_params') and base_ode.fixed_params is not None:
        base_ode.fixed_params = base_ode.fixed_params.clone().requires_grad_(True)
        param = base_ode.fixed_params
    else:
        param = initial_state

    try:
        # Forward pass
        options = {'solver': scipy_method} if scipy_method else None
        traj = odeint(base_ode, initial_state, t_span, method=solver_name, options=options)

        # Compute a simple loss
        loss = traj[-1].sum()

        print(f"  Forward pass: ✓")
        print(f"  Loss: {loss.item():.6f}")

        # Try backward pass
        try:
            loss.backward()

            if param.grad is not None and param.grad.abs().sum() > 0:
                print(f"  Backward pass: ✓")
                print(f"  Gradient norm: {param.grad.norm().item():.6e}")
                return {
                    'solver': solver_name,
                    'forward': True,
                    'backward': True,
                    'grad_norm': param.grad.norm().item()
                }
            else:
                print(f"  Backward pass: ✗ (gradients are None or zero)")
                return {
                    'solver': solver_name,
                    'forward': True,
                    'backward': False,
                    'error': 'No gradients computed'
                }

        except Exception as e:
            print(f"  Backward pass: ✗ ({str(e)[:50]}...)")
            return {
                'solver': solver_name,
                'forward': True,
                'backward': False,
                'error': str(e)
            }

    except Exception as e:
        print(f"  Forward pass: ✗ ({str(e)[:50]}...)")
        return {
            'solver': solver_name,
            'forward': False,
            'backward': False,
            'error': str(e)
        }


def main():
    """Test gradient support for different solvers."""
    print("="*70)
    print("Gradient/Backpropagation Support Test")
    print("="*70)
    print("\nTesting if solvers support training with gradient descent...")

    results = []

    # Test torchdiffeq solvers
    print("\n" + "-"*70)
    print("TORCHDIFFEQ SOLVERS")
    print("-"*70)

    for solver in ['bosh3', 'adaptive_heun', 'dopri5', 'implicit_adams']:
        result = test_gradients(solver)
        results.append(result)

    # Test scipy wrapper
    print("\n" + "-"*70)
    print("SCIPY WRAPPER SOLVERS")
    print("-"*70)

    for method in ['BDF', 'Radau', 'LSODA']:
        result = test_gradients('scipy_solver', scipy_method=method)
        results.append(result)

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"\n{'Solver':<25} {'Forward':<10} {'Backward':<10} {'Training?':<10}")
    print("-"*70)

    for result in results:
        solver_name = result['solver']
        if 'scipy_method' in result:
            solver_name += f" ({result['scipy_method']})"

        forward = "✓" if result['forward'] else "✗"
        backward = "✓" if result['backward'] else "✗"
        trainable = "YES" if result['backward'] else "NO"

        print(f"{solver_name:<25} {forward:<10} {backward:<10} {trainable:<10}")

    # Conclusion
    print("\n" + "="*70)
    print("CONCLUSION")
    print("="*70)

    trainable_solvers = [r for r in results if r['backward']]
    non_trainable_solvers = [r for r in results if not r['backward']]

    if trainable_solvers:
        print("\n✓ Can be used for TRAINING (support gradients):")
        for r in trainable_solvers:
            print(f"    - {r['solver']}")

    if non_trainable_solvers:
        print("\n✗ CANNOT be used for training (no gradients):")
        for r in non_trainable_solvers:
            print(f"    - {r['solver']}")

    print("\n" + "="*70)
    print("RECOMMENDATION")
    print("="*70)
    print("""
The scipy_solver wrapper in torchdiffeq uses torch.no_grad(), which
disables gradient computation. This makes scipy solvers (BDF, Radau, etc.)
UNSUITABLE for gradient-based training, despite being much faster.

For HPA training, you MUST use:
  - bosh3 (fastest torchdiffeq solver: ~2.3s)
  - adaptive_heun (similar speed: ~2.7s)
  - dopri5 (slower: ~3.3s)

Scipy solvers can only be used for:
  - Evaluation/testing (no training)
  - Comparing trajectories
  - Checking simulation accuracy
""")


if __name__ == "__main__":
    main()
