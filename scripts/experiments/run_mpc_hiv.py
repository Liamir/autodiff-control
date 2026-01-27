#!/usr/bin/env python3
"""Run MPC on HIV treatment environment and visualize results."""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import matplotlib.pyplot as plt
from rpasim.ode.classic_control.hiv import HIVTreatment
from rpa_control.mpc import MPCController, MPCConfig, simulate_mpc
from configs.hiv import compute_steady_state


def main():
    """Run MPC on HIV treatment environment."""
    print("HIV Treatment MPC Control")
    print("=" * 60)

    # Create HIV ODE
    hiv_ode = HIVTreatment()

    # Compute steady state target
    x1_B, x3_B = compute_steady_state()
    print(f"\nHealthy steady state:")
    print(f"  x1_B (healthy CD4+): {x1_B:.4f}")
    print(f"  x3_B (CTL precursor): {x3_B:.4f}")

    # Initial condition from paper: x0 = (λ/d, 0.1, 0.1, 0.1, 0.1)
    x0 = torch.tensor([10.0, 0.1, 0.1, 0.1, 0.1])
    print(f"\nInitial state: {x0.tolist()}")

    # Reference state (healthy steady state for all variables)
    # We don't know steady state for all variables, so we use a reasonable target:
    # - x1 = x1_B (healthy CD4+)
    # - x2 = 0 (minimize infection)
    # - x3 = x3_B (CTL precursor)
    # - x4, x5 = steady state (unknown, use zero for now)
    reference_state = torch.tensor([x1_B, 0.0, x3_B, 0.0, 0.0])
    print(f"Reference state: {reference_state.tolist()}")

    # MPC Configuration from paper
    # Paper specs: mp = mc = 24 steps, ΔtM = 2 hrs
    # 24 steps × 2 hrs = 48 hrs = 2 days prediction horizon
    dt_hours = 2.0  # Time step in hours
    dt_days = dt_hours / 24.0  # Convert to days
    dt_weeks = dt_days / 7.0  # Convert to weeks

    mpc_config = MPCConfig(
        prediction_horizon=24,  # 24 steps (2 days ahead)
        dt=dt_weeks,  # 2 hours ≈ 0.0119 weeks
        Q=torch.tensor([1.0, 1.0]),  # Weights for x1 and x3
        Ru=1.0,  # Control magnitude weight (|u| term)
        R_deltau=0.0,  # No control rate-of-change cost in paper
        u_min=0.0,  # Minimum HAART level
        u_max=1.0,  # Maximum HAART level
        ftol=1e-4,
        disp=False,
        cost_type='l1',  # Use L1 cost as in paper: |x1-x̂1| + |x3-x̂3| + |u|
        tracked_state_indices=[0, 2],  # Only track x1 and x3
    )

    print(f"\nMPC Configuration:")
    print(f"  Prediction horizon: {mpc_config.prediction_horizon} steps (2 days)")
    print(f"  Time step: {dt_hours} hours = {dt_weeks:.6f} weeks")
    print(f"  Cost function: L1 (absolute values)")
    print(f"  Tracked states: x1 (healthy CD4+) and x3 (CTL precursor)")

    # Create MPC controller
    print("\nCreating MPC controller...")
    mpc = MPCController(
        ode=hiv_ode,
        config=mpc_config,
        reference_state=reference_state,
        n_controls=1,  # Single control input
    )
    print("MPC controller created successfully!")

    # Simulate MPC control
    print("\nSimulating MPC control...")
    time_horizon = 50.0  # 50 weeks (as in paper)
    times, states, controls = simulate_mpc(hiv_ode, mpc, x0, time_horizon, dt_weeks)
    print(f"Simulation complete! ({len(times)} time steps, {time_horizon} weeks)")

    # Print final state
    final_state = states[-1]
    print(f"\nFinal state:")
    print(f"  x1 (healthy CD4+): {final_state[0]:.4f} (target: {x1_B:.4f})")
    print(f"  x2 (infected CD4+): {final_state[1]:.4f} (target: 0.0)")
    print(f"  x3 (CTL precursor): {final_state[2]:.4f} (target: {x3_B:.4f})")
    print(f"  x4 (CTL indep): {final_state[3]:.4f}")
    print(f"  x5 (CTL dep): {final_state[4]:.4f}")

    # Compute final cost
    final_cost = abs(final_state[0] - x1_B) + abs(final_state[2] - x3_B)
    print(f"\nFinal tracking cost: {final_cost:.4f}")

    # Plot results
    print("\nGenerating plots...")
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))

    # Plot states
    ax = axes[0]
    state_names = ['healthy_CD4', 'infected_CD4', 'CTL_precursor', 'CTL_indep', 'CTL_dep']
    for i in range(5):
        ax.plot(times.numpy(), states[:, i].numpy(), label=state_names[i])

    # Add reference lines for x1 and x3
    ax.axhline(x1_B.item(), color='C0', linestyle='--', alpha=0.5, label='x1 target')
    ax.axhline(x3_B.item(), color='C2', linestyle='--', alpha=0.5, label='x3 target')

    ax.set_xlabel('Time (weeks)')
    ax.set_ylabel('State')
    ax.set_title('HIV Treatment: State Evolution with MPC')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Remove right and top spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    # Plot control
    ax = axes[1]
    control_times = times[:-1].numpy()  # Control is one step shorter
    ax.plot(control_times, controls.numpy(), label='u (HAART level)', color='C3')
    ax.axhline(0.0, color='k', linestyle='--', alpha=0.3)
    ax.axhline(1.0, color='k', linestyle='--', alpha=0.3)
    ax.set_xlabel('Time (weeks)')
    ax.set_ylabel('Control')
    ax.set_title('HIV Treatment: Control Input (HAART Therapy Level)')
    ax.set_ylim(-0.1, 1.1)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Remove right and top spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    plt.tight_layout()

    # Save plot
    output_path = Path(__file__).parent.parent / "plots" / "hiv_mpc_results.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved to: {output_path}")

    plt.show()


if __name__ == "__main__":
    main()
