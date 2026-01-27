"""Compare MPC controller with trained static/dynamic controllers.

This script runs both MPC and trained controllers on the same initial conditions
and compares their performance.
"""
import fire
import torch
from pathlib import Path
import matplotlib.pyplot as plt
from rpasim.ode.classic_control.population import PopulationDynamics
from rpa_control.mpc import MPCController, MPCConfig, simulate_mpc
from rpa_control.controllers import StaticController, ControlledODE
from rpa_control.style import set_style
from torchdiffeq import odeint


def simulate_controlled_ode(
    controlled_ode: ControlledODE,
    x0: torch.Tensor,
    time_horizon: float,
    dt: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Simulate controlled ODE system.

    Args:
        controlled_ode: ControlledODE with trained controller
        x0: Initial state (base state only, controller states added automatically)
        time_horizon: Total simulation time
        dt: Time step

    Returns:
        times, states (base states only), controls
    """
    # Augment with controller states if needed
    if hasattr(controlled_ode.controller, 'n_controller_states'):
        n_controller_states = controlled_ode.controller.n_controller_states
        controller_init = torch.zeros(n_controller_states)
        x0_full = torch.cat([x0, controller_init])
    else:
        x0_full = x0

    # Integrate
    n_steps = int(time_horizon / dt)
    t_span = torch.linspace(0, time_horizon, n_steps + 1)
    states_full = odeint(controlled_ode, x0_full, t_span, method='rk4')

    # Extract base states
    n_base = len(x0)
    states_base = states_full[:, :n_base]

    # Compute control inputs at each time step
    controls = []
    for state in states_full:
        u = controlled_ode.controller(state[:n_base])
        controls.append(u)
    controls = torch.stack(controls[:-1])  # Exclude last time point

    return t_span, states_base, controls


def compare_controllers(
    # Initial condition
    initial_state: list[float] = [80.0, 25.0],
    # Simulation settings
    time_horizon: float = 20.0,
    dt: float = 0.1,
    # MPC settings (paper parameters)
    mpc_horizon: int = 10,  # m_p = m_c = 10
    mpc_dt: float = 0.1,
    mpc_Q: float = 1.0,  # Q = I_2x2
    mpc_Ru: float = 0.5,  # Control magnitude penalty
    mpc_R: float = 0.5,  # Rate-of-change penalty (R_deltau)
    mpc_u_min: float = -20.0,  # u \in [-20, 20]
    mpc_u_max: float = 20.0,
    mpc_state_min: float = 0.0,  # Minimum population size
    # Trained controller settings
    trained_controller_path: str = None,  # Path to trained controller
    controller_order: int = 2,
    include_constant: bool = True,
    # Output
    save_plot: bool = True,
    output_dir: str = "plots/mpc_comparison",
):
    """Compare MPC with trained controller on population dynamics.

    Args:
        initial_state: Initial state [prey, predator]
        time_horizon: Simulation time horizon
        dt: Time step for simulation
        mpc_horizon: MPC prediction horizon (m_p = m_c from paper)
        mpc_dt: MPC time step
        mpc_Q: State tracking weight (Q = I_2x2 from paper)
        mpc_Ru: Control magnitude penalty
        mpc_R: Control rate-of-change penalty (R_deltau from paper)
        mpc_u_min: Minimum control input
        mpc_u_max: Maximum control input
        mpc_state_min: Minimum state value (population constraint)
        trained_controller_path: Path to saved controller parameters (if None, train new one)
        controller_order: Polynomial order for trained controller
        include_constant: Include constant term in trained controller
        save_plot: Whether to save comparison plots
        output_dir: Directory for output plots
    """
    set_style()

    print("="*60)
    print("MPC vs Trained Controller Comparison")
    print("="*60)
    print()

    # Create base ODE
    pop_ode = PopulationDynamics()
    print(f"System: {pop_ode}")
    print()

    # Critical point (target state)
    critical_point = torch.tensor([100.0, 20.0])
    x0 = torch.tensor(initial_state, dtype=torch.float32)

    print(f"Initial state: prey={x0[0]:.1f}, predator={x0[1]:.1f}")
    print(f"Target state: prey={critical_point[0]:.1f}, predator={critical_point[1]:.1f}")
    print()

    # ========== MPC Controller ==========
    print("Setting up MPC controller...")
    # Paper parameters: Q = I_2x2, Ru = R_deltau = 0.5, horizon = 10
    mpc_config = MPCConfig(
        prediction_horizon=mpc_horizon,
        dt=mpc_dt,
        Q=torch.tensor([mpc_Q, mpc_Q]),
        Ru=mpc_Ru,
        R_deltau=mpc_R,
        u_min=mpc_u_min,
        u_max=mpc_u_max,
        state_min=torch.tensor([mpc_state_min, mpc_state_min]),
        ftol=1e-3,
    )

    mpc = MPCController(
        ode=pop_ode,
        config=mpc_config,
        reference_state=critical_point,
        control_indices=[1],  # Control affects predator
    )

    print(f"  Prediction horizon: {mpc_horizon}")
    print(f"  Q weights: [{mpc_Q}, {mpc_Q}]")
    print(f"  Ru (control magnitude): {mpc_Ru}")
    print(f"  R_deltau (rate-of-change): {mpc_R}")
    print(f"  Control bounds: [{mpc_u_min}, {mpc_u_max}]")
    print(f"  State constraints: >= {mpc_state_min}")
    print()

    print("Simulating with MPC...")
    times_mpc, states_mpc, controls_mpc = simulate_mpc(
        pop_ode, mpc, x0, time_horizon, dt
    )
    print(f"  Final state: prey={states_mpc[-1, 0]:.2f}, predator={states_mpc[-1, 1]:.2f}")
    print(f"  Control range: [{controls_mpc.min():.2f}, {controls_mpc.max():.2f}]")
    print()

    # ========== Trained Controller ==========
    print("Setting up trained controller...")

    # Create static controller
    controller = StaticController(
        n_state_vars=2,
        n_control_vars=1,
        order=controller_order,
        include_constant=include_constant,
    )

    # Load or train controller
    if trained_controller_path is not None:
        print(f"  Loading from: {trained_controller_path}")
        import json
        from pathlib import Path

        # Try to load from results file
        results_path = Path(trained_controller_path)
        if results_path.is_dir():
            results_file = results_path / "results.txt"
        else:
            results_file = results_path

        if results_file.exists():
            # Parse best_params_physical from results file (rescaled parameters that work without basis normalization)
            with open(results_file, 'r') as f:
                for line in f:
                    if line.startswith('best_params_physical:'):
                        params_str = line.split(':', 1)[1].strip()
                        params_list = eval(params_str)  # Safe here since it's our own file
                        # Flatten if nested
                        if isinstance(params_list[0], list):
                            params_list = params_list[0]
                        # Reshape to (n_control_vars, n_basis) = (1, 6)
                        controller.params.data = torch.tensor(params_list, dtype=torch.float32).reshape(1, -1)
                        print(f"  Loaded physical parameters: {params_list}")
                        break
        else:
            print(f"  Warning: File not found, using default initialization")
    else:
        print("  Using default initialization (untrained)")

    controlled_ode = ControlledODE(
        base_ode=pop_ode,
        controller=controller,
        control_indices=[1],
    )

    print("Simulating with trained controller...")
    times_trained, states_trained, controls_trained = simulate_controlled_ode(
        controlled_ode, x0, time_horizon, dt
    )
    print(f"  Final state: prey={states_trained[-1, 0]:.2f}, predator={states_trained[-1, 1]:.2f}")
    print(f"  Control range: [{controls_trained.min():.2f}, {controls_trained.max():.2f}]")
    print()

    # ========== Comparison ==========
    print("Performance Metrics:")
    print("-"*60)

    # Compute tracking error
    error_mpc = torch.sum((states_mpc - critical_point) ** 2, dim=1).sqrt()
    error_trained = torch.sum((states_trained - critical_point) ** 2, dim=1).sqrt()

    mse_mpc = torch.mean((states_mpc - critical_point) ** 2).item()
    mse_trained = torch.mean((states_trained - critical_point) ** 2).item()

    # Compute control effort
    control_effort_mpc = torch.sum(controls_mpc ** 2).item()
    control_effort_trained = torch.sum(controls_trained ** 2).item()

    print(f"MPC:")
    print(f"  Mean Squared Error: {mse_mpc:.4f}")
    print(f"  Final tracking error: {error_mpc[-1]:.4f}")
    print(f"  Total control effort: {control_effort_mpc:.4f}")
    print()

    print(f"Trained Controller:")
    print(f"  Mean Squared Error: {mse_trained:.4f}")
    print(f"  Final tracking error: {error_trained[-1]:.4f}")
    print(f"  Total control effort: {control_effort_trained:.4f}")
    print()

    # ========== Plotting ==========
    if save_plot:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        fig, axes = plt.subplots(3, 2, figsize=(12, 12))

        # State trajectories - Prey
        ax = axes[0, 0]
        ax.plot(times_mpc.numpy(), states_mpc[:, 0].numpy(), 'b-', label='MPC', linewidth=2)
        ax.plot(times_trained.numpy(), states_trained[:, 0].detach().numpy(), 'r--', label='Trained', linewidth=2)
        ax.axhline(critical_point[0].item(), color='k', linestyle=':', label='Target', alpha=0.5)
        ax.set_xlabel('Time')
        ax.set_ylabel('Prey Population')
        ax.set_title('Prey Dynamics')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # State trajectories - Predator
        ax = axes[0, 1]
        ax.plot(times_mpc.numpy(), states_mpc[:, 1].numpy(), 'b-', label='MPC', linewidth=2)
        ax.plot(times_trained.numpy(), states_trained[:, 1].detach().numpy(), 'r--', label='Trained', linewidth=2)
        ax.axhline(critical_point[1].item(), color='k', linestyle=':', label='Target', alpha=0.5)
        ax.set_xlabel('Time')
        ax.set_ylabel('Predator Population')
        ax.set_title('Predator Dynamics')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Control inputs
        ax = axes[1, 0]
        ax.plot(times_mpc[:-1].numpy(), controls_mpc.numpy(), 'b-', label='MPC', linewidth=2)
        ax.plot(times_trained[:-1].numpy(), controls_trained.detach().numpy(), 'r--', label='Trained', linewidth=2)
        ax.set_xlabel('Time')
        ax.set_ylabel('Control Input')
        ax.set_title('Control Signals')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Tracking error
        ax = axes[1, 1]
        ax.plot(times_mpc.numpy(), error_mpc.numpy(), 'b-', label='MPC', linewidth=2)
        ax.plot(times_trained.numpy(), error_trained.detach().numpy(), 'r--', label='Trained', linewidth=2)
        ax.set_xlabel('Time')
        ax.set_ylabel('Tracking Error')
        ax.set_title('Distance to Target')
        ax.set_yscale('log')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Accumulated control effort
        ax = axes[2, 0]
        cumulative_effort_mpc = torch.cumsum(controls_mpc ** 2, dim=0) * dt
        cumulative_effort_trained = torch.cumsum(controls_trained ** 2, dim=0) * dt
        ax.plot(times_mpc[:-1].numpy(), cumulative_effort_mpc.numpy(), 'b-', label='MPC', linewidth=2)
        ax.plot(times_trained[:-1].numpy(), cumulative_effort_trained.detach().numpy(), 'r--', label='Trained', linewidth=2)
        ax.set_xlabel('Time')
        ax.set_ylabel('Accumulated Control Effort')
        ax.set_title('Cumulative Control Cost')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Accumulated total cost (state error + control)
        ax = axes[2, 1]
        # MPC cost: Q*error^2 + Ru*u^2
        state_cost_mpc = torch.sum((states_mpc - critical_point) ** 2 * torch.tensor([mpc_Q, mpc_Q]), dim=1)
        control_cost_mpc = torch.cat([torch.zeros(1), mpc_Ru * (controls_mpc ** 2).squeeze()])
        total_cost_mpc = state_cost_mpc + control_cost_mpc
        cumulative_cost_mpc = torch.cumsum(total_cost_mpc, dim=0) * dt

        state_cost_trained = torch.sum((states_trained - critical_point) ** 2 * torch.tensor([mpc_Q, mpc_Q]), dim=1)
        control_cost_trained = torch.cat([torch.zeros(1), mpc_Ru * (controls_trained ** 2).squeeze()])
        total_cost_trained = state_cost_trained + control_cost_trained
        cumulative_cost_trained = torch.cumsum(total_cost_trained, dim=0) * dt

        ax.plot(times_mpc.numpy(), cumulative_cost_mpc.numpy(), 'b-', label='MPC', linewidth=2)
        ax.plot(times_trained.numpy(), cumulative_cost_trained.detach().numpy(), 'r--', label='Trained', linewidth=2)
        ax.set_xlabel('Time')
        ax.set_ylabel('Accumulated Total Cost')
        ax.set_title('Cumulative Objective (State + Control)')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        filename = output_path / 'mpc_vs_trained_comparison.pdf'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Saved plot to: {filename}")

    plt.show()


if __name__ == "__main__":
    fire.Fire(compare_controllers)
