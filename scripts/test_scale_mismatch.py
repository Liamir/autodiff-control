"""Test scale-aware regularization with extreme scale mismatch.

Creates a variant of Lotka-Volterra where prey is measured in units of 10^-6
to create extreme scale difference: prey ~ 0.0001 vs predator ~ 20.
"""
import fire
import torch
from rpasim.ode.base import ODE
from rpasim.env.base import DifferentiableEnv
from rpa_control.controllers import StaticController, ControlledODE
from rpa_control.optimization.gradient import train_ode_parameters, TrainingConfig
from rpa_control.style import set_style


class RescaledPopulationDynamics(ODE):
    """Lotka-Volterra with prey measured in millions (10^-6 scale).

    Original system:
        dP/dt = 0.5*P - 0.005*P*D
        dD/dt = -1.0*D + 0.01*P*D
        Critical point: P=100, D=100

    Rescaled system (prey in millions, P_scaled = P / 10^6):
        dP_scaled/dt = 0.5*P_scaled - 0.005*P_scaled*D
        dD/dt = -1.0*D + (10^6 * 0.01)*P_scaled*D
        Critical point: P_scaled=0.0001, D=100
    """

    def __init__(self):
        super().__init__()
        self.name = "rescaled_population"
        self.variable_names = ["prey_millions", "predator"]

        # Parameters adjusted for rescaled prey
        # a, b stay same; c stays same; d multiplied by 10^6
        self.fixed_params = torch.tensor([
            0.5,      # a: prey growth rate
            0.005,    # b: predation rate
            1.0,      # c: predator death rate
            1e4,      # d: predator growth from prey (10^6 * 0.01 = 10000)
        ])

    def forward(self, t, state, differentiable_params=None, fixed_params=None):
        if fixed_params is None:
            fixed_params = self.fixed_params

        prey, predator = state[0], state[1]
        a, b, c, d = fixed_params[0], fixed_params[1], fixed_params[2], fixed_params[3]

        dprey_dt = a * prey - b * prey * predator
        dpredator_dt = -c * predator + d * prey * predator

        return torch.stack([dprey_dt, dpredator_dt])

    def __str__(self):
        a, b, c, d = self.fixed_params
        return (
            f"Rescaled Population Dynamics (prey in millions)\n"
            f"  dP/dt = {a:.3f}*P - {b:.3f}*P*D\n"
            f"  dD/dt = -{c:.3f}*D + {d:.1f}*P*D\n"
            f"  Critical point: P={c/d:.6f}, D={a/b:.1f}\n"
            f"  Scale: P ~ 1e-4, D ~ 100 (scale ratio ~ 1e6)"
        )


def reward_fn(state, time=None):
    """Reward: stabilize at rescaled critical point."""
    # Critical point: P = c/d = 1.0/10000 = 0.0001, D = a/b = 0.5/0.005 = 100.0
    critical_point = torch.tensor([0.0001, 100.0])  # prey in millions
    return -((state - critical_point) ** 2).sum()


def test_scale_mismatch(
    n_iterations: int = 500,
    learning_rate: float = 1e-3,
    l1_penalty: float = 2.0,
    std_penalty_multiplier: float = 100.0,
    time_horizon: float = 20.0,
    log_interval: int = 100,
):
    """
    Train controllers with and without scale-aware regularization on rescaled system.

    Args:
        n_iterations: Number of training iterations per run
        learning_rate: Learning rate for optimizer
        l1_penalty: L1 regularization coefficient for scale-aware method
        std_penalty_multiplier: Multiplier for standard regularization (to make comparison fair)
        time_horizon: Simulation time horizon
        log_interval: Print progress every N iterations
    """
    set_style()

    print("="*80)
    print("SCALE MISMATCH TEST: Extreme Variable Scaling")
    print("="*80)
    print()

    # Create rescaled population ODE
    pop_ode = RescaledPopulationDynamics()
    print(pop_ode)
    print()

    # Initial state (away from critical point, in rescaled units)
    # Critical point is [0.0001, 100], so start at [0.00008, 80] (20% below critical point)
    initial_state = torch.tensor([0.00008, 80.0])
    print(f"Initial state: prey={initial_state[0]:.6f} million, predator={initial_state[1]:.1f}")
    print(f"Critical point: prey={0.0001:.6f} million, predator={100.0:.1f}")
    print()

    # ============================================================================
    # Test 1: Standard Regularization
    # ============================================================================
    print("TEST 1: STANDARD REGULARIZATION")
    print("-"*80)

    # Create controller
    controller_std = StaticController(
        n_state_vars=2,
        n_control_vars=1,
        order=2,
        include_constant=True
    )

    # Create controlled ODE
    controlled_ode_std = ControlledODE(
        base_ode=pop_ode,
        controller=controller_std,
        control_indices=[1]  # Control affects predator
    )

    # Create environment
    env_std = DifferentiableEnv(
        initial_ode=controlled_ode_std,
        reward_fn=reward_fn,
        initial_state=initial_state,
        time_horizon=time_horizon,
        n_reward_steps=100,
    )

    # Training config with scaled-up penalty for fair comparison
    std_l1_penalty = l1_penalty * std_penalty_multiplier
    config_std = TrainingConfig(
        n_iterations=n_iterations,
        learning_rate=learning_rate,
        l1_penalty=std_l1_penalty,
        log_interval=log_interval,
        verbose=True,
        steady_state_fraction=0.5,
        scale_aware_regularization=False,  # Standard regularization
    )

    print(f"Training with standard regularization (L1 penalty = {std_l1_penalty:.1f})...")
    print()
    history_std = train_ode_parameters(
        env=env_std,
        ode=controlled_ode_std,
        config=config_std,
    )

    print()
    print("Standard regularization results:")
    print(f"  Best reward: {history_std['best_reward']:.3f}")
    print(f"  Final reward: {history_std['reward'][-1]:.3f}")
    print(f"  Final L1 penalty: {history_std['l1_penalty'][-1]:.3f}")
    print(f"  Non-zero params: {history_std['num_nonzero_params'][-1]}")
    print()
    print("Learned controller:")
    print(controlled_ode_std.get_controller_summary(['prey_millions', 'predator'], ['u']))
    print()
    print()

    # ============================================================================
    # Test 2: Scale-Aware Regularization
    # ============================================================================
    print("TEST 2: SCALE-AWARE REGULARIZATION")
    print("-"*80)

    # Create fresh controller
    controller_scale = StaticController(
        n_state_vars=2,
        n_control_vars=1,
        order=2,
        include_constant=True
    )

    # Create controlled ODE
    controlled_ode_scale = ControlledODE(
        base_ode=pop_ode,
        controller=controller_scale,
        control_indices=[1]
    )

    # Create environment
    env_scale = DifferentiableEnv(
        initial_ode=controlled_ode_scale,
        reward_fn=reward_fn,
        initial_state=initial_state,
        time_horizon=time_horizon,
        n_reward_steps=100,
    )

    # Training config with scale-aware regularization
    config_scale = TrainingConfig(
        n_iterations=n_iterations,
        learning_rate=learning_rate,
        l1_penalty=l1_penalty,
        log_interval=log_interval,
        verbose=True,
        steady_state_fraction=0.5,
        scale_aware_regularization=True,  # Scale-aware regularization
        reg_scale_update_interval=0,
    )

    print(f"Training with scale-aware regularization (L1 penalty = {l1_penalty:.1f})...")
    print()
    history_scale = train_ode_parameters(
        env=env_scale,
        ode=controlled_ode_scale,
        config=config_scale,
    )

    print()
    print("Scale-aware regularization results:")
    print(f"  Best reward: {history_scale['best_reward']:.3f}")
    print(f"  Final reward: {history_scale['reward'][-1]:.3f}")
    print(f"  Final L1 penalty: {history_scale['l1_penalty'][-1]:.3f}")
    print(f"  Non-zero params: {history_scale['num_nonzero_params'][-1]}")
    print()
    print("Learned controller:")
    print(controlled_ode_scale.get_controller_summary(['prey_millions', 'predator'], ['u']))
    print()
    print()

    # ============================================================================
    # Comparison
    # ============================================================================
    print("="*80)
    print("COMPARISON")
    print("="*80)
    print(f"Standard regularization (L1 penalty = {std_l1_penalty:.1f}):")
    print(f"  Best reward: {history_std['best_reward']:8.3f}")
    print(f"  Final L1:    {history_std['l1_penalty'][-1]:8.3f}")
    print(f"  Non-zero:    {history_std['num_nonzero_params'][-1]:3d}")
    print()
    print(f"Scale-aware regularization (L1 penalty = {l1_penalty:.1f}):")
    print(f"  Best reward: {history_scale['best_reward']:8.3f}")
    print(f"  Final L1:    {history_scale['l1_penalty'][-1]:8.3f}")
    print(f"  Non-zero:    {history_scale['num_nonzero_params'][-1]:3d}")
    print()

    reward_improvement = history_scale['best_reward'] - history_std['best_reward']
    print(f"Reward improvement with scale-aware: {reward_improvement:+.3f}")

    if reward_improvement > 0:
        print("✓ Scale-aware regularization performs better!")
    elif reward_improvement < -0.1:
        print("✗ Scale-aware regularization performs worse")
    else:
        print("≈ Both methods perform similarly")


if __name__ == "__main__":
    fire.Fire(test_scale_mismatch)
