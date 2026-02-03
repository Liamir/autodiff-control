# MPC with Custom Cost Functions

The MPC implementation now supports **arbitrary cost/reward functions** in addition to the default reference tracking objective.

## Changes Made

### 1. MPCConfig - New Parameter
- Added `stage_cost_fn` parameter: Optional custom stage cost function
- Signature: `stage_cost_fn(x_next, u, x_curr=None, k=None) -> float`
  - `x_next`: Next state (torch.Tensor)
  - `u`: Control input (torch.Tensor)
  - `x_curr`: Current state (optional, torch.Tensor)
  - `k`: Time step index (optional, int)
  - Returns: Scalar cost to minimize

### 2. MPCController - Flexible Initialization
- `reference_state` is now **optional** when using custom cost
- If `stage_cost_fn` is provided, `reference_state` can be `None`
- If `stage_cost_fn` is `None`, `reference_state` is **required** (default reference tracking)

### 3. Cost Function Behavior

**Default Reference Tracking (backward compatible):**
```python
cost = state_tracking_cost + control_magnitude_cost + control_change_cost
```

**Custom Cost Function:**
```python
cost = stage_cost_fn(x_next, u, x_curr, k) + control_change_cost
```

Note: `R_deltau` (control smoothness) is **always applied** for stability, even with custom costs.

## Usage Examples

### Example 1: Maximize |B| (ABControlled)

```python
import torch
from rpa_control.mpc import MPCController, MPCConfig

def maximize_B_cost(x_next, u, x_curr=None, k=None):
    """Maximize |B| while penalizing large control."""
    B = x_next[1]
    reward = torch.abs(B)
    # MPC minimizes cost, so negate reward
    cost = -reward + 0.01 * (u**2)
    return cost

config = MPCConfig(
    prediction_horizon=20,
    dt=0.1,
    stage_cost_fn=maximize_B_cost,  # Custom cost
    R_deltau=0.05,  # Control smoothness
    u_min=0.2,
    u_max=1.0,
)

mpc = MPCController(ode, config, reference_state=None)  # No reference needed!
```

### Example 2: Time-Dependent Cost

```python
def time_varying_cost(x_next, u, x_curr=None, k=None):
    """Cost that changes over prediction horizon."""
    # Higher penalty early in horizon, lower later
    time_weight = 1.0 - 0.5 * (k / 20)  # Decreases from 1.0 to 0.5

    state_cost = torch.sum(x_next**2) * time_weight
    control_cost = 0.1 * (u**2)

    return state_cost + control_cost
```

### Example 3: State-Dependent Reward

```python
def nonlinear_reward_cost(x_next, u, x_curr=None, k=None):
    """Nonlinear reward function (e.g., resonance, oscillation)."""
    x, y = x_next[0], x_next[1]

    # Reward being in a specific region
    target_radius = 5.0
    distance_to_target = torch.sqrt(x**2 + y**2)
    reward = torch.exp(-((distance_to_target - target_radius)**2) / 2.0)

    # Minimize cost = maximize reward
    return -reward + 0.01 * (u**2)
```

### Example 4: Using Environment Reward Function

If you have a reward function from training, you can reuse it for MPC:

```python
from configs.ab_controlled_mlp import reward_fn as training_reward_fn

def mpc_cost_from_reward(x_next, u, x_curr=None, k=None):
    """Convert training reward to MPC cost."""
    reward = training_reward_fn(x_next)  # Reward function from config
    control_penalty = 0.01 * (u**2)
    # MPC minimizes cost, training maximizes reward
    return -reward + control_penalty

config = MPCConfig(
    prediction_horizon=20,
    dt=0.1,
    stage_cost_fn=mpc_cost_from_reward,
    R_deltau=0.05,
    u_min=0.2,
    u_max=1.0,
)
```

## Backward Compatibility

All existing code using reference tracking continues to work unchanged:

```python
# Old way (still works!)
config = MPCConfig(
    prediction_horizon=20,
    dt=0.1,
    Q=[1.0, 1.0],  # State tracking weights
    Ru=0.1,        # Control magnitude penalty
    R_deltau=0.1,  # Control smoothness
    u_min=-1.0,
    u_max=1.0,
)

reference_state = torch.tensor([5.0, 10.0])
mpc = MPCController(ode, config, reference_state=reference_state)
```

## Cost vs Reward

- **MPC minimizes cost** (scipy.optimize.minimize)
- **Training maximizes reward**
- When converting reward to cost: `cost = -reward + regularization`

## Control Smoothness (R_deltau)

The `R_deltau` parameter (control rate-of-change penalty) is **always applied** regardless of cost function:
- Prevents chattering/oscillation in control signals
- Essential for physical systems with actuator constraints
- Recommended: `R_deltau >= 0.01`

## Testing

Run the test script to verify the custom cost function:
```bash
python test_custom_mpc_cost.py
```

Expected output: |B| should increase significantly (from ~2.0 to ~110+).
