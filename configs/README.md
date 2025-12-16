# Environment Configuration Guide

This directory contains environment-specific configurations for the generic training script (`scripts/train.py`).

## Quick Start

Train any environment using:
```bash
python scripts/train.py --config <config_name> [options]
```

Examples:
```bash
# AB Circuit - 1000 iterations
python scripts/train.py --config ab_circuit --n_iterations 1000

# Population - with parameter perturbations
python scripts/train.py --config population --controller_order 2 --perturb_params

# AB Circuit - with robust training
python scripts/train.py --config ab_circuit --perturb_params --perturb_fold_change 2.0
```

## Available Configurations

### `ab_circuit.py`
- **System**: AB Gene Regulatory Network
- **Trainable**: alpha parameters (ODE coefficients)
- **Fixed**: beta parameters
- **Objective**: Drive B variable to 1.0
- **Perturbation**: Can perturb beta1 and beta2 for robustness

### `population.py`
- **System**: Lotka-Volterra Population Dynamics
- **Trainable**: Static controller parameters (polynomial basis)
- **Fixed**: ODE parameters (a, b, c, d)
- **Objective**: Stabilize at critical point (prey=100, predator=20)
- **Perturbation**: Can perturb all 4 ODE parameters for robustness

## Configuration Structure

Each config file exports an `ENV_CONFIG` dictionary with:

```python
ENV_CONFIG = {
    # Basic info
    'name': 'environment_name',
    'experiment_name': 'experiment_prefix',

    # Core functions
    'create_ode': function,  # Creates ODE instance
    'reward_fn': function,   # Reward function

    # Initial conditions
    'initial_state': torch.Tensor,
    'initial_state_range': list or None,
    'eval_initial_states': list[torch.Tensor],

    # Perturbations
    'perturb_param_indices': list or None,

    # Display
    'param_names': list or None,
    'state_var_names': list,
    'target_var_idx': int or None,
    'target_value': float or None,
    'description': str,

    # Defaults (can be overridden via CLI)
    'defaults': {
        'time_horizon': float,
        'learning_rate': float,
        'n_iterations': int,
        'log_interval': int,
        'eval_interval': int,
        'steady_state_fraction': float,
        'scale_aware_regularization': bool,
        # ... other training parameters
    }
}
```

## Command-Line Options

### Training Parameters
- `--n_iterations`: Number of training iterations
- `--learning_rate`: Learning rate for optimizer
- `--l1_penalty`: L1 regularization (sparsity)
- `--l2_penalty`: L2 regularization
- `--time_horizon`: Simulation time horizon
- `--log_interval`: Print progress every N iterations
- `--eval_interval`: Evaluate on fixed ICs every N iterations
- `--steady_state_fraction`: Fraction of trajectory to skip for reward

### Regularization
- `--scale_aware_regularization`: Use basis normalization (default from config)

### Three-Stage Training
- `--use_three_stage`: Enable 3-stage training (normal → L1 → threshold)
- `--n_iterations_stage1`: Stage 1 iterations
- `--n_iterations_stage2`: Stage 2 iterations
- `--l1_penalty_stage2`: L1 penalty for stage 2
- `--n_iterations_stage3`: Stage 3 iterations
- `--threshold_value_stage3`: Threshold for stage 3

### Perturbation Training
- `--perturb_params`: Enable parameter perturbations
- `--perturb_fold_change`: Fold change for perturbations (default: 2.0)
- `--n_param_samples_eval`: Number of parameter samples per IC in eval (default: 10)

### Controller-Specific (for population)
- `--controller_order`: Polynomial order for controller (default from config)
- `--include_constant`: Include constant term in controller

## Creating New Configurations

To add a new environment:

1. Create a new config file in `configs/` (e.g., `my_env.py`)
2. Define `reward_fn()` and `create_ode()` functions
3. Export `ENV_CONFIG` dictionary with all required fields
4. Test with: `python scripts/train.py --config my_env`

## Examples

### Quick Test Run
```bash
python scripts/train.py --config ab_circuit --n_iterations 100 --log_interval 25
```

### Production Training with Perturbations
```bash
python scripts/train.py --config ab_circuit \
  --n_iterations 1000 \
  --perturb_params \
  --perturb_fold_change 2.0 \
  --n_param_samples_eval 10 \
  --eval_interval 100
```

### Population with Custom Controller Order
```bash
python scripts/train.py --config population \
  --controller_order 3 \
  --n_iterations 500 \
  --learning_rate 0.05 \
  --scale_aware_regularization
```

### Three-Stage Training for Sparsity
```bash
python scripts/train.py --config population \
  --use_three_stage \
  --n_iterations_stage1 500 \
  --n_iterations_stage2 300 \
  --l1_penalty_stage2 0.01 \
  --n_iterations_stage3 200 \
  --threshold_value_stage3 0.001
```

## Output

All experiments create timestamped directories in `logs/`:
```
logs/
  circuit_ab_20251216_165941/
    config.json          # Full configuration
    history.json         # Training history
    results.txt          # Final results
    console_output.txt   # All console output
    trajectories.pdf     # Trajectory plots
    training.pdf         # Training curves
```
