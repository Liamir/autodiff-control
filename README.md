# Autodiff Control

A framework for learning sparse polynomial controllers for dynamical systems using automatic differentiation and gradient-based optimization.

## Overview

This project implements a system for controlling ODE-based dynamical systems by learning sparse polynomial controllers through differentiable simulation. The framework supports both **static** and **dynamic** controllers, with built-in sparsification techniques to discover interpretable control laws.

## Key Concepts

### Controller Types

**Static Controller**: Direct state feedback control
```
u = � � �(X)
```
- Observes system state X directly
- Produces control output u via polynomial basis functions
- No internal memory/state

**Dynamic Controller**: Controller with internal state
```
dC/dt = �_obs � �(X, C)  (observing dynamics)
u = �_act � �(C)          (actuating output)
```
- Maintains internal controller state C
- Observing parameters learn controller dynamics based on system state
- Actuating parameters map controller state to control output
- Enables more complex, memory-based control strategies

### Polynomial Basis Functions

Controllers use polynomial basis functions up to specified order. For state [x�, x�] with order=2:
```
�(x) = [1, x�, x�, x��, x��, x��x�]
```

This allows learning nonlinear control laws as sparse linear combinations of basis terms.

### Scale-Aware Regularization

To handle systems with variables at different scales (e.g., A ~ 1, B ~ 100):
1. **Basis Normalization**: Normalize each basis function by its RMS over trajectories
2. **Uniform L1 Penalty**: Apply equal sparsity pressure to all (normalized) parameters
3. **Physical Interpretation**: Convert back to physical parameters for display

This ensures fair comparison across parameters regardless of state variable magnitudes.

## Project Structure

```
autodiff-control/
├── scripts/
│   ├── train.py                  # Generic training script
│   ├── run_mpc.py                # MPC control simulation
│   ├── compare_trajectories.py  # Compare any two experiments
│   └── analyze.py                # Analysis and plotting
├── configs/
│   ├── population.py             # Lotka-Volterra system
│   ├── flight.py                 # Flight control
│   ├── hiv.py                    # HIV treatment model
│   └── ab_circuit.py             # AB gene regulatory network
├── src/rpa_control/
│   ├── controllers/
│   │   ├── static.py             # Static state feedback controller
│   │   ├── dynamic.py            # Dynamic controller with internal state
│   │   └── controlled_ode.py    # ODE wrapper with controller
│   ├── optimization/
│   │   └── gradient.py           # Gradient-based training with sparsification
│   ├── mpc/
│   │   └── controller.py         # Model Predictive Control
│   └── utils/
│       ├── plotting.py           # Visualization utilities
│       └── logging.py            # Experiment logging (full precision)
└── logs/                         # Experiment outputs (auto-generated)
    └── experiment_name_timestamp/
        ├── config.json           # Full configuration (includes model params)
        ├── history.json          # Training history (full precision, training only)
        ├── results.txt           # Summary and controller equations
        ├── console.log           # Complete console output
        ├── trajectory.json       # Saved trajectory (training/MPC)
        ├── mpc_trajectory.json   # MPC trajectory (MPC only)
        ├── training.pdf          # Training curves (analysis)
        └── trajectories.pdf      # Trajectory plots (analysis)
```

## Installation

```bash
pip install -e .
```

Dependencies:
- PyTorch (automatic differentiation)
- torchdiffeq (differentiable ODE solver)
- rpasim (ODE simulation environment)
- numpy, matplotlib, seaborn (visualization)

## Usage

### Basic Training Example

```python
from rpasim.ode.classic_control.population import PopulationDynamics
from rpasim.env.base import DifferentiableEnv
from rpa_control.controllers import StaticController, ControlledODE
from rpa_control.optimization.gradient import train_ode_parameters, TrainingConfig

# Create base ODE system
ode = PopulationDynamics()

# Create controller
controller = StaticController(
    n_state_vars=2,      # prey, predator
    n_control_vars=1,    # control one variable
    order=2,             # quadratic polynomial
)

# Wrap ODE with controller
controlled_ode = ControlledODE(
    base_ode=ode,
    controller=controller,
    control_indices=[1]  # control affects predator
)

# Setup environment
env = DifferentiableEnv(
    initial_ode=controlled_ode,
    reward_fn=reward_fn,
    initial_state=initial_state,
    time_horizon=20.0,
)

# Train with three-stage pipeline
config = TrainingConfig(
    use_three_stage=True,
    n_iterations_stage1=500,    # Dense training
    n_iterations_stage2=300,    # L1 regularization
    n_iterations_stage3=300,    # Sparse refinement
    l1_penalty_stage2=0.01,
    threshold_value_stage3=1e-3,
    scale_aware_regularization=True,
)

history = train_ode_parameters(env, controlled_ode, config=config)
```

### Running Training and Analysis

**Training** (saves all results to logs directory):
```bash
# Train on any config
python scripts/train.py population --n_iterations 1000

# With three-stage training
python scripts/train.py population \
    --use_three_stage True \
    --n_iterations_stage1 500 \
    --n_iterations_stage2 300 \
    --n_iterations_stage3 300 \
    --l1_penalty_stage2 0.01

# With scale-aware regularization
python scripts/train.py population \
    --scale_aware_regularization True \
    --l1_penalty 0.01
```

**Model Mismatch Testing** (train on nominal model, test on true system):
```bash
# Train controller with perfect model knowledge
python scripts/train.py population --n_iterations 1000

# Train on nominal model but test on different parameters (model mismatch)
python scripts/train.py population --n_iterations 1000 \
    --test_model_params '{"a": 0.6, "b": 0.03}'
```

This tests controller robustness when the training model differs from the true system. The controller is trained using gradient-based optimization on the nominal model, but the final trajectory is simulated on the true model with different parameters.

**MPC (Model Predictive Control)**:
```bash
# Run MPC with perfect model knowledge
python scripts/run_mpc.py population --time_horizon 20

# Run MPC with imperfect model knowledge (planning model != true system)
python scripts/run_mpc.py population --time_horizon 20 \
    --test_model_params '{"a": 0.4, "b": 0.02}'
```

MPC uses the planning model for internal predictions/optimization, but the true system (always the default parameters from config) for actual simulation. This tests how well MPC handles model uncertainty.

**Compare Trajectories** (compare any two experiments):
```bash
# Compare MPC vs trained controller
python scripts/compare_trajectories.py \
    logs/population_static_mpc_20251231_154654 \
    logs/population_static_20251231_145238 \
    --output comparison.pdf

# Compare perfect vs imperfect model knowledge
python scripts/compare_trajectories.py \
    logs/experiment1 logs/experiment2 \
    --label1 "Perfect model" \
    --label2 "Model mismatch" \
    --output comparison.pdf
```

**Analysis** (view results and generate plots):
```bash
# Analyze trained experiment
python scripts/analyze.py analyze logs/population_static_20251230_155046

# Skip specific plots
python scripts/analyze.py analyze logs/experiment_dir --plot_training=False
```

**Plot Custom Controllers** (test specific controller parameters):
```bash
# Zero controller (no control)
python scripts/analyze.py plot --config population --controller_params zero

# Constant controller (u = 0.1)
python scripts/analyze.py plot --config population --controller_params "constant:0.1"

# Custom parameters
python scripts/analyze.py plot --config population \
    --controller_params "[-0.413, 0.665, -0.872, 1.598, 0.173, -1.213]" \
    --output my_controller.pdf
```

## Training Pipeline

### Three-Stage Training (Recommended)

Modern approach for learning sparse controllers:

**Stage 1: Dense Training** (no regularization)
- Learn full controller without sparsity constraints
- Find good control strategy

**Stage 2: L1 Regularization**
- Continue training with L1 penalty on (normalized) parameters
- Track parameters with lowest L1 norm (sparsest)
- Pushes small parameters toward zero

**Stage 3: Thresholding & Refinement**
- Apply hard threshold to sparsest params from stage 2
- Freeze zero parameters, retrain remaining ones
- Produces final sparse controller

**Why this works**: Stage 2 identifies which parameters can be zeroed with minimal performance loss. Stage 3 refines the sparse controller.

### Legacy: Iterative Least Squares (Deprecated)

Older approach using iterative threshold-retrain cycles. Replaced by three-stage pipeline.

## Model Mismatch Testing

Both training and MPC support testing controller robustness to model uncertainty:

### Training with Model Mismatch

**Architecture**:
- **TRAINING PHASE**: Controller parameters optimized using nominal/estimated model
- **TESTING PHASE**: Trained controller evaluated on true system with different parameters

**Logged Information**:
```json
{
  "training_model_params": {"a": 0.5, "b": 0.025, "c": 0.5, "d": 0.005},
  "testing_model_params": {"a": 0.6, "b": 0.03, "c": 0.5, "d": 0.005},
  "test_model_params_override": {"a": 0.6, "b": 0.03}
}
```

The saved trajectory (`trajectory.json`) is simulated using the **testing model** (true system).

### MPC with Model Mismatch

**Architecture**:
- **Planning ODE**: Model used inside MPC for predictions and optimization
- **Simulation ODE**: True system used for actual state evolution

When `--test_model_params` is provided:
- **Planning model**: Has incorrect parameters (what MPC thinks)
- **Simulation model**: Always uses default config parameters (the true system)

**Logged Information**:
```json
{
  "planning_model_params": {"a": 0.4, "b": 0.02, "c": 0.5, "d": 0.005},
  "simulation_model_params": {"a": 0.5, "b": 0.025, "c": 0.5, "d": 0.005},
  "test_model_params_override": {"a": 0.4, "b": 0.02}
}
```

This tests how well MPC's closed-loop feedback compensates for model errors.

## Target Variables System

The `target_vars` dictionary specifies which state variables to track and their target values:

```python
# Population: track both prey and predator
'target_vars': {0: 100.0, 1: 20.0}

# Flight: only track x1 (angle of attack), x2 and x3 are free
'target_vars': {0: 0.0}

# HIV: track x1 (healthy CD4+) and x3 (CTL precursor)
'target_vars': {0: 8.225466, 2: 1240.011475}
```

**Key Properties**:
- Only variables in `target_vars` contribute to tracking error/cost
- Other variables don't affect cost (completely free)
- Reference state is automatically constructed from `target_vars` (non-specified indices are zero)
- Used for both cost computation and plotting reference lines

**Example**:
```python
target_vars = {0: 100.0, 1: 20.0}  # Track prey and predator
# Constructs: reference_state = [100.0, 20.0]
# Cost: (prey - 100)² + (predator - 20)²

target_vars = {0: 0.0}  # Only track x1
# Constructs: reference_state = [0.0, 0.0, 0.0]
# Cost: (x1 - 0)², x2 and x3 ignored
```

## Recent Progress

### Latest Improvements

**Model Mismatch Testing**
- Train/test phase separation in `train.py`: optimize on nominal model, evaluate on true system
- MPC planning/simulation separation: MPC plans with imperfect model, simulates on true system
- Full model parameter logging for both training and testing phases
- Enables evaluation of controller robustness to model uncertainty

**Target Variables System**
- Replaced `target_var_idx`, `target_value`, and `reference_state` with unified `target_vars` dict
- Supports selective tracking: only specified state variables contribute to cost
- Automatic reference state construction from `target_vars`
- Flexible for multi-variable tracking (e.g., HIV: track x1 and x3, ignore x2, x4, x5)

**MPC Integration**
- Generic MPC script (`run_mpc.py`) works with any environment config
- Integrated with same logging infrastructure as training
- Trajectory comparison tool for MPC vs training experiments
- Support for model mismatch testing (planning vs simulation model)

**Workflow Improvements**
- Trajectory saving for both training and MPC (unified format)
- Generic comparison tool (`compare_trajectories.py`) for any two experiments
- Comprehensive config logging with model parameters and target variables
- Full precision data storage throughout pipeline

**Basis Normalization**
- Normalize basis functions by RMS for numerical stability
- Enables fair L1 penalties across different scales
- Improves convergence and sparsity quality

**Three-Stage Pipeline**
- More reliable sparsification than iterative LS
- Uses sparsest parameters from stage 2 (not best-reward)
- Better exploration-exploitation tradeoff

**Scale-Aware Regularization**
- Address multi-scale systems (variables spanning orders of magnitude)
- Normalize basis in forward pass, uniform L1 on scaled params
- Essential for systems like population dynamics

## Controller Parameter Naming

### Dynamic Controller
- **Observing parameters** (`�_obs`): Learn controller dynamics dC/dt from state (X, C)
- **Actuating parameters** (`�_act`): Map controller state C to control output u

### Static Controller
- **Parameters** (`�`): Direct mapping from state X to control u
- Could also be called "observing" (observes state to produce control)

## Configuration Options

### TrainingConfig Parameters

**Basic Training**:
- `n_iterations`: Total iterations (default: 1000)
- `learning_rate`: Optimizer learning rate (default: 1e-3)
- `l1_penalty`: L1 regularization coefficient (default: 0.0)
- `l2_penalty`: L2 regularization coefficient (default: 0.0)

**Advanced Features**:
- `scale_aware_regularization`: Enable basis normalization (default: False)
- `steady_state_fraction`: Skip initial transient (0.0 = use all trajectory)
- `gradient_clip_norm`: Gradient clipping (0 = disabled)
- `warmup_iterations`: LR warmup steps (default: 0)

**Three-Stage Training**:
- `use_three_stage`: Enable three-stage pipeline (default: False)
- `n_iterations_stage1`: Stage 1 iterations
- `n_iterations_stage2`: Stage 2 iterations
- `n_iterations_stage3`: Stage 3 iterations
- `l1_penalty_stage2`: L1 penalty for stage 2
- `threshold_value_stage3`: Hard threshold for stage 3

**Robustness Testing**:
- `perturb_param_indices`: Indices of fixed_params to perturb during training
- `perturb_fold_change`: Random perturbation range [1/fold, fold]

## Example Systems

- **Population Dynamics (Lotka-Volterra)**: Stabilize prey-predator critical point
- **Lorenz System**: Stabilize chaotic attractor
- **Circuit Models**: Control electronic circuits
- **Custom ODEs**: Any system implementing rpasim.ode.ODE interface

## Output Files

Each training run creates a timestamped directory in `logs/` containing:

1. **config.json**: Full configuration with all parameters (full precision)
2. **history.json**: Complete training history - loss, reward, parameters at each iteration (full precision)
3. **results.txt**: Human-readable summary with controller equations
4. **console.log**: Complete console output from training
5. **training.pdf**: Training curves (generated by `analyze.py`)
6. **trajectories.pdf**: Trajectory comparison plots (generated by `analyze.py`)

All numerical data is saved with **full precision** (no rounding) for reproducibility.

## Next Steps

See `claude.md` for development workflow preferences and `notebooks/` for exploration examples.
