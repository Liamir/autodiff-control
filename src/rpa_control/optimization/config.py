"""Training configuration dataclass."""
from typing import Optional, List
from dataclasses import dataclass
import torch


@dataclass
class TrainingConfig:
    """Configuration for gradient-based training."""
    n_iterations: int = 1000
    learning_rate: float = 1e-3
    l1_penalty: float = 0.0
    l2_penalty: float = 0.0
    log_interval: int = 10
    verbose: bool = True
    steady_state_fraction: float = 0.0  # Fraction of trajectory to skip before computing reward (0.0 = use all, 0.5 = use second half)
    perturb_param_indices: Optional[List[int]] = None  # Indices of fixed_params to perturb (None = no perturbation)
    perturb_fold_change: float = 2.0  # Perturb params by random factor in [1/fold, fold]
    scale_aware_regularization: bool = False  # Normalize basis functions by their RMS for numerical stability
    reg_scale_update_interval: int = 0  # Update basis scales every N iterations (0 = only compute once at start)
    # Gradient clipping
    gradient_clip_norm: float = 0.0  # Max gradient norm (0 = no clipping)
    # Evaluation on fixed initial conditions
    eval_interval: int = 0  # Evaluate every N iterations on fixed ICs (0 = no evaluation)
    eval_initial_states: Optional[List[torch.Tensor]] = None  # Fixed ICs for evaluation (None = no evaluation)
    n_param_samples_eval: int = 10  # Number of parameter samples per IC during evaluation (only used if perturb_param_indices is set)
    # Truncated Backpropagation Through Time (TBPTT)
    use_tbptt: bool = False  # Enable truncated backprop (reduces gradient computation length)
    tbptt_truncation_steps: int = 5  # Number of time steps to backprop through (0 = full BPTT)
    # Three-stage training pipeline (improved alternative to iterative LS)
    use_three_stage: bool = False  # Enable 3-stage: normal → L1 regularization → thresholding
    n_iterations_stage1: int = 1000  # Stage 1: normal training without regularization
    n_iterations_stage2: int = 500   # Stage 2: continued training with L1 regularization
    l1_penalty_stage2: float = 0.01  # L1 penalty coefficient for stage 2
    n_iterations_stage3: int = 500   # Stage 3: continued training after thresholding
    threshold_value_stage3: float = 1e-3  # Absolute threshold for stage 3: |θ| < threshold → 0
    # Deprecated: Iterative least squares with thresholding (replaced by use_three_stage)
    use_thresholding: bool = False  # DEPRECATED: Apply iterative LS: train, threshold, retrain until convergence
    threshold_value: float = 1e-3  # DEPRECATED: Absolute threshold: |θ| < threshold → 0
    max_threshold_rounds: int = 10  # DEPRECATED: Maximum number of threshold-retrain rounds
