"""Input/output utilities for training results."""
import json
from pathlib import Path


def save_training_results(
    logger,
    history,
    ode,
    env_config,
    eval_interval=0,
):
    """Save training results including history and parameter summary.

    Args:
        logger: ExperimentLogger instance
        history: Training history dict
        ode: Trained ODE instance
        env_config: Environment configuration dict
        eval_interval: Evaluation interval

    Returns:
        dict: Summary statistics
    """
    # Generate parameter summary for saving
    param_names = env_config.get('param_names', None)
    param_summary = ""

    if hasattr(ode, 'get_controller_summary'):
        # Controller-based ODE
        state_var_names = env_config.get('state_var_names', None)
        control_names = env_config.get('control_names', ['u'])
        param_summary = ode.get_controller_summary(state_var_names, control_names)
    elif param_names is not None:
        # Direct parameter ODE (like AB circuit)
        if hasattr(ode, 'differentiable_params'):
            params = ode.differentiable_params.detach()
            for name, value in zip(param_names, params):
                param_summary += f"  {name} = {value.item()}\n"

    # Log results with full precision
    logger.log_history(history)

    summary = {
        'final_loss': history['loss'][-1],
        'final_reward': history['reward'][-1],
        'best_reward': history['best_reward'],
        'num_nonzero_params': history['num_nonzero_params'][-1],
        'best_params': history['best_params'].tolist(),
        'total_iterations': len(history['loss']),
    }

    if eval_interval > 0 and 'eval_reward' in history:
        eval_rewards = [r for r in history['eval_reward'] if r is not None]
        if eval_rewards:
            summary['best_eval_reward'] = max(eval_rewards)

    logger.log_results(summary, param_summary)

    return summary


def save_testing_results(
    logger,
    trajectory_data,
    training_model_params,
    testing_model_params,
    estimated_model_params_override=None,
):
    """Save testing results including trajectory and model parameters.

    Args:
        logger: ExperimentLogger instance
        trajectory_data: Dict with trajectory data (times, states, controls, reference_state)
        training_model_params: Dict of training model parameters
        testing_model_params: Dict of testing model parameters
        estimated_model_params_override: Optional estimated params override (for model mismatch)
    """
    # Save trajectory
    trajectory_path = logger.get_experiment_path() / 'trajectory.json'
    with open(trajectory_path, 'w') as f:
        json.dump(trajectory_data, f, indent=2)

    print(f"Trajectory saved to: {trajectory_path}")
    print()

    # Update config with testing parameters
    config_path = logger.get_experiment_path() / 'config.json'
    with open(config_path, 'r') as f:
        config_dict = json.load(f)

    config_dict['training_model_params'] = training_model_params.get('model_params')
    config_dict['testing_model_params'] = testing_model_params.get('model_params')
    config_dict['estimated_model_params_override'] = estimated_model_params_override

    with open(config_path, 'w') as f:
        json.dump(config_dict, f, indent=2)
