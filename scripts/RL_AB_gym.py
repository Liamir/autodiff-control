"""TD3 reinforcement learning on the AB gym.

Usage:
    python scripts/RL_AB_gym.py search --n_trials=50 --steps_per_trial=20000
    python scripts/RL_AB_gym.py train --total_timesteps=200000
    python scripts/RL_AB_gym.py all
"""

import json
from collections import deque
from pathlib import Path

import fire
import numpy as np
import optuna
import torch
import wandb
from stable_baselines3 import TD3
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.noise import (
    NormalActionNoise,
    OrnsteinUhlenbeckActionNoise,
)
from wandb.integration.sb3 import WandbCallback

from rpasim.gyms import ABGym

SCRIPT_DIR = Path(__file__).parent
BEST_PARAMS_PATH = SCRIPT_DIR / "best_td3_params.json"
MODEL_DIR = SCRIPT_DIR.parent / "models"


# ---------------------------------------------------------------------------
# Environment factory
# ---------------------------------------------------------------------------

def make_env():
    """Create an ABGym instance with default settings."""
    def _init():
        return ABGym(
            reward_fn=lambda state, time: state[1] ** 2,
            initial_state=torch.tensor([0.0, 1.0]),
            time_horizon=10.0,
            dt=0.1,
            n_reward_steps=300,
            alpha=50.0,
            action_low=0.1,
            action_high=1.0,
        )
    return _init


# ---------------------------------------------------------------------------
# Hyperparameter sampler
# ---------------------------------------------------------------------------

NET_ARCH_MAP = {
    "small": [64, 64],
    "medium": [128, 128],
    "big": [256, 256],
}


def sample_td3_params(trial: optuna.Trial) -> dict:
    """Sample TD3 hyperparameters for an Optuna trial."""
    one_minus_gamma = trial.suggest_float("one_minus_gamma", 0.0001, 0.03, log=True)
    batch_size_pow = trial.suggest_int("batch_size_pow", 2, 11)
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 0.002, log=True)
    train_freq = trial.suggest_int("train_freq", 1, 10)
    tau = trial.suggest_float("tau", 0.001, 0.08, log=True)
    noise_type = trial.suggest_categorical("noise_type", ["ornstein-uhlenbeck", "normal", "none"])
    noise_std = trial.suggest_float("noise_std", 0.0, 0.5)
    net_arch = trial.suggest_categorical("net_arch", ["small", "medium", "big"])

    trial.set_user_attr("gamma", 1 - one_minus_gamma)
    trial.set_user_attr("batch_size", 2 ** batch_size_pow)

    n_actions = 1
    action_noise = None
    if noise_type == "normal":
        action_noise = NormalActionNoise(
            mean=np.zeros(n_actions), sigma=noise_std * np.ones(n_actions),
        )
    elif noise_type == "ornstein-uhlenbeck":
        action_noise = OrnsteinUhlenbeckActionNoise(
            mean=np.zeros(n_actions), sigma=noise_std * np.ones(n_actions),
        )

    return {
        "gamma": 1 - one_minus_gamma,
        "learning_rate": learning_rate,
        "batch_size": 2 ** batch_size_pow,
        "train_freq": train_freq,
        "tau": tau,
        "action_noise": action_noise,
        "policy_kwargs": {"net_arch": NET_ARCH_MAP[net_arch]},
    }


# ---------------------------------------------------------------------------
# Custom W&B histogram callback
# ---------------------------------------------------------------------------

class WandbHistogramCallback(BaseCallback):
    """Log observation / reward histograms and gradient norms to W&B."""

    def __init__(self, log_freq: int = 1000, buffer_size: int = 2000, verbose: int = 0):
        super().__init__(verbose)
        self.log_freq = log_freq
        self.buffer_size = buffer_size
        self.obs_buffer: deque = deque(maxlen=buffer_size)
        self.reward_buffer: deque = deque(maxlen=buffer_size)

    def _on_step(self) -> bool:
        obs = self.locals.get("new_obs")
        reward = self.locals.get("rewards")
        if obs is not None:
            self.obs_buffer.extend(obs)
        if reward is not None:
            self.reward_buffer.extend(reward)

        if self.num_timesteps % self.log_freq == 0 and len(self.obs_buffer) > 0:
            obs_arr = np.array(self.obs_buffer)
            rew_arr = np.array(self.reward_buffer)

            log_dict = {
                "histograms/obs_A": wandb.Histogram(obs_arr[:, 0]),
                "histograms/obs_B": wandb.Histogram(obs_arr[:, 1]),
                "histograms/rewards": wandb.Histogram(rew_arr),
            }

            # Gradient norms
            for name, param in self.model.policy.named_parameters():
                if param.grad is not None:
                    log_dict[f"gradients/{name}_norm"] = param.grad.norm().item()

            wandb.log(log_dict, step=self.num_timesteps)

        return True


# ---------------------------------------------------------------------------
# Optuna objective
# ---------------------------------------------------------------------------

def _evaluate_policy(model, env, n_episodes: int = 5) -> float:
    """Run n_episodes and return mean total reward."""
    total_rewards = []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False
        ep_reward = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_reward += reward
            done = terminated or truncated
        total_rewards.append(ep_reward)
    return float(np.mean(total_rewards))


def _make_objective(steps_per_trial: int, n_eval_episodes: int):
    """Return an Optuna objective function (closure over budget)."""

    def objective(trial: optuna.Trial) -> float:
        params = sample_td3_params(trial)
        env = make_env()()
        eval_env = make_env()()

        model = TD3("MlpPolicy", env, verbose=0, **params)
        model.learn(total_timesteps=steps_per_trial)

        mean_reward = _evaluate_policy(model, eval_env, n_episodes=n_eval_episodes)
        env.close()
        eval_env.close()
        return mean_reward

    return objective


# ---------------------------------------------------------------------------
# CLI entry points
# ---------------------------------------------------------------------------

def search(n_trials: int = 50, steps_per_trial: int = 20000, n_eval_episodes: int = 5, n_jobs: int = 1):
    """Run Optuna hyperparameter search for TD3 on ABGym.

    Args:
        n_trials: Number of Optuna trials.
        steps_per_trial: Training steps per trial.
        n_eval_episodes: Episodes for evaluation after each trial.
        n_jobs: Number of parallel workers (-1 for all CPUs).
    """
    study = optuna.create_study(direction="maximize")
    objective = _make_objective(steps_per_trial, n_eval_episodes)
    study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs)

    print(f"\nBest trial reward: {study.best_trial.value:.4f}")
    print(f"Best params: {study.best_params}")

    # Save best params (only JSON-serializable keys)
    serializable = {k: v for k, v in study.best_params.items()}
    with open(BEST_PARAMS_PATH, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"Saved best params to {BEST_PARAMS_PATH}")


def _load_best_params() -> dict:
    """Load best params from JSON and reconstruct TD3-ready dict."""
    if not BEST_PARAMS_PATH.exists():
        print("No best_td3_params.json found, using defaults.")
        return {}

    with open(BEST_PARAMS_PATH) as f:
        raw = json.load(f)

    gamma = 1 - raw["one_minus_gamma"]
    batch_size = 2 ** raw["batch_size_pow"]
    net_arch = NET_ARCH_MAP[raw["net_arch"]]

    n_actions = 1
    noise_type = raw.get("noise_type", "none")
    noise_std = raw.get("noise_std", 0.1)
    action_noise = None
    if noise_type == "normal":
        action_noise = NormalActionNoise(
            mean=np.zeros(n_actions), sigma=noise_std * np.ones(n_actions),
        )
    elif noise_type == "ornstein-uhlenbeck":
        action_noise = OrnsteinUhlenbeckActionNoise(
            mean=np.zeros(n_actions), sigma=noise_std * np.ones(n_actions),
        )

    return {
        "gamma": gamma,
        "learning_rate": raw["learning_rate"],
        "batch_size": batch_size,
        "train_freq": raw["train_freq"],
        "tau": raw["tau"],
        "action_noise": action_noise,
        "policy_kwargs": {"net_arch": net_arch},
    }


def train(total_timesteps: int = 200000, log_freq: int = 1000):
    """Train TD3 on ABGym with W&B logging using best Optuna params."""
    params = _load_best_params()

    run = wandb.init(
        project="ab-gym-td3",
        config=params,
        sync_tensorboard=True,
    )

    env = make_env()()
    eval_env = make_env()()

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    model = TD3("MlpPolicy", env, verbose=1, tensorboard_log=f"runs/{run.id}", **params)

    callbacks = [
        WandbCallback(model_save_path=str(MODEL_DIR), verbose=1),
        WandbHistogramCallback(log_freq=log_freq),
        EvalCallback(
            eval_env,
            best_model_save_path=str(MODEL_DIR),
            eval_freq=5000,
            n_eval_episodes=5,
            deterministic=True,
        ),
    ]

    model.learn(total_timesteps=total_timesteps, callback=callbacks)

    final_path = MODEL_DIR / "td3_ab_final"
    model.save(str(final_path))
    print(f"Final model saved to {final_path}")

    env.close()
    eval_env.close()
    run.finish()


def all(n_trials: int = 50, steps_per_trial: int = 20000, total_timesteps: int = 200000, n_jobs: int = 1):
    """Run search then train."""
    search(n_trials=n_trials, steps_per_trial=steps_per_trial, n_jobs=n_jobs)
    train(total_timesteps=total_timesteps)


if __name__ == "__main__":
    fire.Fire({"search": search, "train": train, "all": all})
