"""RL training (PPO / TD3) on the AB gym.

Usage:
    python scripts/RL_AB_gym.py search --algo=ppo --n_trials=50 --steps_per_trial=20000
    python scripts/RL_AB_gym.py search --algo=td3 --n_trials=50 --steps_per_trial=20000
    python scripts/RL_AB_gym.py train  --algo=ppo --total_timesteps=200000
    python scripts/RL_AB_gym.py train  --algo=td3 --total_timesteps=200000
    python scripts/RL_AB_gym.py all    --algo=ppo
"""

import json
from collections import deque
from pathlib import Path

import fire
import matplotlib.pyplot as plt
import numpy as np
import optuna
import seaborn as sns
import torch
import wandb
from stable_baselines3 import PPO, TD3
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.noise import (
    NormalActionNoise,
    OrnsteinUhlenbeckActionNoise,
)
from wandb.integration.sb3 import WandbCallback

from rpasim.gyms import ABGym

SCRIPT_DIR = Path(__file__).parent
MODEL_DIR = SCRIPT_DIR.parent / "models"

ALGO_CLS = {"td3": TD3, "ppo": PPO}

ACTIVATION_FN_MAP = {
    "tanh": torch.nn.Tanh,
    "relu": torch.nn.ReLU,
}


def _best_params_path(algo: str) -> Path:
    return SCRIPT_DIR / f"best_{algo}_params.json"


# ---------------------------------------------------------------------------
# Environment factory
# ---------------------------------------------------------------------------


def make_env():
    """Create an ABGym instance with default settings."""

    def _init():
        return ABGym(
            reward_fn=lambda state, time: state[1] ** 2,
            initial_state=torch.tensor([0.0, 1.0]),
            # initial_state=torch.rand(2)*2,  # random initial state - two values in [0, 2]
            time_horizon=5,
            dt=0.1,
            n_reward_steps=150,
            alpha=50.0,
            action_low=0.1,
            action_high=1.0,
            ode_method="rk4",
        )

    return _init


# ---------------------------------------------------------------------------
# Hyperparameter samplers
# ---------------------------------------------------------------------------

TD3_NET_ARCH_MAP = {
    "small": [64, 64],
    "medium": [128, 128],
    "big": [256, 256],
}

PPO_NET_ARCH_MAP = {
    "tiny": dict(pi=[64], vf=[64]),
    "small": dict(pi=[64, 64], vf=[64, 64]),
    "medium": dict(pi=[128, 128], vf=[128, 128]),
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
    trial.set_user_attr("batch_size", 2**batch_size_pow)

    n_actions = 1
    action_noise = None
    if noise_type == "normal":
        action_noise = NormalActionNoise(
            mean=np.zeros(n_actions),
            sigma=noise_std * np.ones(n_actions),
        )
    elif noise_type == "ornstein-uhlenbeck":
        action_noise = OrnsteinUhlenbeckActionNoise(
            mean=np.zeros(n_actions),
            sigma=noise_std * np.ones(n_actions),
        )

    return {
        "gamma": 1 - one_minus_gamma,
        "learning_rate": learning_rate,
        "batch_size": 2**batch_size_pow,
        "train_freq": train_freq,
        "tau": tau,
        "action_noise": action_noise,
        "policy_kwargs": {"net_arch": TD3_NET_ARCH_MAP[net_arch]},
    }


def sample_ppo_params(trial: optuna.Trial) -> dict:
    """Sample PPO hyperparameters for an Optuna trial."""
    batch_size_pow = trial.suggest_int("batch_size_pow", 2, 10)
    n_steps_pow = trial.suggest_int("n_steps_pow", 5, 12)
    one_minus_gamma = trial.suggest_float("one_minus_gamma", 0.0001, 0.03, log=True)
    one_minus_gae_lambda = trial.suggest_float("one_minus_gae_lambda", 0.0001, 0.1, log=True)
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 0.002, log=True)
    ent_coef = trial.suggest_float("ent_coef", 0.00000001, 0.1, log=True)
    clip_range = trial.suggest_categorical("clip_range", [0.1, 0.2, 0.3, 0.4])
    n_epochs = trial.suggest_categorical("n_epochs", [1, 5, 10, 20])
    max_grad_norm = trial.suggest_float("max_grad_norm", 0.3, 2.0)
    net_arch = trial.suggest_categorical("net_arch", ["tiny", "small", "medium"])
    activation_fn = trial.suggest_categorical("activation_fn", ["tanh", "relu"])

    trial.set_user_attr("gamma", 1 - one_minus_gamma)
    trial.set_user_attr("n_steps", 2**n_steps_pow)
    trial.set_user_attr("batch_size", 2**batch_size_pow)

    # n_steps = 2**n_steps_pow
    n_steps = 150  # TODO - artificial - one epoch per env
    batch_size = 2**batch_size_pow
    # batch_size must divide n_steps for PPO
    if batch_size > n_steps:
        batch_size = n_steps

    return {
        "n_steps": n_steps,
        "batch_size": batch_size,
        "gamma": 1 - one_minus_gamma,
        "gae_lambda": 1 - one_minus_gae_lambda,
        "learning_rate": learning_rate,
        "ent_coef": ent_coef,
        "clip_range": clip_range,
        "n_epochs": n_epochs,
        "max_grad_norm": max_grad_norm,
        "policy_kwargs": {
            "net_arch": PPO_NET_ARCH_MAP[net_arch],
            "activation_fn": ACTIVATION_FN_MAP[activation_fn],
        },
    }


SAMPLER = {"td3": sample_td3_params, "ppo": sample_ppo_params}


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
# Plotting helpers
# ---------------------------------------------------------------------------


def plot_trajectory(model, env):
    """Run one episode and plot A(t), B(t)."""
    obs, _ = env.reset()
    observations = [obs.copy()]
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, _ = env.step(action)
        observations.append(obs.copy())
        done = terminated or truncated

    traj = np.array(observations)
    t = np.arange(len(traj)) * env.dt

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(t, traj[:, 0], label="A")
    ax.plot(t, traj[:, 1], label="B")
    ax.set_xlabel("time")
    ax.set_ylabel("state")
    ax.legend(frameon=False, bbox_to_anchor=(1, 1))
    sns.despine(ax=ax)
    fig.tight_layout()
    return fig


def plot_control_field(model, a_range=(0.0, 2.0), b_range=(0.0, 2.0), n_grid=50):
    """Plot the policy's control output u(A, B) as a heatmap."""
    a_vals = np.linspace(*a_range, n_grid)
    b_vals = np.linspace(*b_range, n_grid)
    aa, bb = np.meshgrid(a_vals, b_vals)
    states = np.stack([aa.ravel(), bb.ravel()], axis=1).astype(np.float32)

    actions = []
    for s in states:
        action, _ = model.predict(s, deterministic=True)
        actions.append(float(action[0]))
    uu = np.array(actions).reshape(n_grid, n_grid)

    fig, ax = plt.subplots(figsize=(6, 3))
    im = ax.pcolormesh(aa, bb, uu, shading="auto", cmap="viridis")
    fig.colorbar(im, ax=ax, label="u")
    ax.set_xlabel("A")
    ax.set_ylabel("B")
    sns.despine(ax=ax)
    fig.tight_layout()
    return fig


class WandbPlotCallback(BaseCallback):
    """Periodically log trajectory and control-field plots to W&B."""

    def __init__(self, log_freq: int = 10000, verbose: int = 0):
        super().__init__(verbose)
        self.log_freq = log_freq

    def _on_step(self) -> bool:
        if self.num_timesteps % self.log_freq != 0:
            return True

        env = make_env()()

        fig_traj = plot_trajectory(self.model, env)
        fig_ctrl = plot_control_field(self.model)

        wandb.log(
            {
                "plots/trajectory": wandb.Image(fig_traj),
                "plots/control_field": wandb.Image(fig_ctrl),
            },
            step=self.num_timesteps,
        )

        plt.close(fig_traj)
        plt.close(fig_ctrl)
        env.close()
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


def _make_objective(algo: str, steps_per_trial: int, n_eval_episodes: int, n_envs: int):
    """Return an Optuna objective function (closure over budget + algo)."""
    algo_cls = ALGO_CLS[algo]
    sampler_fn = SAMPLER[algo]

    def objective(trial: optuna.Trial) -> float:
        params = sampler_fn(trial)
        env = SubprocVecEnv([make_env() for _ in range(n_envs)])
        eval_env = make_env()()

        model = algo_cls("MlpPolicy", env, verbose=0, device="cpu", **params)
        model.learn(total_timesteps=steps_per_trial)

        mean_reward = _evaluate_policy(model, eval_env, n_episodes=n_eval_episodes)
        env.close()
        eval_env.close()
        return mean_reward

    return objective


# ---------------------------------------------------------------------------
# Param loading
# ---------------------------------------------------------------------------


def _load_best_td3_params(raw: dict) -> dict:
    """Reconstruct TD3-ready params from saved JSON."""
    n_actions = 1
    noise_type = raw.get("noise_type", "none")
    noise_std = raw.get("noise_std", 0.1)
    action_noise = None
    if noise_type == "normal":
        action_noise = NormalActionNoise(
            mean=np.zeros(n_actions),
            sigma=noise_std * np.ones(n_actions),
        )
    elif noise_type == "ornstein-uhlenbeck":
        action_noise = OrnsteinUhlenbeckActionNoise(
            mean=np.zeros(n_actions),
            sigma=noise_std * np.ones(n_actions),
        )

    return {
        "gamma": 1 - raw["one_minus_gamma"],
        "learning_rate": raw["learning_rate"],
        "batch_size": 2 ** raw["batch_size_pow"],
        "train_freq": raw["train_freq"],
        "tau": raw["tau"],
        "action_noise": action_noise,
        "policy_kwargs": {"net_arch": TD3_NET_ARCH_MAP[raw["net_arch"]]},
    }


def _load_best_ppo_params(raw: dict) -> dict:
    """Reconstruct PPO-ready params from saved JSON."""
    n_steps = 2 ** raw["n_steps_pow"]
    batch_size = 2 ** raw["batch_size_pow"]
    if batch_size > n_steps:
        batch_size = n_steps

    return {
        "n_steps": n_steps,
        "batch_size": batch_size,
        "gamma": 1 - raw["one_minus_gamma"],
        "gae_lambda": 1 - raw["one_minus_gae_lambda"],
        "learning_rate": raw["learning_rate"],
        "ent_coef": raw["ent_coef"],
        "clip_range": raw["clip_range"],
        "n_epochs": raw["n_epochs"],
        "max_grad_norm": raw["max_grad_norm"],
        "policy_kwargs": {
            "net_arch": PPO_NET_ARCH_MAP[raw["net_arch"]],
            "activation_fn": ACTIVATION_FN_MAP[raw["activation_fn"]],
        },
    }


_PARAM_LOADER = {"td3": _load_best_td3_params, "ppo": _load_best_ppo_params}


def _load_best_params(algo: str) -> dict:
    """Load best params from JSON for the given algorithm."""
    path = _best_params_path(algo)
    if not path.exists():
        print(f"No {path.name} found, using defaults.")
        return {}

    with open(path) as f:
        raw = json.load(f)

    return _PARAM_LOADER[algo](raw)


# ---------------------------------------------------------------------------
# CLI entry points
# ---------------------------------------------------------------------------


def search(
    algo: str = "ppo",
    n_trials: int = 50,
    steps_per_trial: int = 20000,
    n_eval_episodes: int = 5,
    n_jobs: int = 1,
    n_envs: int = 4,
):
    """Run Optuna hyperparameter search.

    Args:
        algo: Algorithm to search ("ppo" or "td3").
        n_trials: Number of Optuna trials.
        steps_per_trial: Training steps per trial.
        n_eval_episodes: Episodes for evaluation after each trial.
        n_jobs: Number of parallel workers (-1 for all CPUs).
        n_envs: Number of parallel environments per trial.
    """
    algo = algo.lower()
    assert algo in ALGO_CLS, f"Unknown algo '{algo}', choose from {list(ALGO_CLS)}"

    study = optuna.create_study(direction="maximize")
    objective = _make_objective(algo, steps_per_trial, n_eval_episodes, n_envs)
    study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs)

    print(f"\nBest trial reward: {study.best_trial.value:.4f}")
    print(f"Best params: {study.best_params}")

    path = _best_params_path(algo)
    serializable = {k: v for k, v in study.best_params.items()}
    with open(path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"Saved best params to {path}")


def train(algo: str = "ppo", total_timesteps: int = 200000, log_freq: int = 1000, n_envs: int = 4):
    """Train on ABGym with W&B logging using best Optuna params.

    Args:
        algo: Algorithm to train ("ppo" or "td3").
        total_timesteps: Total environment steps.
        log_freq: Steps between histogram logs.
        n_envs: Number of parallel environments.
    """
    algo = algo.lower()
    assert algo in ALGO_CLS, f"Unknown algo '{algo}', choose from {list(ALGO_CLS)}"

    params = _load_best_params(algo)

    run = wandb.init(
        project=f"ab-gym-{algo}",
        config={"algo": algo, **params},
        sync_tensorboard=True,
    )

    env = SubprocVecEnv([make_env() for _ in range(n_envs)])
    eval_env = make_env()()

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    algo_cls = ALGO_CLS[algo]
    model = algo_cls("MlpPolicy", env, verbose=1, device="cpu", tensorboard_log=f"runs/{run.id}", **params)

    callbacks = [
        WandbCallback(model_save_path=str(MODEL_DIR), verbose=1),
        WandbHistogramCallback(log_freq=log_freq),
        WandbPlotCallback(log_freq=log_freq * 10),
        EvalCallback(
            eval_env,
            best_model_save_path=str(MODEL_DIR),
            eval_freq=5000,
            n_eval_episodes=5,
            deterministic=True,
        ),
    ]

    model.learn(total_timesteps=total_timesteps, callback=callbacks)

    final_path = MODEL_DIR / f"{algo}_ab_final"
    model.save(str(final_path))
    print(f"Final model saved to {final_path}")

    env.close()
    eval_env.close()
    run.finish()


def all(
    algo: str = "ppo",
    n_trials: int = 50,
    steps_per_trial: int = 20000,
    total_timesteps: int = 200000,
    n_jobs: int = 1,
    n_envs: int = 4,
):
    """Run search then train."""
    search(algo=algo, n_trials=n_trials, steps_per_trial=steps_per_trial, n_jobs=n_jobs, n_envs=n_envs)
    train(algo=algo, total_timesteps=total_timesteps, n_envs=n_envs)


if __name__ == "__main__":
    fire.Fire({"search": search, "train": train, "all": all})
