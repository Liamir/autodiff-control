"""RL training (PPO) on the IFFL2Vars gym.

Usage:
    python scripts/RL_IFFL2Vars_gym.py search --n_trials=50 --steps_per_trial=20000
    python scripts/RL_IFFL2Vars_gym.py train  --total_timesteps=200000
    python scripts/RL_IFFL2Vars_gym.py all
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
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.vec_env import SubprocVecEnv
from wandb.integration.sb3 import WandbCallback

from rpasim.gyms import IFFL2VarsGym

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
MODEL_DIR = PROJECT_ROOT / "models"
PARAM_DIR = PROJECT_ROOT / "model_params"

CONTROLLED_PARAMS = ["alpha", "delta", "beta", "gamma"]
N_ACTIONS = len(CONTROLLED_PARAMS)

ACTIVATION_FN_MAP = {
    "tanh": torch.nn.Tanh,
    "relu": torch.nn.ReLU,
}


def _best_params_path() -> Path:
    return PARAM_DIR / "best_ppo_iffl2vars_params.json"


def _default_model_path() -> Path:
    return MODEL_DIR / "ppo_iffl2vars_final"


# ---------------------------------------------------------------------------
# Environment factory
# ---------------------------------------------------------------------------


def make_env():
    """Create an IFFL2VarsGym instance with default settings."""

    def _init():
        return IFFL2VarsGym(
            reward_fn=lambda state, time: state[1] ** 2,
            initial_state=torch.tensor([0.0, 0.0]),
            time_horizon=10.0,
            base_params={
                "alpha": 1.0,
                "delta": 10,
                "beta": 1.0,
                "gamma": 100.0,
            },
            dt=0.1,
            n_reward_steps=300,
            controlled_params=CONTROLLED_PARAMS,
            action_low=0.5,
            action_high=2.0,
            ode_method="rk4",
        )

    return _init


# ---------------------------------------------------------------------------
# Hyperparameter sampler
# ---------------------------------------------------------------------------

PPO_NET_ARCH_MAP = {
    "tiny": dict(pi=[64], vf=[64]),
    "small": dict(pi=[64, 64], vf=[64, 64]),
    "medium": dict(pi=[128, 128], vf=[128, 128]),
}


def sample_ppo_params(trial: optuna.Trial) -> dict:
    """Sample PPO hyperparameters for an Optuna trial."""
    batch_size_pow = trial.suggest_int("batch_size_pow", 2, 10)
    one_minus_gamma = trial.suggest_float("one_minus_gamma", 0.0001, 0.03, log=True)
    one_minus_gae_lambda = trial.suggest_float("one_minus_gae_lambda", 0.0001, 0.1, log=True)
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 0.002, log=True)
    ent_coef = trial.suggest_float("ent_coef", 0.00000001, 0.1, log=True)
    clip_range = trial.suggest_categorical("clip_range", [0.1, 0.2, 0.3, 0.4])
    n_epochs = trial.suggest_categorical("n_epochs", [1, 5, 10, 20])
    max_grad_norm = trial.suggest_float("max_grad_norm", 0.3, 2.0)
    net_arch = trial.suggest_categorical("net_arch", ["tiny", "small", "medium"])
    activation_fn = trial.suggest_categorical("activation_fn", ["tanh", "relu"])

    n_steps = 20  # time_horizon / dt = 2.0 / 0.1
    batch_size = 2**batch_size_pow
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
                "histograms/obs_x": wandb.Histogram(obs_arr[:, 0]),
                "histograms/obs_y": wandb.Histogram(obs_arr[:, 1]),
                "histograms/rewards": wandb.Histogram(rew_arr),
            }

            for name, param in self.model.policy.named_parameters():
                if param.grad is not None:
                    log_dict[f"gradients/{name}_norm"] = param.grad.norm().item()

            wandb.log(log_dict, step=self.num_timesteps)

        return True


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------


def plot_trajectory(model, env):
    """Run one episode and plot x(t), y(t), and all 4 control actions."""
    obs, _ = env.reset()
    observations = [obs.copy()]
    actions_list = []
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, _ = env.step(action)
        observations.append(obs.copy())
        actions_list.append(action.copy())
        done = terminated or truncated

    traj = np.array(observations)
    acts = np.array(actions_list)
    t_states = np.arange(len(traj)) * env.dt
    t_actions = np.arange(len(acts)) * env.dt

    fig, axes = plt.subplots(6, 1, figsize=(6, 8), sharex=True)

    axes[0].plot(t_states, traj[:, 0])
    axes[0].set_ylabel("x")
    sns.despine(ax=axes[0])

    axes[1].plot(t_states, traj[:, 1])
    axes[1].set_ylabel("y")
    sns.despine(ax=axes[1])

    for i, name in enumerate(CONTROLLED_PARAMS):
        ax = axes[2 + i]
        ax.step(t_actions, acts[:, i], where="post")
        ax.set_ylabel(f"u_{name}")
        ax.set_ylim(env.action_low, env.action_high)
        sns.despine(ax=ax)

    axes[-1].set_xlabel("time")
    fig.tight_layout()
    return fig


def plot_control_field(model, env, x_range=(0.0, 2.0), y_range=(0.0, 2.0), n_grid=50):
    """Plot each control action as a heatmap over (x, y) state space."""
    x_vals = np.linspace(*x_range, n_grid)
    y_vals = np.linspace(*y_range, n_grid)
    xx, yy = np.meshgrid(x_vals, y_vals)
    states = np.stack([xx.ravel(), yy.ravel()], axis=1).astype(np.float32)

    actions = []
    for s in states:
        action, _ = model.predict(s, deterministic=True)
        actions.append(action.copy())
    actions = np.array(actions).reshape(n_grid, n_grid, N_ACTIONS)

    fig, axes = plt.subplots(2, 2, figsize=(6, 5))
    for i, (ax, name) in enumerate(zip(axes.ravel(), CONTROLLED_PARAMS)):
        im = ax.pcolormesh(
            xx, yy, actions[:, :, i], shading="auto", cmap="viridis", vmin=env.action_low, vmax=env.action_high
        )
        fig.colorbar(im, ax=ax, label=f"u_{name}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(name)
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
        fig_ctrl = plot_control_field(self.model, env)

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


def _make_objective(steps_per_trial: int, n_eval_episodes: int, n_envs: int):
    """Return an Optuna objective function."""

    def objective(trial: optuna.Trial) -> float:
        params = sample_ppo_params(trial)
        env = SubprocVecEnv([make_env() for _ in range(n_envs)])
        eval_env = make_env()()

        model = PPO("MlpPolicy", env, verbose=0, device="cpu", **params)
        model.learn(total_timesteps=steps_per_trial)

        mean_reward = _evaluate_policy(model, eval_env, n_episodes=n_eval_episodes)
        env.close()
        eval_env.close()
        return mean_reward

    return objective


# ---------------------------------------------------------------------------
# Param loading
# ---------------------------------------------------------------------------


def _load_best_ppo_params(raw: dict) -> dict:
    """Reconstruct PPO-ready params from saved JSON."""
    n_steps = 20  # time_horizon / dt
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


def _load_best_params() -> dict:
    """Load best params from JSON."""
    path = _best_params_path()
    if not path.exists():
        print(f"No {path.name} found, using defaults.")
        return {}

    with open(path) as f:
        raw = json.load(f)

    return _load_best_ppo_params(raw)


# ---------------------------------------------------------------------------
# CLI entry points
# ---------------------------------------------------------------------------


def search(
    n_trials: int = 50,
    steps_per_trial: int = 20000,
    n_eval_episodes: int = 5,
    n_jobs: int = 1,
    n_envs: int = 4,
):
    """Run Optuna hyperparameter search.

    Args:
        n_trials: Number of Optuna trials.
        steps_per_trial: Training steps per trial.
        n_eval_episodes: Episodes for evaluation after each trial.
        n_jobs: Number of parallel workers (-1 for all CPUs).
        n_envs: Number of parallel environments per trial.
    """
    study = optuna.create_study(direction="maximize")
    objective = _make_objective(steps_per_trial, n_eval_episodes, n_envs)
    study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs)

    print(f"\nBest trial reward: {study.best_trial.value:.4f}")
    print(f"Best params: {study.best_params}")

    path = _best_params_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    serializable = {k: v for k, v in study.best_params.items()}
    with open(path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"Saved best params to {path}")


def train(total_timesteps: int = 200000, log_freq: int = 1000, n_envs: int = 4):
    """Train on IFFL2VarsGym with W&B logging using best Optuna params.

    Args:
        total_timesteps: Total environment steps.
        log_freq: Steps between histogram logs.
        n_envs: Number of parallel environments.
    """
    params = _load_best_params()

    run = wandb.init(
        project="iffl2vars-gym-ppo",
        config=params,
        sync_tensorboard=True,
    )

    env = SubprocVecEnv([make_env() for _ in range(n_envs)])
    eval_env = make_env()()

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    model = PPO("MlpPolicy", env, verbose=1, device="cpu", tensorboard_log=f"runs/{run.id}", **params)

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

    final_path = _default_model_path()
    model.save(str(final_path))
    print(f"Final model saved to {final_path}")

    env.close()
    eval_env.close()
    run.finish()


def all(
    n_trials: int = 50,
    steps_per_trial: int = 20000,
    total_timesteps: int = 200000,
    n_jobs: int = 1,
    n_envs: int = 4,
):
    """Run search then train."""
    search(n_trials=n_trials, steps_per_trial=steps_per_trial, n_jobs=n_jobs, n_envs=n_envs)
    train(total_timesteps=total_timesteps, n_envs=n_envs)


def plot(model_path: str = None):
    """Plot trajectory from a saved model.

    Args:
        model_path: Path to saved model. Defaults to models/ppo_iffl2vars_final.
    """
    if model_path is None:
        model_path = str(_default_model_path())
    model = PPO.load(model_path, device="cpu")

    eval_env = make_env()()
    fig = plot_trajectory(model, eval_env)
    eval_env.close()

    out_dir = PROJECT_ROOT / "plots" / "IFFL2Vars" / "ppo"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "test_ppo_trajectory.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    fire.Fire({"search": search, "train": train, "all": all, "plot": plot})
