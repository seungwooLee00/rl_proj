import os
import json
from typing import Optional

import torch
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecMonitor

from rl.subnet_env import SubnetEnv
from common.encoder import Encoder
from common.subnet_config import SubnetConfig
from common.predictor import MLPPredictor
from rl.rl_config import SubnetEnvConfig, PPOConfig


class PPOAgent:
    """
    PPO agent wrapper for training and evaluating subnet architectures.
    """

    def __init__(
        self,
        encoder: Encoder,
        acc_predictor: MLPPredictor,
        lat_predictor: MLPPredictor,
        flash_predictor: MLPPredictor,
        sram_predictor: MLPPredictor,
        subnet_config: SubnetConfig,
        env_config: SubnetEnvConfig,
        ppo_config: PPOConfig,
        device: torch.device,
        log_dir: str,
        seed: int = 42,
    ):
        """
        Initialize PPOAgent with environment components and PPO configuration.
        """
        self.encoder = encoder
        self.acc_predictor = acc_predictor
        self.lat_predictor = lat_predictor
        self.flash_predictor = flash_predictor
        self.sram_predictor = sram_predictor

        self.subnet_config = subnet_config
        self.env_config = env_config
        self.ppo_config = ppo_config

        self.device = device
        self.log_dir = log_dir
        self.seed = seed

        self.model: Optional[PPO] = None

    def make_subnet_env(self):
        """
        Create a single SubnetEnv instance for PPO.
        Used by make_vec_env.
        """

        def _init():
            env = SubnetEnv(
                encoder=self.encoder,
                acc_predictor=self.acc_predictor,
                lat_predictor=self.lat_predictor,
                flash_predictor=self.flash_predictor,
                sram_predictor=self.sram_predictor,
                subnet_config=self.subnet_config,
                env_config=self.env_config,
                device=self.device,
            )
            return env

        return _init

    def train(self, total_timesteps: int = 100_000):
        """
        Train PPO on the subnet environment.
        """
        n_envs = self.ppo_config.n_envs

        # Vectorized environment with monitoring to CSV
        env = make_vec_env(
            self.make_subnet_env(),
            n_envs=n_envs,
            seed=self.seed,
        )
        env = VecMonitor(env, filename=os.path.join(self.log_dir, "monitor.csv"))

        # PPO model
        self.model = PPO(
            policy=self.ppo_config.policy,
            env=env,
            gamma=self.ppo_config.gamma,
            gae_lambda=self.ppo_config.gae_lambda,
            clip_range=self.ppo_config.clip_range,
            ent_coef=self.ppo_config.ent_coef,
            verbose=1,
            device=self.device,
            tensorboard_log=os.path.join(self.log_dir, "tb"),
            n_steps=self.ppo_config.n_steps // n_envs,
            batch_size=self.ppo_config.batch_size,
            n_epochs=self.ppo_config.n_epochs,
            learning_rate=self.ppo_config.learning_rate,
        )

        # Train
        self.model.learn(
            total_timesteps=total_timesteps,
            progress_bar=True,
        )

    def load_agent(self, model_path: str) -> PPO:
        """
        Load a trained PPO model from disk.
        """
        env_fn = self.make_subnet_env()
        env = env_fn()

        self.model = PPO.load(model_path, env=env, device=self.device)
        return self.model

    def save_agent(self, save_path: str):
        """
        Save the current PPO model to disk.
        """
        if self.model is None:
            raise ValueError("Model is not initialized. Train or load a model first.")

        self.model.save(save_path)

    def evaluate_agent(self, n_episodes: int = 5, filename: str = "evaluation_results.json"):
        """
        Evaluate the current agent for each target latency and save results as JSON.

        The JSON structure:
        {
            "n_episodes": int,
            "target_latencies": [...],
            "episodes": [
                {
                    "target_latency": float,
                    "episode": int,
                    "total_reward": float,
                    "accuracy": float or null,
                    "latency": float or null,
                    "flash": float or null,
                    "sram": float or null,
                    "arch": dict or null
                },
                ...
            ]
        }
        """
        if self.model is None:
            raise ValueError("Model is not initialized. Train or load a model first.")

        env = self.make_subnet_env()()

        results = {
            "n_episodes": int(n_episodes),
            "target_latencies": [float(t) for t in self.env_config.target_latency],
            "episodes": [],
        }

        for target_latency in self.env_config.target_latency:
            for episode in range(n_episodes):
                obs, info = env.reset(options={"target_latency": target_latency})
                done = False
                total_reward = 0.0

                # PPO expects batch dimension
                obs = np.array([obs], dtype=np.float32)

                while not done:
                    action, _ = self.model.predict(obs, deterministic=True)
                    obs, reward, done, _, info = env.step(action.item())
                    obs = np.array([obs], dtype=np.float32)
                    total_reward += float(reward)

                    if done:
                        acc = info.get("accuracy", None)
                        lat = info.get("latency", None)
                        flash = info.get("flash", None)
                        sram = info.get("sram", None)
                        arch = info.get("arch", None)

                        episode_result = {
                            "target_latency": float(target_latency),
                            "episode": int(episode + 1),
                            "total_reward": float(total_reward),
                            "accuracy": float(acc) if acc is not None else None,
                            "latency": float(lat) if lat is not None else None,
                            "flash": float(flash) if flash is not None else None,
                            "sram": float(sram) if sram is not None else None,
                            "arch": arch,
                        }
                        results["episodes"].append(episode_result)

        os.makedirs(self.log_dir, exist_ok=True)
        save_path = os.path.join(self.log_dir, filename)

        with open(save_path, "w") as f:
            json.dump(results, f, indent=4)
