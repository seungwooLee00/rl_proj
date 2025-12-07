from typing import Dict, Any, List, Tuple
import copy
import random

import numpy as np
import torch
import gymnasium as gym
from gymnasium import spaces

from common.encoder import Encoder
from common.subnet_config import SubnetConfig
from common.predictor import MLPPredictor
from rl.rl_config import SubnetEnvConfig


class SubnetEnv(gym.Env):
    """
    Gymnasium environment for subnet architecture search with learned predictors.

    At each step:
        - The agent selects a discrete action that changes one part of the subnet.
        - The environment updates the current architecture according to a decision schedule.
        - A scalar score is computed from predicted accuracy and constraints (latency, flash, sram).
        - The reward is the improvement of this score compared to the previous step.

    Episode ends after all decisions in the schedule have been made.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        encoder: Encoder,
        acc_predictor: MLPPredictor,
        lat_predictor: MLPPredictor,
        flash_predictor: MLPPredictor,
        sram_predictor: MLPPredictor,
        subnet_config: SubnetConfig,
        env_config: SubnetEnvConfig,
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__()

        self.encoder = encoder
        self.acc_predictor = acc_predictor
        self.lat_predictor = lat_predictor
        self.flash_predictor = flash_predictor
        self.sram_predictor = sram_predictor

        self.subnet_config = subnet_config
        self.env_config = env_config
        self.device = device

        self.current_arch: Dict[str, List] = {}
        self.current_step: int = 0

        self.current_target_latency: float = 1.0
        self.current_target_flash: float = 1.0
        self.current_target_sram: float = 1.0

        self.prev_score: float = 0.0

        # Decision schedule defines which key and index is modified at each step.
        self.decision_schedule = self.build_decision_schedule()
        self.n_steps = len(self.decision_schedule)

        # Discrete action index is interpreted as an index into candidate lists.
        self.max_action_size = max(
            [
                len(self.subnet_config.ks_list),
                len(self.subnet_config.e_list),
                len(self.subnet_config.d_list),
                len(self.subnet_config.aux_a_list),
                len(self.subnet_config.aux_t_list),
                len(self.subnet_config.aux_k_list),
                len(self.subnet_config.aux_e_list),
            ]
        )
        self.action_space = spaces.Discrete(self.max_action_size)

        # Build a dummy observation to define observation_space.
        dummy_arch = self.get_largest_subnet()
        dummy_accuracy, dummy_latency, dummy_flash, dummy_sram = self.evaluate(dummy_arch)
        dummy_obs = self.get_observation(
            arch=dummy_arch,
            step=0,
            accuracy=dummy_accuracy,
            latency=dummy_latency,
            flash=dummy_flash,
            sram=dummy_sram,
        )

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=dummy_obs.shape,
            dtype=np.float32,
        )

    # -------------------------------------------------------------------------
    # Architecture and predictor utilities
    # -------------------------------------------------------------------------

    def encode(self, arch: Dict[str, List]) -> np.ndarray:
        """
        Encode a subnet architecture into a flat float32 vector using the encoder.
        """
        vector = self.encoder.encode([arch])
        if isinstance(vector, torch.Tensor):
            vector = vector.detach().cpu().numpy()
        return vector.reshape(-1).astype(np.float32)

    def evaluate(self, arch: Dict[str, List]) -> Tuple[float, float, float, float]:
        """
        Predict accuracy, latency, flash, sram for a given architecture.
        """
        vector = self.encoder.encode([arch])

        with torch.no_grad():
            acc_tensor = self.acc_predictor.run(vector).reshape(-1)[0]
            lat_tensor = self.lat_predictor.run(vector).reshape(-1)[0]
            flash_tensor = self.flash_predictor.run(vector).reshape(-1)[0]
            sram_tensor = self.sram_predictor.run(vector).reshape(-1)[0]

        accuracy = float(acc_tensor.item())
        latency = float(lat_tensor.item())
        flash = float(flash_tensor.item())
        sram = float(sram_tensor.item())

        return accuracy, latency, flash, sram

    def get_largest_subnet(self) -> Dict[str, List]:
        """
        Build the largest subnet given the subnet configuration.
        """
        subnet_config = self.subnet_config

        arch = {
            "ks": [subnet_config.ks_list[-1] for _ in range(subnet_config.n_blocks)],
            "e": [subnet_config.e_list[-1] for _ in range(subnet_config.n_blocks)],
            "d": [subnet_config.d_list[-1] for _ in range(subnet_config.n_layers)],
            "aux_a": [subnet_config.aux_a_list[-1] for _ in range(subnet_config.n_layers)],
            "aux_t": [subnet_config.aux_t_list[-1] for _ in range(subnet_config.n_layers)],
            "aux_k": [subnet_config.aux_k_list[-1] for _ in range(subnet_config.n_layers)],
            "aux_e": [subnet_config.aux_e_list[-1] for _ in range(subnet_config.n_layers)],
        }
        return arch

    def normalize(
        self,
        lat: float,
        flash: float,
        sram: float,
    ) -> Tuple[float, float, float]:
        """
        Normalize latency, flash, sram into [0, 1] ranges based on env_config.
        """
        norm_lat = (lat - self.env_config.min_latency) / (
            self.env_config.max_latency - self.env_config.min_latency
        )
        norm_flash = (flash - self.env_config.min_flash) / (
            self.env_config.max_flash - self.env_config.min_flash
        )
        norm_sram = (sram - self.env_config.min_sram) / (
            self.env_config.max_sram - self.env_config.min_sram
        )
        return float(norm_lat), float(norm_flash), float(norm_sram)

    # -------------------------------------------------------------------------
    # Score function and observation
    # -------------------------------------------------------------------------

    def compute_score(
        self,
        accuracy: float,
        latency: float,
        flash: float,
        sram: float,
    ) -> float:
        """
        Compute a scalar score for the current subnet.

        Higher accuracy and smaller resource usage relative to targets
        should lead to a larger score.

        Score structure:
            - Accuracy term: accuracy ** accuracy_scale
            - Hard constraint term: flash, sram violations
            - Soft constraint term: latency violation
        """
        # Ratios to current targets
        flash_ratio = flash / self.current_target_flash
        sram_ratio = sram / self.current_target_sram
        lat_ratio = latency / self.current_target_latency

        # Constraint violations (only portions above the target)
        flash_over = max(0.0, flash_ratio - 1.0)
        sram_over = max(0.0, sram_ratio - 1.0)
        lat_over = max(0.0, lat_ratio - 1.0)

        # Hard constraints: flash and sram
        hard_violation = flash_over + sram_over
        hard_factor = float(
            np.exp(-self.env_config.hard_violation * hard_violation)
        )

        # Soft constraint: latency
        soft_factor = float(
            np.exp(-self.env_config.soft_violation * lat_over)
        )

        # Accuracy emphasis
        accuracy_clamped = max(accuracy, 0.0)
        acc_term = accuracy_clamped ** self.env_config.accuracy_scale

        score = acc_term * hard_factor * soft_factor
        return float(score)

    def get_observation(
        self,
        arch: Dict[str, List],
        step: int,
        accuracy: float,
        latency: float,
        flash: float,
        sram: float,
    ) -> np.ndarray:
        """
        Build observation vector from architecture and predicted metrics.

        Observation consists of:
            - Encoded architecture vector
            - Accuracy
            - Constraint slack ratios
            - Normalized progress in the decision schedule
        """
        arch_vector = self.encode(arch)

        # Step progress in [0, 1]
        step_fraction = (
            step / (self.n_steps - 1) if self.n_steps > 1 else 0.0
        )

        # Slack relative to current targets (positive means under the budget)
        lat_slack = 1.0 - (latency / self.current_target_latency)
        flash_slack = 1.0 - (flash / self.current_target_flash)
        sram_slack = 1.0 - (sram / self.current_target_sram)

        extra_features = np.array(
            [
                accuracy,
                lat_slack,
                flash_slack,
                sram_slack,
                step_fraction,
            ],
            dtype=np.float32,
        )

        observation = np.concatenate([arch_vector, extra_features], axis=0)
        return observation.astype(np.float32)

    # -------------------------------------------------------------------------
    # Decision schedule and action application
    # -------------------------------------------------------------------------

    def build_decision_schedule(self) -> List[Tuple[str, int]]:
        """
        Build the sequence of (key, index) pairs that define which part of
        the architecture is modified at each step.
        """
        subnet_config = self.subnet_config
        decision_schedule: List[Tuple[str, int]] = []

        # ks and e per block
        for index in range(subnet_config.n_blocks):
            decision_schedule.append(("ks", index))
        for index in range(subnet_config.n_blocks):
            decision_schedule.append(("e", index))

        # d per layer
        for index in range(subnet_config.n_layers):
            decision_schedule.append(("d", index))

        # auxiliary parts per layer
        for index in range(subnet_config.n_layers):
            decision_schedule.append(("aux_a", index))
        for index in range(subnet_config.n_layers):
            decision_schedule.append(("aux_t", index))
        for index in range(subnet_config.n_layers):
            decision_schedule.append(("aux_e", index))
        for index in range(subnet_config.n_layers):
            decision_schedule.append(("aux_k", index))

        return decision_schedule

    def apply_action(
        self,
        arch: Dict[str, List],
        step: int,
        action: int,
    ) -> Dict[str, List]:
        """
        Apply a discrete action to the current architecture for the given step.
        """
        new_arch = copy.deepcopy(arch)
        key, index = self.decision_schedule[step]

        candidates = getattr(self.subnet_config, f"{key}_list")

        # Clamp action index to valid range for this key.
        action_index = min(action, len(candidates) - 1)
        new_arch[key][index] = candidates[action_index]

        return new_arch

    # -------------------------------------------------------------------------
    # Gymnasium API: reset, step, render
    # -------------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: int | None = None,
        options: Dict[str, Any] | None = None,
    ):
        """
        Reset the environment for a new episode.

        Targets are sampled or taken from options.
        The initial architecture is the largest subnet.
        The previous score is set using the initial architecture.
        """
        super().reset(seed=seed)

        if options is not None:
            target_lat = options.get(
                "target_latency",
                random.choice(self.env_config.target_latency),
            )
        else:
            target_lat = random.choice(self.env_config.target_latency)

        self.current_target_latency = float(target_lat)
        self.current_target_flash = float(self.env_config.target_flash)
        self.current_target_sram = float(self.env_config.target_sram)

        self.current_arch = self.get_largest_subnet()
        self.current_step = 0

        # Evaluate initial architecture and set previous score
        accuracy, latency, flash, sram = self.evaluate(self.current_arch)
        self.prev_score = self.compute_score(
            accuracy=accuracy,
            latency=latency,
            flash=flash,
            sram=sram,
        )

        observation = self.get_observation(
            arch=self.current_arch,
            step=self.current_step,
            accuracy=accuracy,
            latency=latency,
            flash=flash,
            sram=sram,
        )

        info: Dict[str, Any] = {}
        return observation, info

    def step(self, action: int):
        """
        Apply the action, update the architecture, and return next state and reward.

        Reward is defined as the improvement of the scalar score between
        the previous step and the current step.
        """
        # Apply action and move to the next step
        self.current_arch = self.apply_action(
            arch=self.current_arch,
            step=self.current_step,
            action=action,
        )
        self.current_step += 1

        terminated = self.current_step == self.n_steps
        truncated = False

        # Predict metrics and compute new score
        accuracy, latency, flash, sram = self.evaluate(self.current_arch)
        current_score = self.compute_score(
            accuracy=accuracy,
            latency=latency,
            flash=flash,
            sram=sram,
        )

        # Local improvement reward
        reward = current_score - self.prev_score
        self.prev_score = current_score

        # Info dictionary is richer at the terminal step
        if terminated:
            info: Dict[str, Any] = {
                "accuracy": accuracy,
                "latency": latency,
                "flash": flash,
                "sram": sram,
                "score": current_score,
                "arch": self.current_arch,
            }
        else:
            info = {}

        # Build next observation
        observation = self.get_observation(
            arch=self.current_arch,
            step=min(self.current_step, self.n_steps - 1),
            accuracy=accuracy,
            latency=latency,
            flash=flash,
            sram=sram,
        )

        return observation, float(reward), terminated, truncated, info

    def render(self):
        """
        Rendering is not implemented for this environment.
        """
        return
