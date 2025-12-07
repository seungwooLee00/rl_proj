from dataclasses import dataclass, field
from typing import List


@dataclass
class SubnetEnvConfig:
    """
    Environment configuration for subnet evaluation.

    Defines valid metric ranges, target resource usage,
    and penalty values used in reinforcement learning settings.

    Attributes
    ----------
    min_latency : float
        Minimum allowed latency.
    max_latency : float
        Maximum allowed latency.
    min_flash : float
        Minimum allowed flash usage.
    max_flash : float
        Maximum allowed flash usage.
    min_sram : float
        Minimum allowed SRAM usage.
    max_sram : float
        Maximum allowed SRAM usage.
    target_flash : float
        Desired flash usage target.
    target_sram : float
        Desired SRAM usage target.
    target_latency : List[float]
        Set of target latency points for reward shaping.
    accuracy_scale : float
        Scaling factor applied to accuracy reward.
    hard_violation : float
        Penalty applied for exceeding hard constraints.
    soft_violation : float
        Penalty applied for exceeding soft constraints.
    """
    min_latency: float = 0.6
    max_latency: float = 7.0
    min_flash: float = 1.0
    max_flash: float = 4.0
    min_sram: float = 0.784
    max_sram: float = 1.5

    target_flash: float = 1.8
    target_sram: float = 0.88
    target_latency: List[float] = field(default_factory=lambda: [1.0, 1.2, 1.4, 1.6, 1.8, 2.0])

    accuracy_scale: float = 4.0
    hard_violation: float = 3.0
    soft_violation: float = 1.5


@dataclass
class PPOConfig:
    """
    Hyperparameter configuration for PPO training.

    Attributes
    ----------
    policy : str
        Policy network type used by the PPO algorithm.
    learning_rate : float
        Optimizer learning rate.
    n_steps : int
        Number of steps per environment rollout.
    n_envs : int
        Number of parallel environments.
    batch_size : int
        Training batch size.
    n_epochs : int
        Number of optimization epochs per update.
    gamma : float
        Discount factor for future rewards.
    gae_lambda : float
        Lambda parameter for generalized advantage estimation.
    clip_range : float
        Clipping range for PPO policy updates.
    ent_coef : float
        Entropy coefficient for exploration regularization.
    """
    policy: str = "MlpPolicy"
    learning_rate: float = 3e-4
    n_steps: int = 2048
    n_envs: int = 8
    batch_size: int = 64
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.0
