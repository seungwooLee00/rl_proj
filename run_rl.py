import os
import json
import gc
import random
from typing import Dict, Any, List, Tuple

import torch
import numpy as np

from common.encoder import Encoder
from common.subnet_config import SubnetConfig
from common.predictor import MLPPredictor
from rl.rl_config import SubnetEnvConfig, PPOConfig
from rl.ppo_agent import PPOAgent


LOG_DIR_ROOT = "rl_results"
SEED = 120250336


def set_seed(seed: int) -> None:
    """
    Set random seeds for reproducibility across Python, NumPy and PyTorch.

    Parameters
    ----------
    seed : int
        Seed value to be used for all random number generators.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_predictors(device: torch.device) -> Tuple[
    MLPPredictor,
    MLPPredictor,
    MLPPredictor,
    MLPPredictor,
]:
    """
    Build and load all performance predictors onto the given device.

    Parameters
    ----------
    device : torch.device
        Device on which predictors will be placed.

    Returns
    -------
    Tuple[MLPPredictor, MLPPredictor, MLPPredictor, MLPPredictor]
        Tuple of (accuracy, latency, flash, sram) predictors.
    """
    acc_predictor = MLPPredictor(use_sigmoid=True)
    acc_predictor.load_state_dict(
        torch.load("assets/predictors/accuracy_predictor.pth", map_location=device)
    )
    acc_predictor.to(device)
    acc_predictor.eval()

    lat_predictor = MLPPredictor()
    lat_predictor.load_state_dict(
        torch.load("assets/predictors/latency_predictor.pth", map_location=device)
    )
    lat_predictor.to(device)
    lat_predictor.eval()

    flash_predictor = MLPPredictor()
    flash_predictor.load_state_dict(
        torch.load("assets/predictors/flash_predictor.pth", map_location=device)
    )
    flash_predictor.to(device)
    flash_predictor.eval()

    sram_predictor = MLPPredictor()
    sram_predictor.load_state_dict(
        torch.load("assets/predictors/sram_predictor.pth", map_location=device)
    )
    sram_predictor.to(device)
    sram_predictor.eval()

    return acc_predictor, lat_predictor, flash_predictor, sram_predictor


def summarize_evaluation(
    eval_json_path: str,
    env_config: SubnetEnvConfig,
) -> Dict[str, Any]:
    """
    Summarize evaluation JSON into simple metrics for grid search comparison.

    Metrics include:
        - number of episodes
        - mean total reward
        - constraint violation rates (flash, sram, latency, any)
        - accuracy statistics for constraint-satisfying episodes

    Parameters
    ----------
    eval_json_path : str
        Path to the evaluation JSON file.
    env_config : SubnetEnvConfig
        Environment configuration used during evaluation.

    Returns
    -------
    Dict[str, Any]
        Dictionary of aggregated metrics.
    """
    if not os.path.exists(eval_json_path):
        raise FileNotFoundError(f"Evaluation JSON not found: {eval_json_path}")

    with open(eval_json_path, "r") as f:
        data = json.load(f)

    episodes: List[Dict[str, Any]] = data.get("episodes", [])
    if len(episodes) == 0:
        return {
            "n_episodes": 0,
            "mean_total_reward": 0.0,
            "flash_violation_rate": 1.0,
            "sram_violation_rate": 1.0,
            "latency_violation_rate": 1.0,
            "any_violation_rate": 1.0,
            "valid_mean_accuracy": 0.0,
            "valid_max_accuracy": 0.0,
        }

    n_episodes = len(episodes)

    total_rewards: List[float] = []
    flash_violations = 0
    sram_violations = 0
    latency_violations = 0
    any_violations = 0
    valid_accuracies: List[float] = []

    target_flash = float(env_config.target_flash)
    target_sram = float(env_config.target_sram)

    for episode in episodes:
        total_reward = float(episode.get("total_reward", 0.0))
        total_rewards.append(total_reward)

        acc = episode.get("accuracy", None)
        lat = episode.get("latency", None)
        flash = episode.get("flash", None)
        sram = episode.get("sram", None)
        target_latency = float(episode.get("target_latency", 0.0))

        flash_violation = False
        sram_violation = False
        latency_violation = False

        if flash is not None:
            flash_violation = flash > target_flash
        if sram is not None:
            sram_violation = sram > target_sram
        if lat is not None:
            latency_violation = lat > target_latency

        any_violation = flash_violation or sram_violation or latency_violation

        if flash_violation:
            flash_violations += 1
        if sram_violation:
            sram_violations += 1
        if latency_violation:
            latency_violations += 1
        if any_violation:
            any_violations += 1

        if not any_violation and acc is not None:
            valid_accuracies.append(float(acc))

    mean_total_reward = float(np.mean(total_rewards)) if total_rewards else 0.0

    flash_violation_rate = flash_violations / n_episodes
    sram_violation_rate = sram_violations / n_episodes
    latency_violation_rate = latency_violations / n_episodes
    any_violation_rate = any_violations / n_episodes

    if len(valid_accuracies) > 0:
        valid_mean_accuracy = float(np.mean(valid_accuracies))
        valid_max_accuracy = float(np.max(valid_accuracies))
    else:
        valid_mean_accuracy = 0.0
        valid_max_accuracy = 0.0

    summary = {
        "n_episodes": n_episodes,
        "mean_total_reward": mean_total_reward,
        "flash_violation_rate": flash_violation_rate,
        "sram_violation_rate": sram_violation_rate,
        "latency_violation_rate": latency_violation_rate,
        "any_violation_rate": any_violation_rate,
        "valid_mean_accuracy": valid_mean_accuracy,
        "valid_max_accuracy": valid_max_accuracy,
    }
    return summary


def stage1_grid_search_ppo(
    device: torch.device,
    encoder: Encoder,
    subnet_config: SubnetConfig,
    predictors: Tuple[
        MLPPredictor,
        MLPPredictor,
        MLPPredictor,
        MLPPredictor,
    ],
) -> Dict[str, Any]:
    """
    Stage 1: grid search over PPO hyperparameters with fixed reward parameters.

    Parameters
    ----------
    device : torch.device
        Device used for training.
    encoder : Encoder
        Architecture encoder.
    subnet_config : SubnetConfig
        Subnet configuration defining search space.
    predictors : Tuple[MLPPredictor, MLPPredictor, MLPPredictor, MLPPredictor]
        Tuple of (accuracy, latency, flash, sram) predictors.

    Returns
    -------
    Dict[str, Any]
        Best PPO hyperparameter configuration and its summary.
    """
    (
        acc_predictor,
        lat_predictor,
        flash_predictor,
        sram_predictor,
    ) = predictors

    base_env_config = SubnetEnvConfig()
    base_env_config.hard_violation = 4.0
    base_env_config.soft_violation = 0.8
    base_env_config.accuracy_scale = 5.0

    learning_rates = [3e-4, 1e-4]
    ent_coefs = [0.01, 0.03]
    clip_ranges = [0.1, 0.2]

    total_timesteps_stage1 = 50_000
    eval_episodes_stage1 = 5

    results_stage1: List[Dict[str, Any]] = []

    os.makedirs(LOG_DIR_ROOT, exist_ok=True)

    for lr in learning_rates:
        for ent in ent_coefs:
            for clip in clip_ranges:
                env_config = SubnetEnvConfig()
                env_config.hard_violation = base_env_config.hard_violation
                env_config.soft_violation = base_env_config.soft_violation
                env_config.accuracy_scale = base_env_config.accuracy_scale

                ppo_config = PPOConfig(
                    policy="MlpPolicy",
                    learning_rate=lr,
                    n_steps=2048,
                    n_envs=8,
                    batch_size=256,
                    n_epochs=4,
                    gamma=0.99,
                    gae_lambda=0.95,
                    clip_range=clip,
                    ent_coef=ent,
                )

                log_dir = os.path.join(
                    LOG_DIR_ROOT,
                    f"stage1_lr_{lr}_ent_{ent}_clip_{clip}",
                )

                set_seed(SEED)

                agent = PPOAgent(
                    encoder=encoder,
                    acc_predictor=acc_predictor,
                    lat_predictor=lat_predictor,
                    flash_predictor=flash_predictor,
                    sram_predictor=sram_predictor,
                    subnet_config=subnet_config,
                    env_config=env_config,
                    ppo_config=ppo_config,
                    device=device,
                    log_dir=log_dir,
                    seed=SEED,
                )

                agent.train(total_timesteps=total_timesteps_stage1)

                eval_filename = "evaluation_stage1.json"
                agent.evaluate_agent(
                    n_episodes=eval_episodes_stage1,
                    filename=eval_filename,
                )

                eval_path = os.path.join(log_dir, eval_filename)
                summary = summarize_evaluation(
                    eval_json_path=eval_path,
                    env_config=env_config,
                )

                result_entry = {
                    "learning_rate": lr,
                    "ent_coef": ent,
                    "clip_range": clip,
                    "summary": summary,
                    "log_dir": log_dir,
                }
                results_stage1.append(result_entry)

                agent.save_agent(os.path.join(log_dir, "final_model"))

                del agent
                torch.cuda.empty_cache()
                gc.collect()

    summary_path = os.path.join(LOG_DIR_ROOT, "grid_summary_stage1.json")
    with open(summary_path, "w") as f:
        json.dump(results_stage1, f, indent=4)

    def score_fn(entry: Dict[str, Any]) -> float:
        summary = entry["summary"]
        return summary["valid_mean_accuracy"] * (1.0 - summary["any_violation_rate"])

    best_entry = max(results_stage1, key=score_fn)

    best_ppo_info = {
        "learning_rate": best_entry["learning_rate"],
        "ent_coef": best_entry["ent_coef"],
        "clip_range": best_entry["clip_range"],
        "summary": best_entry["summary"],
    }
    return best_ppo_info


def stage2_grid_search_reward(
    device: torch.device,
    encoder: Encoder,
    subnet_config: SubnetConfig,
    predictors: Tuple[
        MLPPredictor,
        MLPPredictor,
        MLPPredictor,
        MLPPredictor,
    ],
    best_ppo_info: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Stage 2: grid search over reward parameters with PPO fixed.

    Search over (hard_violation, soft_violation, accuracy_scale) while
    using the best PPO hyperparameters found in stage 1.

    Parameters
    ----------
    device : torch.device
        Device used for training.
    encoder : Encoder
        Architecture encoder.
    subnet_config : SubnetConfig
        Subnet configuration defining search space.
    predictors : Tuple[MLPPredictor, MLPPredictor, MLPPredictor, MLPPredictor]
        Tuple of (accuracy, latency, flash, sram) predictors.
    best_ppo_info : Dict[str, Any]
        Best PPO configuration from stage 1.

    Returns
    -------
    Dict[str, Any]
        Best reward configuration and its summary.
    """
    (
        acc_predictor,
        lat_predictor,
        flash_predictor,
        sram_predictor,
    ) = predictors

    best_lr = best_ppo_info["learning_rate"]
    best_ent = best_ppo_info["ent_coef"]
    best_clip = best_ppo_info["clip_range"]

    hard_violations = [3.0, 5.0]
    soft_violations = [0.5, 1.0]
    accuracy_scales = [4.0, 6.0]

    total_timesteps_stage2 = 100_000
    eval_episodes_stage2 = 5

    results_stage2: List[Dict[str, Any]] = []

    os.makedirs(LOG_DIR_ROOT, exist_ok=True)

    for hard_violation in hard_violations:
        for soft_violation in soft_violations:
            for accuracy_scale in accuracy_scales:
                env_config = SubnetEnvConfig()
                env_config.hard_violation = hard_violation
                env_config.soft_violation = soft_violation
                env_config.accuracy_scale = accuracy_scale

                ppo_config = PPOConfig(
                    policy="MlpPolicy",
                    learning_rate=best_lr,
                    n_steps=2048,
                    n_envs=8,
                    batch_size=256,
                    n_epochs=4,
                    gamma=0.99,
                    gae_lambda=0.95,
                    clip_range=best_clip,
                    ent_coef=best_ent,
                )

                log_dir = os.path.join(
                    LOG_DIR_ROOT,
                    f"stage2_hv_{hard_violation}_sv_{soft_violation}_asc_{accuracy_scale}",
                )

                set_seed(SEED)

                agent = PPOAgent(
                    encoder=encoder,
                    acc_predictor=acc_predictor,
                    lat_predictor=lat_predictor,
                    flash_predictor=flash_predictor,
                    sram_predictor=sram_predictor,
                    subnet_config=subnet_config,
                    env_config=env_config,
                    ppo_config=ppo_config,
                    device=device,
                    log_dir=log_dir,
                    seed=SEED,
                )

                agent.train(total_timesteps=total_timesteps_stage2)

                eval_filename = "evaluation_stage2.json"
                agent.evaluate_agent(
                    n_episodes=eval_episodes_stage2,
                    filename=eval_filename,
                )

                eval_path = os.path.join(log_dir, eval_filename)
                summary = summarize_evaluation(
                    eval_json_path=eval_path,
                    env_config=env_config,
                )

                result_entry = {
                    "hard_violation": hard_violation,
                    "soft_violation": soft_violation,
                    "accuracy_scale": accuracy_scale,
                    "summary": summary,
                    "log_dir": log_dir,
                }
                results_stage2.append(result_entry)

                agent.save_agent(os.path.join(log_dir, "final_model"))

                del agent
                torch.cuda.empty_cache()
                gc.collect()

    summary_path = os.path.join(LOG_DIR_ROOT, "grid_summary_stage2.json")
    with open(summary_path, "w") as f:
        json.dump(results_stage2, f, indent=4)

    def score_fn(entry: Dict[str, Any]) -> float:
        summary = entry["summary"]
        return summary["valid_mean_accuracy"] * (1.0 - summary["any_violation_rate"])

    best_entry = max(results_stage2, key=score_fn)

    best_reward_info = {
        "hard_violation": best_entry["hard_violation"],
        "soft_violation": best_entry["soft_violation"],
        "accuracy_scale": best_entry["accuracy_scale"],
        "summary": best_entry["summary"],
    }
    return best_reward_info


def main() -> None:
    """
    Run two-stage grid search for PPO and reward parameters and save results.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    set_seed(SEED)

    predictors = build_predictors(device)
    subnet_config = SubnetConfig()
    encoder = Encoder(subnet_config)

    best_ppo_info = stage1_grid_search_ppo(
        device=device,
        encoder=encoder,
        subnet_config=subnet_config,
        predictors=predictors,
    )

    best_reward_info = stage2_grid_search_reward(
        device=device,
        encoder=encoder,
        subnet_config=subnet_config,
        predictors=predictors,
        best_ppo_info=best_ppo_info,
    )

    final_summary = {
        "best_ppo": best_ppo_info,
        "best_reward": best_reward_info,
    }
    final_summary_path = os.path.join(LOG_DIR_ROOT, "best_configs.json")
    with open(final_summary_path, "w") as f:
        json.dump(final_summary, f, indent=4)


if __name__ == "__main__":
    main()
