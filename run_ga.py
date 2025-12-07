import os
import json
import random
from dataclasses import asdict
from typing import Dict, Any, List

import torch

from common.encoder import Encoder
from common.subnet_config import SubnetConfig
from common.predictor import MLPPredictor
from ga.ga_searcher import GASearcher
from ga.ga_config import GAConfig


def load_predictor(model_path: str, device: torch.device) -> MLPPredictor:
    """
    Load an MLP predictor model checkpoint.

    Parameters
    ----------
    model_path : str
        Path to the saved predictor checkpoint.
    device : torch.device
        Device on which the model will be loaded.

    Returns
    -------
    MLPPredictor
        Loaded predictor model in evaluation mode.
    """
    # Use sigmoid only for the accuracy predictor
    use_sigmoid = "accuracy_predictor" in model_path

    predictor = MLPPredictor(use_sigmoid=use_sigmoid)
    state_dict = torch.load(model_path, map_location=device)
    predictor.load_state_dict(state_dict)
    predictor.to(device)
    predictor.eval()
    return predictor


def set_seed(seed: int) -> None:
    """
    Set random seeds for reproducibility.

    Parameters
    ----------
    seed : int
        Seed value to be used for random number generators.
    """
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    """
    Run GA-based architecture search for multiple latency constraints.

    For each target maximum latency, multiple trials are executed and
    the best found individual per trial is recorded and saved to disk.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Shared subnet configuration and encoder
    subnet_config = SubnetConfig()
    encoder = Encoder(subnet_config)

    # Load performance predictors
    acc_pred = load_predictor("assets/predictors/accuracy_predictor.pth", device)
    lat_pred = load_predictor("assets/predictors/latency_predictor.pth", device)
    flash_pred = load_predictor("assets/predictors/flash_predictor.pth", device)
    sram_pred = load_predictor("assets/predictors/sram_predictor.pth", device)

    # Latency constraints and number of trials
    latency_list: List[float] = [2.0, 1.8, 1.6, 1.4, 1.2, 1.0]
    num_trials: int = 10

    results: Dict[str, List[Dict[str, Any]]] = {}

    for max_latency in latency_list:
        latency_key = f"{max_latency:.1f}"
        results[latency_key] = []

        for trial in range(num_trials):
            print(f"\n=== Latency {max_latency:.1f}, trial {trial} ===")

            # Per-trial seed for reproducibility
            seed = 1000 + int(max_latency * 10) * 100 + trial
            set_seed(seed)

            ga_config = GAConfig(
                population_size=500,
                mutation_prob=0.3,
                crossover_prob=0.9,
                patience=5,
                threshold=0.0001,
                max_gen=30,
                max_latency=max_latency,
                max_flash=1.8,
                max_sram=0.88,
            )

            searcher = GASearcher(
                encoder=encoder,
                acc_predictor=acc_pred,
                lat_predictor=lat_pred,
                flash_predictor=flash_pred,
                sram_predictor=sram_pred,
                subnet_config=subnet_config,
                config=ga_config,
            )

            best = searcher.run()

            trial_result: Dict[str, Any] = {
                "config": asdict(ga_config),
                "trial_index": trial,
                "seed": seed,
                "best_accuracy": best.accuracy,
                "best_latency": best.latency,
                "best_flash": best.flash,
                "best_sram": best.sram,
                "violation": best.violation,
            }
            results[latency_key].append(trial_result)

    os.makedirs("ga_results", exist_ok=True)
    save_path = "ga_results/ga_latency_sweep.json"

    with open(save_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nAll GA experiments finished. Results saved to {save_path}")


if __name__ == "__main__":
    main()
