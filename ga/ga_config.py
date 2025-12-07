from dataclasses import dataclass, field
from typing import Dict, Any


@dataclass
class Individual:
    """
    Representation of a subnet architecture and its evaluated metrics.

    Attributes
    ----------
    arch : Dict[str, Any]
        Architecture configuration including ks, e, d, aux parameters.
    accuracy : float
        Predicted accuracy score.
    latency : float
        Predicted latency.
    flash : float
        Predicted flash usage.
    sram : float
        Predicted SRAM usage.
    violation : bool
        Constraint violation flag.
    """
    arch: Dict[str, Any] = field(default_factory=dict)
    accuracy: float = 0.0
    latency: float = 0.0
    flash: float = 0.0
    sram: float = 0.0
    violation: bool = True


@dataclass
class GAConfig:
    """
    Configuration for genetic algorithm search.

    Attributes
    ----------
    population_size : int
        Number of individuals in each generation.
    mutation_prob : float
        Mutation probability per gene.
    crossover_prob : float
        Crossover probability per gene group.
    patience : int
        Early stopping patience based on accuracy improvement.
    threshold : float
        Minimum accuracy improvement required to reset patience.
    max_gen : int
        Maximum number of generations.
    max_latency : float
        Maximum allowed latency for feasible individuals.
    max_flash : float
        Maximum allowed flash usage.
    max_sram : float
        Maximum allowed SRAM usage.
    """
    population_size: int = 500
    mutation_prob: float = 0.1
    crossover_prob: float = 0.9

    patience: int = 5
    threshold: float = 0.01
    max_gen: int = 30

    max_latency: float = 1.5
    max_flash: float = 1.8
    max_sram: float = 880.0
