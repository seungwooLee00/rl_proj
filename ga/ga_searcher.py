import torch
from torch import nn
from typing import List, Tuple
import random
import copy
from tqdm import tqdm

from common.predictor import MLPPredictor
from common.encoder import Encoder
from common.subnet_config import SubnetConfig
from ga.ga_config import Individual, GAConfig


class GASearcher:
    """
    Genetic algorithm based subnet searcher.

    The searcher finds the best subnet that satisfies latency, flash,
    and SRAM constraints while maximizing accuracy.
    """

    def __init__(
        self,
        encoder: Encoder,
        acc_predictor: MLPPredictor,
        lat_predictor: MLPPredictor,
        flash_predictor: MLPPredictor,
        sram_predictor: MLPPredictor,
        subnet_config: SubnetConfig,
        config: GAConfig,
    ) -> None:
        self.encoder = encoder
        self.acc_predictor = acc_predictor
        self.lat_predictor = lat_predictor
        self.flash_predictor = flash_predictor
        self.sram_predictor = sram_predictor

        self.subnet_config = subnet_config

        self.population_size = config.population_size
        self.mutation_prob = config.mutation_prob
        self.crossover_prob = config.crossover_prob

        self.patience = config.patience
        self.threshold = config.threshold
        self.max_gen = config.max_gen

        self.max_latency = config.max_latency
        self.max_flash = config.max_flash
        self.max_sram = config.max_sram

    def run(self) -> Individual:
        """
        Run genetic algorithm search to find the best feasible subnet.

        Returns
        -------
        Individual
            The best individual found that satisfies all constraints.
        """
        # Initialize population
        population: List[Individual] = self.random_sample(self.population_size)
        self.evaluate(population)

        best_accuracy_history: float = -float("inf")
        no_improve_count: int = 0

        best_individual: Individual = max(population, key=lambda ind: ind.accuracy)
        print(
            f"Initial Best Accuracy: {best_individual.accuracy:.4f} "
            f"(Acc: {best_individual.accuracy:.4f}, "
            f"Lat: {best_individual.latency:.3f}, "
            f"Flash: {getattr(best_individual, 'flash', 0.0):.3f}, "
            f"Sram: {getattr(best_individual, 'sram', 0.0):.3f})"
        )

        # Evolution loop
        pbar = tqdm(range(self.max_gen), desc="GA Search")
        for gen in pbar:
            # Tournament selection
            mating_pool = self.selection(population)

            # Crossover and mutation
            offspring: List[Individual] = []
            for i in range(0, len(mating_pool), 2):
                parent1 = mating_pool[i]
                parent2 = mating_pool[i + 1]

                child1, child2 = self.crossover(parent1, parent2)
                child1 = self.mutation(child1)
                child2 = self.mutation(child2)

                offspring.append(child1)
                offspring.append(child2)

            # Evaluate offspring
            self.evaluate(offspring)

            # Combine parents and offspring
            combined_pop = population + offspring

            # Split into feasible and infeasible individuals
            feasibles = [ind for ind in combined_pop if not ind.violation]
            infeasibles = [ind for ind in combined_pop if ind.violation]

            feasibles.sort(key=lambda ind: ind.accuracy, reverse=True)
            infeasibles.sort(key=lambda ind: ind.accuracy, reverse=True)

            # Build next generation
            if len(feasibles) >= self.population_size:
                population = feasibles[: self.population_size]
            else:
                needed = self.population_size - len(feasibles)
                population = feasibles + infeasibles[:needed]

            # Choose current best from feasible if possible,
            # otherwise from entire population
            if len(feasibles) > 0:
                current_best = max(feasibles, key=lambda ind: ind.accuracy)
            else:
                current_best = max(population, key=lambda ind: ind.accuracy)

            pbar.set_description(
                f"Gen {gen + 1}/{self.max_gen} | "
                f"Acc: {current_best.accuracy:.4f} | "
                f"Lat: {current_best.latency:.3f} | "
                f"Fl: {getattr(current_best, 'flash', 0.0):.3f} | "
                f"Sr: {getattr(current_best, 'sram', 0.0):.3f} | "
                f"Patience: {no_improve_count}/{self.patience} | "
                f"Feasibles: {len(feasibles)}/{self.population_size}"
            )

            # Check improvement based on accuracy
            if current_best.accuracy > best_accuracy_history + self.threshold:
                best_accuracy_history = current_best.accuracy
                no_improve_count = 0
                best_individual = copy.deepcopy(current_best)
            else:
                no_improve_count += 1

            # Early stopping by patience
            if no_improve_count >= self.patience:
                print(
                    f"\n[Early Stopping] No improvement for {self.patience} generations."
                )
                break

        print(f"\nSearch Finished. Best Accuracy: {best_individual.accuracy:.4f}")
        print(
            "Best Individual "
            f"(Acc: {best_individual.accuracy:.4f}, "
            f"Lat: {best_individual.latency:.3f}, "
            f"Flash: {best_individual.flash:.3f}, "
            f"Sram: {best_individual.sram:.3f}, "
            f"Violation: {best_individual.violation})"
        )

        return best_individual

    def evaluate(self, population: List[Individual]) -> None:
        """
        Evaluate accuracy, latency, flash, and SRAM for a population
        and set constraint violation flags.

        Parameters
        ----------
        population : List[Individual]
            Individuals to evaluate.
        """
        batch_archs = [ind.arch for ind in population]
        vecs = self.encoder.encode(batch_archs)

        accs = self.acc_predictor.run(vecs).reshape(-1)
        lats = self.lat_predictor.run(vecs).reshape(-1)
        flashes = self.flash_predictor.run(vecs).reshape(-1)
        srams = self.sram_predictor.run(vecs).reshape(-1)

        for i, ind in enumerate(population):
            ind.accuracy = accs[i].item()
            ind.latency = lats[i].item()
            ind.flash = flashes[i].item()
            ind.sram = srams[i].item()

            latency_violate = ind.latency > self.max_latency
            flash_violate = ind.flash > self.max_flash
            sram_violate = ind.sram > self.max_sram

            ind.violation = latency_violate or flash_violate or sram_violate

    def random_sample(self, n: int) -> List[Individual]:
        """
        Sample random individuals from the subnet configuration.

        Parameters
        ----------
        n : int
            Number of individuals to sample.

        Returns
        -------
        List[Individual]
            Randomly generated population.
        """
        cfg = self.subnet_config
        population: List[Individual] = []

        for _ in range(n):
            arch = {
                "ks": [random.choice(cfg.ks_list) for _ in range(cfg.n_blocks)],
                "e": [random.choice(cfg.e_list) for _ in range(cfg.n_blocks)],
                "d": [random.choice(cfg.d_list) for _ in range(cfg.n_layers)],
                "aux_a": [random.choice(cfg.aux_a_list) for _ in range(cfg.n_layers)],
                "aux_t": [random.choice(cfg.aux_t_list) for _ in range(cfg.n_layers)],
                "aux_e": [random.choice(cfg.aux_e_list) for _ in range(cfg.n_layers)],
                "aux_k": [random.choice(cfg.aux_k_list) for _ in range(cfg.n_layers)],
            }
            population.append(Individual(arch=arch))

        return population

    def selection(self, population: List[Individual]) -> List[Individual]:
        """
        Tournament selection over the population.

        Two individuals are randomly sampled and the one with higher
        accuracy is selected as the winner.

        Parameters
        ----------
        population : List[Individual]
            Current population.

        Returns
        -------
        List[Individual]
            Mating pool of selected individuals.
        """
        mating_pool: List[Individual] = []
        n: int = len(population)

        for _ in range(n):
            a: Individual = random.choice(population)
            b: Individual = random.choice(population)
            winner = a if a.accuracy > b.accuracy else b
            mating_pool.append(winner)

        return mating_pool

    def crossover(
        self,
        ind1: Individual,
        ind2: Individual,
    ) -> Tuple[Individual, Individual]:
        """
        One-point crossover for each gene group.

        For each gene group (ks, e, d, aux_*), apply crossover with
        probability self.crossover_prob and swap the tail segments
        after a randomly chosen point.

        Parameters
        ----------
        ind1 : Individual
            First parent individual.
        ind2 : Individual
            Second parent individual.

        Returns
        -------
        Tuple[Individual, Individual]
            Two child individuals after crossover.
        """
        arch1 = copy.deepcopy(ind1.arch)
        arch2 = copy.deepcopy(ind2.arch)

        for key in arch1.keys():
            if random.random() < self.crossover_prob:
                gene1 = arch1[key]
                gene2 = arch2[key]

                if len(gene1) <= 1:
                    continue

                crossover_point = random.randint(1, len(gene1) - 1)
                arch1[key] = gene1[:crossover_point] + gene2[crossover_point:]
                arch2[key] = gene2[:crossover_point] + gene1[crossover_point:]

        return Individual(arch=arch1), Individual(arch=arch2)

    def mutation(self, ind: Individual) -> Individual:
        """
        Mutate each gene with probability self.mutation_prob.

        Each mutated position is resampled from the corresponding
        candidate list in the subnet configuration.

        Parameters
        ----------
        ind : Individual
            Individual to mutate.

        Returns
        -------
        Individual
            New individual after mutation.
        """
        arch = copy.deepcopy(ind.arch)

        for key in arch.keys():
            candidates = getattr(self.subnet_config, f"{key}_list")
            for i in range(len(arch[key])):
                if random.random() < self.mutation_prob:
                    arch[key][i] = random.choice(candidates)

        return Individual(arch=arch)
