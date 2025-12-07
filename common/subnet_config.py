from dataclasses import dataclass, field
from typing import List

@dataclass
class SubnetConfig:
    n_layers: int = 6
    n_blocks: int = 20
    n_aux: int = 6

    ks_list: List[int] = field(default_factory=lambda: [3, 5, 7])
    e_list:  List[int] = field(default_factory=lambda: [3, 4, 6])
    d_list:  List[int] = field(default_factory=lambda: [0, 1, 2])

    base_depth: List[int] = field(default_factory=lambda: [1, 2, 2, 2, 2, 1])
    max_depth: List[int] = field(default_factory=lambda: [3, 4, 4, 4, 4, 1])

    aux_a_list: List[bool] = field(default_factory=lambda: [False, True])  # auxiliary branch activation
    aux_t_list: List[float] = field(default_factory=lambda: [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0])   # cifar100
    aux_k_list: List[int] = field(default_factory=lambda: [3, 5, 7])
    aux_e_list: List[float] = field(default_factory=lambda: [1.0, 1.5, 2.0])
    