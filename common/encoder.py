from typing import Dict, Any, List, Union
import numpy as np

from common.subnet_config import SubnetConfig


class Encoder:
    """
    Encoder for architecture configurations into one-hot vector representations.
    """

    def __init__(self, subnet_config: SubnetConfig) -> None:
        """
        Initialize the encoder from a subnet configuration.

        Args:
            subnet_config (SubnetConfig): Configuration object describing
                layers, blocks, and candidate values for architecture choices.
        """
        info: Dict[str, Any] = create_onehot_info(subnet_config)

        # Keep references required at runtime
        self.info: Dict[str, Any] = info
        self.cfg: SubnetConfig = subnet_config

        self.n_layers: int = info["n_layers"]
        self.n_blocks: int = info["n_blocks"]
        self.base_depth: List[int] = info["base_depth"]
        self.block2layer: List[int] = info["block2layer"]
        self.layer2block: List[int] = info["layer2block"]
        self.n_dim: int = info["n_dim"]

        self.k_info: Dict[str, Any] = info["k_info"]
        self.e_info: Dict[str, Any] = info["e_info"]
        self.a_t_info: Dict[str, Any] = info["a_t_info"]
        self.a_e_info: Dict[str, Any] = info["a_e_info"]
        self.a_k_info: Dict[str, Any] = info["a_k_info"]

    def encode(
        self,
        arch_list: Union[Dict[str, Any], List[Dict[str, Any]]],
    ) -> np.ndarray:
        """
        Encode architecture configuration(s) into one-hot vectors.

        Each architecture dict is expected to contain:
            - "ks": List[...]  kernel size choices per block
            - "e":  List[...]  expansion ratio choices per block
            - "d":  List[int]  depth increments per layer
            - "aux_a": List[bool]  mask for auxiliary branch activation per layer
            - "aux_t": List[...]   auxiliary type choices per layer
            - "aux_e": List[...]   auxiliary expansion choices per layer
            - "aux_k": List[...]   auxiliary kernel choices per layer

        Args:
            arch_list (Union[Dict[str, Any], List[Dict[str, Any]]]):
                Single architecture dict or list of architecture dicts.

        Returns:
            np.ndarray: Encoded one-hot matrix with shape (B, n_dim),
                where B is the batch size.
        """
        if isinstance(arch_list, dict):
            arch_list_list: List[Dict[str, Any]] = [arch_list]
        else:
            arch_list_list = arch_list

        batch_size: int = len(arch_list_list)
        out: np.ndarray = np.zeros((batch_size, self.n_dim), dtype=np.float32)

        for i, arch in enumerate(arch_list_list):
            ks = arch["ks"]
            e = arch["e"]
            d = arch["d"]
            vec: np.ndarray = out[i]

            # Encode per-block kernel size and expansion ratio for active blocks only
            for b_idx in range(self.n_blocks):
                layer_id: int = self.block2layer[b_idx]
                block_in_layer: int = b_idx - self.layer2block[layer_id]

                if layer_id == self.n_layers - 1:
                    active_slots: int = self.base_depth[layer_id]
                else:
                    active_slots = self.base_depth[layer_id] + d[layer_id]

                if block_in_layer < active_slots:
                    vec[self.k_info["val2id"][b_idx][ks[b_idx]]] = 1.0
                    vec[self.e_info["val2id"][b_idx][e[b_idx]]] = 1.0

            aux_a = arch["aux_a"]
            aux_t = arch["aux_t"]
            aux_e = arch["aux_e"]
            aux_k = arch["aux_k"]

            # Encode per-layer auxiliary branches only if activated
            for l in range(self.n_layers):
                if aux_a[l]:
                    if self.a_t_info and aux_t is not None:
                        vec[self.a_t_info["val2id"][l][aux_t[l]]] = 1.0
                    if self.a_e_info and aux_e is not None:
                        vec[self.a_e_info["val2id"][l][aux_e[l]]] = 1.0
                    if self.a_k_info and aux_k is not None:
                        vec[self.a_k_info["val2id"][l][aux_k[l]]] = 1.0

        return out


def create_onehot_info(config: SubnetConfig) -> Dict[str, Any]:
    """
    Build one-hot index mapping and shape information from a subnet configuration.

    The configuration is expected to provide:
        - n_layers (int): Number of layers.
        - base_depth (List[int]): Base number of active blocks per layer.
        - d_list (List[int]): Candidate depth increments (for max depth derivation).
        - ks_list: Candidate kernel sizes.
        - e_list: Candidate expansion ratios.
        - aux_t_list, aux_e_list, aux_k_list: Candidate values for auxiliary branches.

    The returned dictionary contains:
        - k_info / e_info / a_t_info / a_e_info / a_k_info:
            Dict with keys "val2id", "id2val", "L", "R" defining index ranges and mappings.
        - block2layer (List[int]): Mapping from block index to layer index.
        - layer2block (List[int]): Starting block index for each layer.
        - n_layers (int): Number of layers.
        - n_blocks (int): Total number of blocks.
        - base_depth (List[int]): Base depth per layer.
        - n_dim (int): Total dimensionality of the one-hot vector.
    
    Args:
        config (SubnetConfig): Subnet configuration describing architecture space.

    Returns:
        Dict[str, Any]: Dictionary with index mappings and meta information
            for one-hot encoding.
    """
    n_layers: int = int(config.n_layers)
    base_depth: List[int] = list(config.base_depth)
    max_depth: int = max(config.d_list)

    # Compute maximum depth per layer based on base depth and global max increment
    cur_depths: List[int] = []
    for layer_id in range(n_layers - 1):
        cur: int = base_depth[layer_id] + (max_depth if layer_id < n_layers - 1 else 0)
        cur_depths.append(cur)

    # Build block <-> layer mapping given maximum possible depth
    block2layer: List[int] = []
    layer2block: List[int] = []
    offset: int = 0
    for layer_id, cur in enumerate(cur_depths):
        layer2block.append(offset)
        block2layer.extend([layer_id] * cur)
        offset += cur
    n_blocks: int = offset

    info: Dict[str, Any] = {}
    n_dim: int = 0

    def _build_info(candidates: List[Any], n: int) -> Dict[str, List[Any]]:
        """
        Create one-hot index ranges and value mappings for a repeated structure.

        Args:
            candidates (List[Any]): Candidate values to be one-hot encoded.
            n (int): Number of repetitions (e.g., blocks or layers).

        Returns:
            Dict[str, List[Any]]: Dictionary with keys:
                - "id2val": per-position mapping from index to value
                - "val2id": per-position mapping from value to index
                - "L": left index (inclusive) of each position in global vector
                - "R": right index (exclusive) of each position in global vector
        """
        nonlocal n_dim

        d: Dict[str, List[Any]] = {"id2val": [], "val2id": [], "L": [], "R": []}
        for pos in range(n):
            d["val2id"].append({})
            d["id2val"].append({})
            d["L"].append(n_dim)

            # Assign continuous index range for all candidates at this position
            for c in candidates:
                d["val2id"][pos][c] = n_dim
                d["id2val"][pos][n_dim] = c
                n_dim += 1

            d["R"].append(n_dim)
        return d

    # Per-block: kernel size and expansion ratio
    info["k_info"] = _build_info(list(config.ks_list), n_blocks)
    info["e_info"] = _build_info(list(config.e_list), n_blocks)

    # Per-layer: auxiliary branches (type / expansion / kernel)
    info["a_t_info"] = _build_info(list(config.aux_t_list), n_layers)
    info["a_e_info"] = _build_info(list(config.aux_e_list), n_layers)
    info["a_k_info"] = _build_info(list(config.aux_k_list), n_layers)

    info["block2layer"] = block2layer
    info["layer2block"] = layer2block
    info["n_layers"] = n_layers
    info["n_blocks"] = n_blocks
    info["base_depth"] = base_depth
    info["n_dim"] = n_dim

    return info
