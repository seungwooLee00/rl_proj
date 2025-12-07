from dataclasses import dataclass
from typing import List

import numpy as np
import torch
import torch.nn as nn


@dataclass
class MLPPredictorConfig:
    """
    Configuration for MLPPredictor network.
    """
    input_dim: int = 198
    hidden_size: int = 512
    n_layers: int = 3
    dropout: float = 0.1
    use_sigmoid: bool = False


class MLPPredictor(nn.Module):
    """
    Multi-layer perceptron predictor for scalar regression or bounded prediction.
    """

    def __init__(
        self,
        input_dim: int = 198,
        hidden_size: int = 512,
        n_layers: int = 3,
        dropout: float = 0.1,
        use_sigmoid: bool = False,
    ) -> None:
        """
        Initialize MLP predictor.

        Args:
            input_dim (int): Dimensionality of input features.
            hidden_size (int): Number of hidden units in each layer.
            n_layers (int): Number of hidden layers.
            dropout (float): Dropout probability applied after each hidden layer.
            use_sigmoid (bool): If True, apply sigmoid to the final output.
        """
        super().__init__()

        self.input_dim: int = input_dim
        self.hidden_size: int = hidden_size
        self.n_layers: int = n_layers
        self.dropout: float = dropout
        self.use_sigmoid: bool = use_sigmoid

        layers: List[nn.Module] = []
        for i in range(n_layers):
            in_dim: int = self.input_dim if i == 0 else hidden_size

            layers.append(nn.Linear(in_dim, hidden_size))
            layers.append(nn.LayerNorm(hidden_size))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(dropout))

        layers.append(nn.Linear(hidden_size, 1))
        if use_sigmoid:
            layers.append(nn.Sigmoid())

        self.layers: nn.Sequential = nn.Sequential(*layers)

    @property
    def config(self) -> MLPPredictorConfig:
        """
        Export current model hyperparameters as a configuration object.

        Returns:
            MLPPredictorConfig: Dataclass with the current predictor settings.
        """
        return MLPPredictorConfig(
            input_dim=self.input_dim,
            hidden_size=self.hidden_size,
            n_layers=self.n_layers,
            dropout=self.dropout,
            use_sigmoid=self.use_sigmoid,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the MLP predictor.

        Args:
            x (torch.Tensor): Input tensor of shape (B, input_dim).

        Returns:
            torch.Tensor: Output tensor of shape (B,) after optional sigmoid.
        """
        # Squeeze final dimension because last linear outputs shape (B, 1)
        return self.layers(x).squeeze(-1)

    def run(self, x: np.ndarray) -> torch.Tensor:
        """
        Run the predictor on a NumPy input array.

        Converts NumPy input to a tensor on the same device as the model
        parameters and then calls forward.

        Args:
            x (np.ndarray): Input array of shape (B, input_dim).

        Returns:
            torch.Tensor: Model output tensor of shape (B,).
        """
        # Use the model's parameter device to place input tensor
        device = next(self.parameters()).device

        # Convert numpy array to float tensor on target device
        x_tensor: torch.Tensor = torch.from_numpy(x).to(
            device=device,
            dtype=torch.float32,
        )

        return self.forward(x_tensor)
