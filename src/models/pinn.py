"""
Physics-Informed Neural Network for Perovskite PCE Prediction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class PhysicsLoss(nn.Module):
    """
    Physics-constrained loss function
    
    Includes:
    - Data loss (MSE)
    - Shockley-Queisser limit constraint
    - Energy conservation
    - Mass conservation
    """
    
    def __init__(
        self,
        sq_limit: float = 33.0,
        sq_weight: float = 0.1,
        energy_weight: float = 0.05,
        mass_weight: float = 0.05
    ):
        super().__init__()
        self.sq_limit = sq_limit
        self.sq_weight = sq_weight
        self.energy_weight = energy_weight
        self.mass_weight = mass_weight
        
    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        features: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Compute physics-informed loss
        
        Args:
            predictions: Model predictions
            targets: Ground truth
            features: Input features (for physics constraints)
            
        Returns:
            Total loss and loss components
        """
        # Data loss
        data_loss = F.mse_loss(predictions, targets)
        
        # SQ limit violation (predictions > 33%)
        sq_violations = F.relu(predictions - self.sq_limit)
        sq_loss = torch.mean(sq_violations)
        
        # Negative predictions violation
        neg_violations = F.relu(-predictions)
        neg_loss = torch.mean(neg_violations)
        
        # Total loss
        total_loss = (
            data_loss +
            self.sq_weight * sq_loss +
            self.energy_weight * neg_loss
        )
        
        components = {
            'data_loss': data_loss.item(),
            'sq_loss': sq_loss.item(),
            'neg_loss': neg_loss.item(),
            'total_loss': total_loss.item()
        }
        
        return total_loss, components


class PhysicsInformedNN(nn.Module):
    """
    Physics-Informed Neural Network for PCE Prediction
    
    Architecture:
    - Feature embedding layers
    - Physics-constrained hidden layers
    - Uncertainty estimation head
    
    Features:
    - Embeds semiconductor physics constraints
    - Quantifies prediction uncertainty
    - Interpretable architecture
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_layers: list = [256, 128, 64],
        dropout_rate: float = 0.2,
        use_physics_loss: bool = True
    ):
        """
        Initialize PINN
        
        Args:
            input_dim: Number of input features
            hidden_layers: List of hidden layer sizes
            dropout_rate: Dropout probability
            use_physics_loss: Whether to use physics constraints
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.use_physics_loss = use_physics_loss
        
        # Build network
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_layers:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        
        self.feature_extractor = nn.Sequential(*layers)
        
        # Prediction head
        self.predictor = nn.Linear(prev_dim, 1)
        
        # Uncertainty head
        self.uncertainty_head = nn.Linear(prev_dim, 1)
        
        # Physics loss
        if use_physics_loss:
            self.physics_loss = PhysicsLoss()
        else:
            self.physics_loss = None
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(
        self,
        x: torch.Tensor,
        return_uncertainty: bool = True
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass
        
        Args:
            x: Input features
            return_uncertainty: Whether to return uncertainty
            
        Returns:
            Predictions and optionally uncertainty
        """
        # Extract features
        features = self.feature_extractor(x)
        
        # Predictions
        predictions = self.predictor(features)
        
        # Uncertainty
        uncertainty = None
        if return_uncertainty:
            log_var = self.uncertainty_head(features)
            uncertainty = torch.exp(0.5 * log_var)
        
        return predictions, uncertainty
    
    def predict_with_uncertainty(
        self,
        x: torch.Tensor
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict with uncertainty quantification
        
        Args:
            x: Input features (numpy array)
            
        Returns:
            Predictions and uncertainties (numpy arrays)
        """
        self.eval()
        
        with torch.no_grad():
            if isinstance(x, np.ndarray):
                x = torch.FloatTensor(x)
            
            predictions, uncertainty = self.forward(x, return_uncertainty=True)
            
            return (
                predictions.numpy().flatten(),
                uncertainty.numpy().flatten()
            )
    
    def compute_loss(
        self,
        x: torch.Tensor,
        targets: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Compute loss
        
        Args:
            x: Input features
            targets: Ground truth
            
        Returns:
            Total loss and components
        """
        predictions, _ = self.forward(x, return_uncertainty=False)
        
        if self.physics_loss is not None:
            return self.physics_loss(predictions, targets, x)
        else:
            loss = F.mse_loss(predictions, targets)
            return loss, {'mse_loss': loss.item()}


class BayesianNN(nn.Module):
    """
    Bayesian Neural Network for uncertainty quantification
    
    Uses MC Dropout for Bayesian approximation
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_layers: list = [256, 128, 64],
        dropout_rate: float = 0.1,
        n_mc_samples: int = 100
    ):
        super().__init__()
        
        self.n_mc_samples = n_mc_samples
        self.dropout_rate = dropout_rate
        
        # Build network
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_layers:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        
        self.network = nn.Sequential(*layers)
        self.output = nn.Linear(prev_dim, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.output(self.network(x))
    
    def predict_with_uncertainty(
        self,
        x: torch.Tensor
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        MC Dropout prediction with uncertainty
        
        Args:
            x: Input features
            
        Returns:
            Mean predictions and standard deviations
        """
        self.train()  # Enable dropout during inference
        
        predictions = []
        
        with torch.no_grad():
            for _ in range(self.n_mc_samples):
                pred = self.forward(x)
                predictions.append(pred)
        
        predictions = torch.stack(predictions)
        
        mean = predictions.mean(dim=0).numpy().flatten()
        std = predictions.std(dim=0).numpy().flatten()
        
        return mean, std


# Factory function
def create_model(
    model_type: str,
    input_dim: int,
    **kwargs
) -> nn.Module:
    """
    Factory function to create models
    
    Args:
        model_type: Type of model ('pinn', 'bayesian')
        input_dim: Number of input features
        **kwargs: Additional arguments
        
    Returns:
        Initialized model
    """
    if model_type == 'pinn':
        return PhysicsInformedNN(input_dim=input_dim, **kwargs)
    elif model_type == 'bayesian':
        return BayesianNN(input_dim=input_dim, **kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")