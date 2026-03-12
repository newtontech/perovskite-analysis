"""
Configuration for Perovskite PCE Prediction Project
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional
from pathlib import Path

@dataclass
class DataConfig:
    """Data configuration"""
    raw_data_path: str = "data/raw/perovskite_raw_data_for_import.csv"
    processed_data_path: str = "data/processed/"
    train_test_split: float = 0.8
    random_seed: int = 42
    
    # Target variable
    target_column: str = "jv_reverse_scan_pce"
    
    # Feature columns
    feature_columns: List[str] = field(default_factory=lambda: [
        "perovskite_composition",
        "perovskite_thickness", 
        "annealing_temperature",
        "annealing_time",
        "bandgap",
        "grain_size"
    ])
    
    # Missing value threshold
    missing_threshold: float = 0.3

@dataclass
class ModelConfig:
    """Model configuration"""
    model_type: str = "pinn"  # pinn, ensemble, gp
    
    # Neural network
    hidden_layers: List[int] = field(default_factory=lambda: [256, 128, 64])
    dropout_rate: float = 0.2
    activation: str = "relu"
    
    # Physics constraints
    use_physics_loss: bool = True
    sq_limit_weight: float = 0.1
    energy_conservation_weight: float = 0.05
    
    # Training
    learning_rate: float = 0.001
    batch_size: int = 32
    epochs: int = 100
    early_stopping_patience: int = 10
    
    # Multi-fidelity
    use_multifidelity: bool = True
    dft_weight: float = 0.3

@dataclass
class FeatureConfig:
    """Feature engineering configuration"""
    use_physics_features: bool = True
    use_structure_features: bool = True
    use_graph_features: bool = False
    
    # Physics features
    calculate_bandgap: bool = True
    calculate_tolerance_factor: bool = True
    
    # Feature selection
    feature_selection_method: str = "shap"  # shap, mutual_info, correlation
    n_features: int = 20

@dataclass
class ValidationConfig:
    """Validation configuration"""
    cv_folds: int = 5
    metrics: List[str] = field(default_factory=lambda: ["mae", "rmse", "r2", "mape"])
    
    # Performance thresholds
    mae_threshold: float = 1.0  # %
    r2_threshold: float = 0.85
    mape_threshold: float = 8.0  # %

@dataclass
class ExplainabilityConfig:
    """Explainability configuration"""
    use_shap: bool = True
    shap_samples: int = 100
    
    # Physics validation
    validate_physics_consistency: bool = True
    
    # Causal analysis
    use_causal_discovery: bool = True
    causal_alpha: float = 0.05

@dataclass
class Config:
    """Main configuration"""
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    explainability: ExplainabilityConfig = field(default_factory=ExplainabilityConfig)
    
    # Paths
    project_root: Path = field(default_factory=lambda: Path(__file__).parent.parent.parent)
    output_dir: str = "outputs"
    log_dir: str = "logs"
    
    def __post_init__(self):
        # Create directories
        (self.project_root / self.output_dir).mkdir(parents=True, exist_ok=True)
        (self.project_root / self.log_dir).mkdir(parents=True, exist_ok=True)

# Global config instance
config = Config()