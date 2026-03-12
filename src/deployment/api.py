"""
Deployment module for perovskite PCE prediction API
"""

from typing import Dict, List, Optional, Union
import numpy as np
import json
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')


@dataclass
class PredictionResult:
    """Prediction result container"""
    pce: float
    uncertainty: float
    confidence_interval: tuple
    feature_importance: Dict[str, float]
    
    def to_dict(self) -> Dict:
        return {
            'pce': self.pce,
            'uncertainty': self.uncertainty,
            'confidence_interval': list(self.confidence_interval),
            'feature_importance': self.feature_importance
        }
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


class PCEPredictorAPI:
    """
    PCE Prediction API
    
    REST-like interface for model serving
    
    Features:
    - Single prediction
    - Batch prediction
    - Uncertainty quantification
    - Feature importance
    """
    
    def __init__(self, model, feature_names: List[str], config=None):
        """
        Initialize API
        
        Args:
            model: Trained model
            feature_names: List of feature names
            config: Configuration
        """
        self.model = model
        self.feature_names = feature_names
        self.config = config
        self._shap_explainer = None
        
    def predict(
        self,
        features: Dict[str, float],
        return_uncertainty: bool = True,
        return_importance: bool = True
    ) -> PredictionResult:
        """
        Predict PCE for a single sample
        
        Args:
            features: Dictionary of feature values
            return_uncertainty: Whether to compute uncertainty
            return_importance: Whether to compute feature importance
            
        Returns:
            PredictionResult object
        """
        # Validate input
        self._validate_features(features)
        
        # Prepare input
        X = np.array([[features.get(name, 0) for name in self.feature_names]])
        
        # Predict
        if return_uncertainty and hasattr(self.model, 'predict_with_uncertainty'):
            pce, uncertainty = self.model.predict_with_uncertainty(X)
            pce = float(pce[0])
            uncertainty = float(uncertainty[0])
        else:
            pce = float(self.model.predict(X)[0])
            uncertainty = 0.0
        
        # Confidence interval
        ci = (pce - 1.96 * uncertainty, pce + 1.96 * uncertainty)
        
        # Feature importance
        importance = {}
        if return_importance:
            importance = self._compute_importance(X)
        
        return PredictionResult(
            pce=pce,
            uncertainty=uncertainty,
            confidence_interval=ci,
            feature_importance=importance
        )
    
    def predict_batch(
        self,
        samples: List[Dict[str, float]],
        return_uncertainty: bool = True
    ) -> List[PredictionResult]:
        """
        Batch prediction
        
        Args:
            samples: List of feature dictionaries
            return_uncertainty: Whether to compute uncertainty
            
        Returns:
            List of PredictionResult objects
        """
        return [self.predict(s, return_uncertainty) for s in samples]
    
    def optimize(
        self,
        fixed_features: Dict[str, float],
        optimize_features: List[str],
        bounds: Dict[str, tuple],
        n_iterations: int = 50
    ) -> Dict:
        """
        Optimize features for maximum PCE
        
        Args:
            fixed_features: Features to keep constant
            optimize_features: Features to optimize
            bounds: Bounds for each feature to optimize
            n_iterations: Number of optimization iterations
            
        Returns:
            Optimal configuration and predicted PCE
        """
        from scipy.optimize import minimize
        
        def objective(x):
            features = fixed_features.copy()
            for i, name in enumerate(optimize_features):
                features[name] = x[i]
            result = self.predict(features, return_uncertainty=False, return_importance=False)
            return -result.pce  # Minimize negative PCE
        
        # Initial guess (midpoint)
        x0 = [(bounds[name][0] + bounds[name][1]) / 2 for name in optimize_features]
        
        # Bounds
        opt_bounds = [bounds[name] for name in optimize_features]
        
        # Optimize
        result = minimize(objective, x0, bounds=opt_bounds, method='L-BFGS-B',
                         options={'maxiter': n_iterations})
        
        # Build optimal config
        optimal_config = fixed_features.copy()
        for i, name in enumerate(optimize_features):
            optimal_config[name] = result.x[i]
        
        return {
            'optimal_features': optimal_config,
            'predicted_pce': -result.fun,
            'optimization_success': result.success
        }
    
    def _validate_features(self, features: Dict[str, float]):
        """Validate input features"""
        missing = set(self.feature_names) - set(features.keys())
        if missing:
            # Use default values instead of raising error
            pass
        
        # Check for invalid values
        for name, value in features.items():
            if not isinstance(value, (int, float)):
                raise ValueError(f"Feature {name} must be numeric, got {type(value)}")
    
    def _compute_importance(self, X: np.ndarray) -> Dict[str, float]:
        """Compute feature importance"""
        if self._shap_explainer is None:
            try:
                import shap
                self._shap_explainer = shap.TreeExplainer(self.model)
            except:
                return {}
        
        try:
            shap_values = self._shap_explainer.shap_values(X)[0]
            return dict(zip(self.feature_names, map(float, shap_values)))
        except:
            return {}


class ModelServer:
    """
    Model server for production deployment
    
    Features:
    - Model loading
    - Request handling
    - Logging
    - Performance monitoring
    """
    
    def __init__(self, model_path: str, config=None):
        """
        Initialize server
        
        Args:
            model_path: Path to saved model
            config: Server configuration
        """
        self.model_path = model_path
        self.config = config
        self.model = None
        self.api = None
        self.request_count = 0
        
    def load_model(self):
        """Load model from disk"""
        import joblib
        self.model = joblib.load(self.model_path)
        print(f"✓ Model loaded from {self.model_path}")
        
    def start(self):
        """Start server"""
        self.load_model()
        print("✓ Server ready")
        
    def handle_request(self, request: Dict) -> Dict:
        """
        Handle prediction request
        
        Args:
            request: Request dictionary with 'features' key
            
        Returns:
            Response dictionary
        """
        self.request_count += 1
        
        try:
            result = self.api.predict(request['features'])
            return {
                'success': True,
                'result': result.to_dict()
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def get_stats(self) -> Dict:
        """Get server statistics"""
        return {
            'request_count': self.request_count,
            'model_path': self.model_path,
            'status': 'running'
        }