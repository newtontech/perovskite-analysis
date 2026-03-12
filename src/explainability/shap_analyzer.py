"""
SHAP-based Explainability for Perovskite PCE Prediction
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
import warnings
warnings.filterwarnings('ignore')


class PhysicsConstrainedSHAP:
    """
    SHAP analyzer with physics constraint validation
    
    Features:
    - Compute SHAP values
    - Validate against physics principles
    - Detect anomalies and potential discoveries
    """
    
    def __init__(
        self,
        model: Any,
        feature_names: List[str],
        physics_rules: Optional[Dict] = None
    ):
        """
        Initialize SHAP analyzer
        
        Args:
            model: Trained model
            feature_names: List of feature names
            physics_rules: Dictionary of physics validation rules
        """
        self.model = model
        self.feature_names = feature_names
        self.physics_rules = physics_rules or self._default_physics_rules()
        self.explainer = None
        self.shap_values = None
        
    def _default_physics_rules(self) -> Dict:
        """Default physics rules for validation"""
        return {
            'bandgap': {
                'optimal_range': (1.2, 1.8),  # eV
                'importance_rank': 'high',
                'direction': 'optimal'  # Should have optimal value
            },
            'annealing_temperature': {
                'importance_rank': 'high',
                'direction': 'positive'  # Higher generally better (up to limit)
            },
            'grain_size': {
                'importance_rank': 'medium',
                'direction': 'positive'
            },
            'defect_density': {
                'importance_rank': 'high',
                'direction': 'negative'  # Lower is better
            }
        }
    
    def fit(self, X: np.ndarray):
        """
        Fit SHAP explainer
        
        Args:
            X: Background dataset
        """
        try:
            import shap
            
            # Choose explainer based on model type
            if hasattr(self.model, 'predict_proba'):
                self.explainer = shap.TreeExplainer(self.model)
            else:
                self.explainer = shap.KernelExplainer(
                    self.model.predict,
                    shap.kmeans(X, 50)  # Summarize background
                )
        except ImportError:
            raise ImportError("Please install shap: pip install shap")
    
    def explain(
        self,
        X: np.ndarray,
        check_physics: bool = True
    ) -> Dict:
        """
        Compute SHAP values and validate physics
        
        Args:
            X: Instances to explain
            check_physics: Whether to validate physics constraints
            
        Returns:
            Dictionary with SHAP values and validation results
        """
        if self.explainer is None:
            raise ValueError("Please call fit() first")
        
        # Compute SHAP values
        self.shap_values = self.explainer.shap_values(X)
        
        # Handle multi-output models
        if isinstance(self.shap_values, list):
            self.shap_values = self.shap_values[0]
        
        # Feature importance
        importance = self._compute_importance()
        
        # Physics validation
        validation = None
        if check_physics:
            validation = self._validate_physics()
        
        return {
            'shap_values': self.shap_values,
            'feature_importance': importance,
            'physics_validation': validation
        }
    
    def _compute_importance(self) -> Dict[str, float]:
        """Compute global feature importance"""
        if self.shap_values is None:
            return {}
        
        # Mean absolute SHAP value per feature
        importance = np.abs(self.shap_values).mean(axis=0)
        
        return dict(zip(self.feature_names, importance))
    
    def _validate_physics(self) -> Dict:
        """
        Validate SHAP values against physics principles
        
        Returns:
            Dictionary of validation results
        """
        if self.shap_values is None:
            return {}
        
        importance = self._compute_importance()
        validation = {
            'passed': True,
            'violations': [],
            'discoveries': []
        }
        
        # Sort features by importance
        sorted_features = sorted(
            importance.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        for feature, rule in self.physics_rules.items():
            if feature not in importance:
                continue
            
            # Check importance rank
            rank = next(
                i for i, (f, _) in enumerate(sorted_features)
                if f == feature
            ) + 1
            
            expected_rank = rule.get('importance_rank', 'any')
            if expected_rank == 'high' and rank > 5:
                validation['violations'].append({
                    'feature': feature,
                    'violation': 'low_importance',
                    'expected': 'high',
                    'actual_rank': rank
                })
                validation['passed'] = False
            
            # Check direction
            mean_shap = self.shap_values[:, self.feature_names.index(feature)].mean()
            expected_dir = rule.get('direction', 'any')
            
            if expected_dir == 'positive' and mean_shap < 0:
                validation['discoveries'].append({
                    'feature': feature,
                    'finding': 'unexpected_negative_effect',
                    'mean_shap': mean_shap,
                    'interpretation': f'{feature} shows unexpected negative effect'
                })
        
        return validation
    
    def get_top_features(self, n: int = 10) -> List[Tuple[str, float]]:
        """
        Get top N most important features
        
        Args:
            n: Number of features to return
            
        Returns:
            List of (feature_name, importance) tuples
        """
        importance = self._compute_importance()
        sorted_features = sorted(
            importance.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return sorted_features[:n]
    
    def generate_report(self) -> str:
        """
        Generate markdown report
        
        Returns:
            Markdown formatted report
        """
        importance = self._compute_importance()
        sorted_features = sorted(
            importance.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        report = [
            "# SHAP Analysis Report",
            "",
            "## Feature Importance Ranking",
            "",
            "| Rank | Feature | Importance |",
            "|------|---------|------------|"
        ]
        
        for i, (feat, imp) in enumerate(sorted_features[:10], 1):
            report.append(f"| {i} | {feat} | {imp:.4f} |")
        
        # Physics validation
        if hasattr(self, '_last_validation') and self._last_validation:
            report.extend([
                "",
                "## Physics Validation",
                "",
                f"**Status**: {'✅ Passed' if self._last_validation['passed'] else '⚠️ Violations Found'}",
                ""
            ])
            
            if self._last_validation['violations']:
                report.append("### Violations")
                for v in self._last_validation['violations']:
                    report.append(f"- {v['feature']}: {v['violation']}")
            
            if self._last_validation['discoveries']:
                report.extend(["", "### Potential Discoveries 🔬"])
                for d in self._last_validation['discoveries']:
                    report.append(f"- {d['feature']}: {d['interpretation']}")
        
        return "\n".join(report)


class CausalAnalyzer:
    """
    Causal analysis for feature relationships
    
    Features:
    - Causal discovery (PC algorithm)
    - Intervention analysis
    - Counterfactual explanations
    """
    
    def __init__(self, feature_names: List[str]):
        """
        Initialize causal analyzer
        
        Args:
            feature_names: List of feature names
        """
        self.feature_names = feature_names
        self.causal_graph = None
    
    def discover_causal_structure(
        self,
        data: pd.DataFrame,
        method: str = 'pc',
        alpha: float = 0.05
    ) -> Dict:
        """
        Discover causal relationships
        
        Args:
            data: Dataset
            method: Discovery method ('pc', 'ges')
            alpha: Significance level
            
        Returns:
            Causal graph as adjacency matrix
        """
        # Simplified causal discovery
        # In practice, use causal discovery libraries like causal-learn
        
        n_features = len(self.feature_names)
        self.causal_graph = np.zeros((n_features, n_features))
        
        # Use correlation as proxy for causal relationship
        # (This is a simplification - real causal discovery is more complex)
        corr_matrix = data.corr().values
        
        # Threshold
        self.causal_graph = (np.abs(corr_matrix) > 0.5).astype(int)
        np.fill_diagonal(self.causal_graph, 0)
        
        return {
            'adjacency_matrix': self.causal_graph,
            'edges': self._extract_edges()
        }
    
    def _extract_edges(self) -> List[Tuple[str, str]]:
        """Extract edges from graph"""
        if self.causal_graph is None:
            return []
        
        edges = []
        for i in range(len(self.feature_names)):
            for j in range(len(self.feature_names)):
                if self.causal_graph[i, j] == 1:
                    edges.append((self.feature_names[i], self.feature_names[j]))
        
        return edges
    
    def intervention_effect(
        self,
        model: Any,
        data: np.ndarray,
        feature_idx: int,
        intervention_value: float
    ) -> float:
        """
        Compute intervention effect
        
        Args:
            model: Trained model
            data: Original data
            feature_idx: Index of feature to intervene
            intervention_value: Value to set
            
        Returns:
            Average treatment effect
        """
        # Original predictions
        original_pred = model.predict(data)
        
        # Intervened predictions
        intervened_data = data.copy()
        intervened_data[:, feature_idx] = intervention_value
        intervened_pred = model.predict(intervened_data)
        
        # Average treatment effect
        ate = np.mean(intervened_pred - original_pred)
        
        return ate
    
    def counterfactual_explanation(
        self,
        model: Any,
        sample: np.ndarray,
        changes: Dict[int, float]
    ) -> Dict:
        """
        Generate counterfactual explanation
        
        Args:
            model: Trained model
            sample: Original sample
            changes: Dictionary of {feature_idx: new_value}
            
        Returns:
            Counterfactual analysis results
        """
        # Original prediction
        original_pred = model.predict(sample.reshape(1, -1))[0]
        
        # Counterfactual sample
        cf_sample = sample.copy()
        for idx, value in changes.items():
            cf_sample[idx] = value
        
        cf_pred = model.predict(cf_sample.reshape(1, -1))[0]
        
        # Attribution
        delta = cf_pred - original_pred
        
        return {
            'original_prediction': original_pred,
            'counterfactual_prediction': cf_pred,
            'delta': delta,
            'changes': {self.feature_names[k]: v for k, v in changes.items()}
        }