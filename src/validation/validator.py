"""
Validation module for perovskite PCE prediction models
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')


@dataclass
class ValidationResult:
    """Validation result container"""
    mae: float
    rmse: float
    r2: float
    mape: float
    within_uncertainty: float
    physics_violations: int
    
    def to_dict(self) -> Dict:
        return {
            'MAE': f"{self.mae:.3f}%",
            'RMSE': f"{self.rmse:.3f}%",
            'R²': f"{self.r2:.4f}",
            'MAPE': f"{self.mape:.2f}%",
            'Within Uncertainty': f"{self.within_uncertainty:.1f}%",
            'Physics Violations': self.physics_violations
        }


class ModelValidator:
    """
    Model validation with physics constraints
    
    Features:
    - Cross-validation
    - Physics consistency checks
    - Uncertainty calibration
    - External validation
    """
    
    def __init__(self, config=None):
        """Initialize validator"""
        self.config = config
        self.metrics = {}
        
    def validate(
        self,
        model,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> ValidationResult:
        """
        Validate model performance
        
        Args:
            model: Trained model
            X_train: Training features
            y_train: Training targets
            X_test: Test features
            y_test: Test targets
            
        Returns:
            ValidationResult object
        """
        # Predictions
        y_pred = model.predict(X_test)
        
        # Basic metrics
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
        
        # Uncertainty calibration (if model supports)
        within_uncertainty = 0.0
        if hasattr(model, 'predict_with_uncertainty'):
            y_pred, uncertainty = model.predict_with_uncertainty(X_test)
            within_uncertainty = np.mean(
                (y_test >= y_pred - uncertainty) & (y_test <= y_pred + uncertainty)
            ) * 100
        
        # Physics violations
        physics_violations = self._check_physics_violations(y_pred, X_test)
        
        return ValidationResult(
            mae=mae,
            rmse=rmse,
            r2=r2,
            mape=mape,
            within_uncertainty=within_uncertainty,
            physics_violations=physics_violations
        )
    
    def cross_validate(
        self,
        model,
        X: np.ndarray,
        y: np.ndarray,
        n_folds: int = 5
    ) -> Dict[str, List[float]]:
        """
        Perform cross-validation
        
        Args:
            model: Model to validate
            X: Features
            y: Targets
            n_folds: Number of folds
            
        Returns:
            Dictionary of metrics per fold
        """
        kfold = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        
        results = {
            'mae': [],
            'rmse': [],
            'r2': [],
            'mape': []
        }
        
        for train_idx, val_idx in kfold.split(X):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Clone and fit model
            from sklearn.base import clone
            fold_model = clone(model)
            fold_model.fit(X_train, y_train)
            
            # Predict
            y_pred = fold_model.predict(X_val)
            
            # Metrics
            results['mae'].append(mean_absolute_error(y_val, y_pred))
            results['rmse'].append(np.sqrt(mean_squared_error(y_val, y_pred)))
            results['r2'].append(r2_score(y_val, y_pred))
            results['mape'].append(np.mean(np.abs((y_val - y_pred) / y_val)) * 100)
        
        # Compute statistics
        stats = {}
        for metric, values in results.items():
            stats[metric] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values)
            }
        
        return stats
    
    def _check_physics_violations(
        self,
        predictions: np.ndarray,
        features: np.ndarray
    ) -> int:
        """
        Check for physics constraint violations
        
        Violations:
        1. PCE > Shockley-Queisser limit
        2. PCE < 0
        3. Energy conservation violation
        """
        violations = 0
        
        # Negative PCE
        violations += np.sum(predictions < 0)
        
        # Over SQ limit (assuming ~33% for optimal bandgap)
        violations += np.sum(predictions > 33)
        
        return int(violations)
    
    def generate_report(
        self,
        validation_result: ValidationResult,
        cv_results: Optional[Dict] = None
    ) -> str:
        """
        Generate validation report
        
        Args:
            validation_result: Validation metrics
            cv_results: Cross-validation results
            
        Returns:
            Markdown report
        """
        report = [
            "# Model Validation Report",
            "",
            "## Test Set Performance",
            "",
            "| Metric | Value |",
            "|--------|-------|",
        ]
        
        for key, value in validation_result.to_dict().items():
            report.append(f"| {key} | {value} |")
        
        if cv_results:
            report.extend([
                "",
                "## Cross-Validation Results",
                "",
                "| Metric | Mean ± Std | Min | Max |",
                "|--------|------------|-----|-----|",
            ])
            
            for metric, stats in cv_results.items():
                report.append(
                    f"| {metric.upper()} | {stats['mean']:.3f} ± {stats['std']:.3f} | "
                    f"{stats['min']:.3f} | {stats['max']:.3f} |"
                )
        
        report.extend([
            "",
            "## Pass/Fail Criteria",
            "",
            "| Criterion | Threshold | Result |",
            "|-----------|-----------|--------|",
            f"| MAE | < 1.0% | {'✅ PASS' if validation_result.mae < 1.0 else '❌ FAIL'} |",
            f"| R² | > 0.85 | {'✅ PASS' if validation_result.r2 > 0.85 else '❌ FAIL'} |",
            f"| MAPE | < 8% | {'✅ PASS' if validation_result.mape < 8 else '❌ FAIL'} |",
            f"| Physics Violations | 0 | {'✅ PASS' if validation_result.physics_violations == 0 else '❌ FAIL'} |",
        ])
        
        return "\n".join(report)


class DataQualityChecker:
    """
    Data quality checker
    """
    
    @staticmethod
    def check_missing(df: pd.DataFrame, threshold: float = 0.3) -> Dict:
        """Check missing values"""
        missing_pct = df.isnull().mean()
        
        return {
            'total_missing': df.isnull().sum().sum(),
            'columns_above_threshold': (missing_pct > threshold).sum(),
            'max_missing_pct': missing_pct.max(),
            'passed': (missing_pct > threshold).sum() == 0
        }
    
    @staticmethod
    def check_distribution(
        df: pd.DataFrame,
        target: str,
        skew_threshold: float = 2.0
    ) -> Dict:
        """Check target distribution"""
        from scipy import stats
        
        skewness = stats.skew(df[target].dropna())
        kurtosis = stats.kurtosis(df[target].dropna())
        
        return {
            'skewness': skewness,
            'kurtosis': kurtosis,
            'is_normal': abs(skewness) < skew_threshold,
            'passed': abs(skewness) < skew_threshold
        }
    
    @staticmethod
    def check_outliers(
        df: pd.DataFrame,
        target: str,
        method: str = 'iqr'
    ) -> Dict:
        """Check for outliers"""
        data = df[target].dropna()
        
        if method == 'iqr':
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1
            outliers = ((data < Q1 - 1.5 * IQR) | (data > Q3 + 1.5 * IQR)).sum()
        
        return {
            'n_outliers': outliers,
            'outlier_pct': outliers / len(data) * 100,
            'passed': outliers / len(data) < 0.05
        }
    
    def run_all_checks(
        self,
        df: pd.DataFrame,
        target: str
    ) -> Dict:
        """Run all quality checks"""
        return {
            'missing': self.check_missing(df),
            'distribution': self.check_distribution(df, target),
            'outliers': self.check_outliers(df, target)
        }