"""Data cleaner for perovskite solar cells datasets.

Handles missing values, outliers, physical consistency validation,
and normalization/standardization.
"""

import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


class DataCleaner:
    """Clean and preprocess perovskite solar cell data.
    
    Handles missing values, outliers, physical consistency checks,
    and data normalization/standardization.
    
    Attributes:
        config: Configuration dictionary for cleaning parameters.
        
    Example:
        >>> cleaner = DataCleaner()
        >>> df_clean = cleaner.clean(df)
        >>> df_normalized = cleaner.normalize(df_clean, method="minmax")
    """
    
    # Physical constraints for perovskite solar cell parameters
    PHYSICAL_CONSTRAINTS = {
        "efficiency": {"min": 0, "max": 100, "typical_max": 30},
        "voc": {"min": 0, "max": 10, "typical_max": 1.5},
        "jsc": {"min": 0, "max": 100, "typical_max": 30},
        "ff": {"min": 0, "max": 100, "typical_max": 90},
        "bandgap": {"min": 0, "max": 10, "typical_max": 4},
        "thickness": {"min": 0, "max": 10000, "typical_max": 1000},
        "temperature": {"min": 0, "max": 1000, "typical_max": 400},
    }
    
    def __init__(
        self,
        config: Optional[Dict] = None,
    ):
        """Initialize the data cleaner.
        
        Args:
            config: Configuration dictionary with cleaning parameters.
                - missing_threshold: Maximum allowed missing percentage (default: 0.5)
                - outlier_method: Method for outlier detection ('zscore', 'iqr', 'isolation')
                - outlier_threshold: Threshold for outlier detection
                - imputation_strategy: Strategy for missing value imputation
        """
        self.config = config or {}
        self.config.setdefault("missing_threshold", 0.5)
        self.config.setdefault("outlier_method", "iqr")
        self.config.setdefault("outlier_threshold", 3.0)
        self.config.setdefault("imputation_strategy", "median")
        
        # Store fitted parameters for normalization
        self._fitted_params: Dict[str, Dict] = {}
    
    def clean(
        self,
        df: pd.DataFrame,
        handle_missing: bool = True,
        handle_outliers: bool = True,
        validate_physics: bool = True,
    ) -> pd.DataFrame:
        """Apply full cleaning pipeline to the data.
        
        Args:
            df: Input DataFrame.
            handle_missing: Whether to handle missing values.
            handle_outliers: Whether to handle outliers.
            validate_physics: Whether to validate physical consistency.
            
        Returns:
            Cleaned DataFrame.
        """
        df_clean = df.copy()
        
        logger.info(f"Starting cleaning pipeline. Input shape: {df_clean.shape}")
        
        # Step 1: Remove columns with too many missing values
        df_clean = self._remove_sparse_columns(df_clean)
        
        # Step 2: Validate physical constraints
        if validate_physics:
            df_clean = self.validate_physical_consistency(df_clean)
        
        # Step 3: Handle missing values
        if handle_missing:
            df_clean = self.handle_missing_values(df_clean)
        
        # Step 4: Handle outliers
        if handle_outliers:
            df_clean = self.handle_outliers(df_clean)
        
        logger.info(f"Cleaning complete. Output shape: {df_clean.shape}")
        
        return df_clean
    
    def _remove_sparse_columns(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Remove columns with too many missing values.
        
        Args:
            df: Input DataFrame.
            
        Returns:
            DataFrame with sparse columns removed.
        """
        threshold = self.config["missing_threshold"]
        missing_pct = df.isnull().sum() / len(df)
        
        cols_to_keep = missing_pct[missing_pct <= threshold].index.tolist()
        
        removed_cols = set(df.columns) - set(cols_to_keep)
        if removed_cols:
            logger.info(f"Removing sparse columns (>{threshold*100}% missing): {removed_cols}")
        
        return df[cols_to_keep]
    
    def handle_missing_values(
        self,
        df: pd.DataFrame,
        strategy: Optional[str] = None,
        fill_values: Optional[Dict[str, Union[str, float]]] = None,
    ) -> pd.DataFrame:
        """Handle missing values in the DataFrame.
        
        Args:
            df: Input DataFrame.
            strategy: Imputation strategy ('mean', 'median', 'mode', 'knn', 'constant', 'drop').
            fill_values: Dictionary mapping column names to fill values for 'constant' strategy.
            
        Returns:
            DataFrame with missing values handled.
        """
        strategy = strategy or self.config["imputation_strategy"]
        df_clean = df.copy()
        
        # Check for missing values
        missing_count = df_clean.isnull().sum().sum()
        if missing_count == 0:
            logger.info("No missing values found")
            return df_clean
        
        logger.info(f"Handling {missing_count} missing values using '{strategy}' strategy")
        
        if strategy == "drop":
            # Drop rows with any missing values
            df_clean = df_clean.dropna()
            logger.info(f"Dropped rows with missing values. New shape: {df_clean.shape}")
            
        elif strategy == "constant":
            # Fill with constant values
            if fill_values:
                for col, value in fill_values.items():
                    if col in df_clean.columns:
                        df_clean[col] = df_clean[col].fillna(value)
            else:
                raise ValueError("fill_values must be provided for 'constant' strategy")
                
        elif strategy in ["mean", "median", "mode"]:
            # Simple statistical imputation
            df_clean = self._statistical_imputation(df_clean, strategy)
            
        elif strategy == "knn":
            # KNN imputation
            df_clean = self._knn_imputation(df_clean)
            
        elif strategy == "iterative":
            # Iterative imputation (MICE)
            df_clean = self._iterative_imputation(df_clean)
            
        else:
            raise ValueError(f"Unknown imputation strategy: {strategy}")
        
        return df_clean
    
    def _statistical_imputation(
        self,
        df: pd.DataFrame,
        strategy: str,
    ) -> pd.DataFrame:
        """Perform statistical imputation (mean, median, mode).
        
        Args:
            df: Input DataFrame.
            strategy: Statistical method to use.
            
        Returns:
            DataFrame with imputed values.
        """
        df_imputed = df.copy()
        
        for col in df_imputed.columns:
            if df_imputed[col].isnull().any():
                if np.issubdtype(df_imputed[col].dtype, np.number):
                    if strategy == "mean":
                        fill_value = df_imputed[col].mean()
                    elif strategy == "median":
                        fill_value = df_imputed[col].median()
                    else:  # mode
                        fill_value = df_imputed[col].mode()[0]
                else:
                    # For non-numeric columns, use mode
                    fill_value = df_imputed[col].mode()[0]
                
                df_imputed[col] = df_imputed[col].fillna(fill_value)
                logger.debug(f"Filled {col} with {strategy}: {fill_value}")
        
        return df_imputed
    
    def _knn_imputation(
        self,
        df: pd.DataFrame,
        n_neighbors: int = 5,
    ) -> pd.DataFrame:
        """Perform K-Nearest Neighbors imputation.
        
        Args:
            df: Input DataFrame.
            n_neighbors: Number of neighbors to use.
            
        Returns:
            DataFrame with imputed values.
        """
        from sklearn.impute import KNNImputer
        
        # Separate numeric and non-numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns
        
        # Impute numeric columns
        if len(numeric_cols) > 0:
            imputer = KNNImputer(n_neighbors=n_neighbors)
            df_numeric = pd.DataFrame(
                imputer.fit_transform(df[numeric_cols]),
                columns=numeric_cols,
                index=df.index,
            )
        else:
            df_numeric = pd.DataFrame(index=df.index)
        
        # Handle non-numeric columns with mode
        df_non_numeric = df[non_numeric_cols].copy() if len(non_numeric_cols) > 0 else pd.DataFrame(index=df.index)
        for col in non_numeric_cols:
            if df_non_numeric[col].isnull().any():
                mode_val = df_non_numeric[col].mode()[0]
                df_non_numeric[col] = df_non_numeric[col].fillna(mode_val)
        
        # Combine
        return pd.concat([df_numeric, df_non_numeric], axis=1)[df.columns]
    
    def _iterative_imputation(
        self,
        df: pd.DataFrame,
        max_iter: int = 10,
    ) -> pd.DataFrame:
        """Perform iterative imputation (MICE-like).
        
        Args:
            df: Input DataFrame.
            max_iter: Maximum number of iterations.
            
        Returns:
            DataFrame with imputed values.
        """
        from sklearn.experimental import enable_iterative_imputer
        from sklearn.impute import IterativeImputer
        
        # Separate numeric and non-numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns
        
        # Impute numeric columns
        if len(numeric_cols) > 0:
            imputer = IterativeImputer(max_iter=max_iter, random_state=42)
            df_numeric = pd.DataFrame(
                imputer.fit_transform(df[numeric_cols]),
                columns=numeric_cols,
                index=df.index,
            )
        else:
            df_numeric = pd.DataFrame(index=df.index)
        
        # Handle non-numeric columns with mode
        df_non_numeric = df[non_numeric_cols].copy() if len(non_numeric_cols) > 0 else pd.DataFrame(index=df.index)
        for col in non_numeric_cols:
            if df_non_numeric[col].isnull().any():
                mode_val = df_non_numeric[col].mode()[0]
                df_non_numeric[col] = df_non_numeric[col].fillna(mode_val)
        
        # Combine
        return pd.concat([df_numeric, df_non_numeric], axis=1)[df.columns]
    
    def handle_outliers(
        self,
        df: pd.DataFrame,
        method: Optional[str] = None,
        threshold: Optional[float] = None,
        action: str = "clip",
    ) -> pd.DataFrame:
        """Detect and handle outliers in the data.
        
        Args:
            df: Input DataFrame.
            method: Detection method ('zscore', 'iqr', 'isolation', 'percentile').
            threshold: Threshold for outlier detection.
            action: Action to take ('clip', 'remove', 'mark').
            
        Returns:
            DataFrame with outliers handled.
        """
        method = method or self.config["outlier_method"]
        threshold = threshold or self.config["outlier_threshold"]
        
        df_clean = df.copy()
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        
        logger.info(f"Detecting outliers using '{method}' method")
        
        for col in numeric_cols:
            outliers_mask = self._detect_outliers(df_clean[col], method, threshold)
            
            if outliers_mask.sum() > 0:
                logger.info(f"Found {outliers_mask.sum()} outliers in column '{col}'")
                
                if action == "clip":
                    df_clean = self._clip_outliers(df_clean, col, method, threshold)
                elif action == "remove":
                    df_clean = df_clean[~outliers_mask]
                elif action == "mark":
                    df_clean[f"{col}_is_outlier"] = outliers_mask
        
        return df_clean
    
    def _detect_outliers(
        self,
        series: pd.Series,
        method: str,
        threshold: float,
    ) -> pd.Series:
        """Detect outliers in a series.
        
        Args:
            series: Input series.
            method: Detection method.
            threshold: Detection threshold.
            
        Returns:
            Boolean mask where True indicates outlier.
        """
        if method == "zscore":
            z_scores = np.abs(stats.zscore(series, nan_policy="omit"))
            return z_scores > threshold
        
        elif method == "iqr":
            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - threshold * IQR
            upper = Q3 + threshold * IQR
            return (series < lower) | (series > upper)
        
        elif method == "isolation":
            from sklearn.ensemble import IsolationForest
            
            # Reshape for sklearn
            data = series.values.reshape(-1, 1)
            
            # Fit isolation forest
            iso = IsolationForest(contamination=0.1, random_state=42)
            predictions = iso.fit_predict(data)
            
            return predictions == -1
        
        elif method == "percentile":
            lower = series.quantile(0.01)
            upper = series.quantile(0.99)
            return (series < lower) | (series > upper)
        
        else:
            raise ValueError(f"Unknown outlier detection method: {method}")
    
    def _clip_outliers(
        self,
        df: pd.DataFrame,
        col: str,
        method: str,
        threshold: float,
    ) -> pd.DataFrame:
        """Clip outliers to threshold boundaries.
        
        Args:
            df: Input DataFrame.
            col: Column name.
            method: Detection method.
            threshold: Detection threshold.
            
        Returns:
            DataFrame with outliers clipped.
        """
        df_clipped = df.copy()
        
        if method == "iqr":
            Q1 = df_clipped[col].quantile(0.25)
            Q3 = df_clipped[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - threshold * IQR
            upper = Q3 + threshold * IQR
        else:
            lower = df_clipped[col].quantile(0.01)
            upper = df_clipped[col].quantile(0.99)
        
        df_clipped[col] = df_clipped[col].clip(lower=lower, upper=upper)
        
        return df_clipped
    
    def validate_physical_consistency(
        self,
        df: pd.DataFrame,
        remove_violations: bool = False,
    ) -> pd.DataFrame:
        """Validate physical consistency of perovskite solar cell parameters.
        
        Checks that values are within physically reasonable ranges and
        that derived relationships hold (e.g., PCE = Voc × Jsc × FF / Pin).
        
        Args:
            df: Input DataFrame.
            remove_violations: Whether to remove rows with violations.
            
        Returns:
            DataFrame with physical consistency validated.
        """
        df_validated = df.copy()
        violations = pd.DataFrame(index=df.index)
        
        # Check individual parameter ranges
        for param, constraints in self.PHYSICAL_CONSTRAINTS.items():
            if param in df_validated.columns:
                # Check absolute bounds
                below_min = df_validated[param] < constraints["min"]
                above_max = df_validated[param] > constraints["max"]
                
                if below_min.any():
                    logger.warning(
                        f"{below_min.sum()} values of '{param}' below physical minimum ({constraints['min']})"
                    )
                    violations[f"{param}_below_min"] = below_min
                
                if above_max.any():
                    logger.warning(
                        f"{above_max.sum()} values of '{param}' above physical maximum ({constraints['max']})"
                    )
                    violations[f"{param}_above_max"] = above_max
                
                # Warn about unusual values
                unusual = df_validated[param] > constraints["typical_max"]
                if unusual.any():
                    logger.warning(
                        f"{unusual.sum()} values of '{param}' above typical maximum ({constraints['typical_max']})"
                    )
        
        # Check derived relationships
        df_validated = self._check_derived_relationships(df_validated, violations)
        
        # Remove violations if requested
        if remove_violations:
            violation_mask = violations.any(axis=1)
            if violation_mask.any():
                logger.info(f"Removing {violation_mask.sum()} rows with physical violations")
                df_validated = df_validated[~violation_mask]
        
        return df_validated
    
    def _check_derived_relationships(
        self,
        df: pd.DataFrame,
        violations: pd.DataFrame,
    ) -> pd.DataFrame:
        """Check derived physical relationships.
        
        Args:
            df: Input DataFrame.
            violations: DataFrame to track violations.
            
        Returns:
            DataFrame with relationship checks applied.
        """
        # Check: PCE ≈ Voc × Jsc × FF / Pin (assuming Pin = 100 mW/cm²)
        if all(col in df.columns for col in ["efficiency", "voc", "jsc", "ff"]):
            # Calculate theoretical efficiency
            # PCE (%) = Voc (V) × Jsc (mA/cm²) × FF (%) / Pin (mW/cm²) × 100
            # Assuming standard test conditions: Pin = 100 mW/cm²
            theoretical_eff = (df["voc"] * df["jsc"] * df["ff"]) / 100
            
            # Allow 10% relative error
            actual_eff = df["efficiency"]
            relative_error = abs(actual_eff - theoretical_eff) / theoretical_eff
            
            inconsistent = relative_error > 0.1
            if inconsistent.any():
                logger.warning(
                    f"{inconsistent.sum()} rows have inconsistent PCE relationship "
                    f"(>10% error between actual and theoretical)"
                )
                violations["pce_inconsistent"] = inconsistent
                
                # Optionally correct the values (use theoretical)
                df["efficiency_theoretical"] = theoretical_eff
                df["pce_relative_error"] = relative_error
        
        # Check: Bandgap-Voc relationship (Voc ≤ Eg/q - 0.3V typical loss)
        if all(col in df.columns for col in ["voc", "bandgap"]):
            # Maximum possible Voc ≈ Eg - 0.3V (thermodynamic loss)
            max_voc = df["bandgap"] - 0.3
            exceeds_max = df["voc"] > max_voc
            
            if exceeds_max.any():
                logger.warning(
                    f"{exceeds_max.sum()} rows have Voc exceeding theoretical maximum "
                    f"(Voc > Eg - 0.3V)"
                )
                violations["voc_exceeds_theoretical"] = exceeds_max
        
        return df
    
    def normalize(
        self,
        df: pd.DataFrame,
        method: str = "standard",
        columns: Optional[List[str]] = None,
        fit: bool = True,
    ) -> pd.DataFrame:
        """Normalize/standardize numeric columns.
        
        Args:
            df: Input DataFrame.
            method: Normalization method ('standard', 'minmax', 'robust', 'log').
            columns: Columns to normalize. If None, all numeric columns.
            fit: Whether to fit the normalizer (True for training, False for inference).
            
        Returns:
            Normalized DataFrame.
        """
        df_normalized = df.copy()
        
        # Select columns to normalize
        if columns is None:
            columns = df_normalized.select_dtypes(include=[np.number]).columns.tolist()
        
        if not columns:
            logger.warning("No numeric columns to normalize")
            return df_normalized
        
        logger.info(f"Normalizing {len(columns)} columns using '{method}' method")
        
        for col in columns:
            if col not in df_normalized.columns:
                continue
                
            if method == "standard":
                df_normalized = self._standard_normalize(df_normalized, col, fit)
            elif method == "minmax":
                df_normalized = self._minmax_normalize(df_normalized, col, fit)
            elif method == "robust":
                df_normalized = self._robust_normalize(df_normalized, col, fit)
            elif method == "log":
                df_normalized = self._log_normalize(df_normalized, col)
            else:
                raise ValueError(f"Unknown normalization method: {method}")
        
        return df_normalized
    
    def _standard_normalize(
        self,
        df: pd.DataFrame,
        col: str,
        fit: bool,
    ) -> pd.DataFrame:
        """Apply standard (z-score) normalization.
        
        Args:
            df: Input DataFrame.
            col: Column name.
            fit: Whether to fit parameters.
            
        Returns:
            DataFrame with normalized column.
        """
        df_norm = df.copy()
        
        if fit:
            self._fitted_params[col] = {
                "method": "standard",
                "mean": df_norm[col].mean(),
                "std": df_norm[col].std(),
            }
        
        params = self._fitted_params.get(col, {})
        if "mean" in params and "std" in params:
            if params["std"] > 0:
                df_norm[col] = (df_norm[col] - params["mean"]) / params["std"]
        
        return df_norm
    
    def _minmax_normalize(
        self,
        df: pd.DataFrame,
        col: str,
        fit: bool,
    ) -> pd.DataFrame:
        """Apply min-max normalization.
        
        Args:
            df: Input DataFrame.
            col: Column name.
            fit: Whether to fit parameters.
            
        Returns:
            DataFrame with normalized column.
        """
        df_norm = df.copy()
        
        if fit:
            self._fitted_params[col] = {
                "method": "minmax",
                "min": df_norm[col].min(),
                "max": df_norm[col].max(),
            }
        
        params = self._fitted_params.get(col, {})
        if "min" in params and "max" in params:
            range_val = params["max"] - params["min"]
            if range_val > 0:
                df_norm[col] = (df_norm[col] - params["min"]) / range_val
        
        return df_norm
    
    def _robust_normalize(
        self,
        df: pd.DataFrame,
        col: str,
        fit: bool,
    ) -> pd.DataFrame:
        """Apply robust normalization (using median and IQR).
        
        Args:
            df: Input DataFrame.
            col: Column name.
            fit: Whether to fit parameters.
            
        Returns:
            DataFrame with normalized column.
        """
        df_norm = df.copy()
        
        if fit:
            self._fitted_params[col] = {
                "method": "robust",
                "median": df_norm[col].median(),
                "iqr": df_norm[col].quantile(0.75) - df_norm[col].quantile(0.25),
            }
        
        params = self._fitted_params.get(col, {})
        if "median" in params and "iqr" in params:
            if params["iqr"] > 0:
                df_norm[col] = (df_norm[col] - params["median"]) / params["iqr"]
        
        return df_norm
    
    def _log_normalize(
        self,
        df: pd.DataFrame,
        col: str,
    ) -> pd.DataFrame:
        """Apply log normalization.
        
        Args:
            df: Input DataFrame.
            col: Column name.
            
        Returns:
            DataFrame with log-normalized column.
        """
        df_norm = df.copy()
        
        # Handle negative values
        min_val = df_norm[col].min()
        if min_val <= 0:
            df_norm[col] = np.log1p(df_norm[col] - min_val + 1)
        else:
            df_norm[col] = np.log1p(df_norm[col])
        
        self._fitted_params[col] = {
            "method": "log",
            "min_shift": min_val if min_val <= 0 else 0,
        }
        
        return df_norm
    
    def inverse_normalize(
        self,
        df: pd.DataFrame,
        columns: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Reverse normalization to get original scale values.
        
        Args:
            df: Normalized DataFrame.
            columns: Columns to inverse normalize. If None, all fitted columns.
            
        Returns:
            DataFrame with original scale values.
        """
        df_orig = df.copy()
        
        if columns is None:
            columns = list(self._fitted_params.keys())
        
        for col in columns:
            if col not in self._fitted_params:
                continue
            
            params = self._fitted_params[col]
            method = params.get("method", "standard")
            
            if method == "standard":
                df_orig[col] = df_orig[col] * params["std"] + params["mean"]
            elif method == "minmax":
                df_orig[col] = df_orig[col] * (params["max"] - params["min"]) + params["min"]
            elif method == "robust":
                df_orig[col] = df_orig[col] * params["iqr"] + params["median"]
            elif method == "log":
                df_orig[col] = np.expm1(df_orig[col]) + params.get("min_shift", 0)
        
        return df_orig
    
    def get_fitted_params(self) -> Dict[str, Dict]:
        """Get fitted normalization parameters.
        
        Returns:
            Dictionary of fitted parameters for each column.
        """
        return self._fitted_params.copy()
    
    def set_fitted_params(self, params: Dict[str, Dict]) -> None:
        """Set fitted normalization parameters.
        
        Args:
            params: Dictionary of fitted parameters.
        """
        self._fitted_params = params.copy()
