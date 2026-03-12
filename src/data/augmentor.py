"""Physics-informed data augmentation for perovskite solar cells.

Implements augmentation strategies based on semiconductor physics
to generate realistic synthetic data while respecting physical constraints.
"""

import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import interpolate
from scipy.stats import truncnorm

logger = logging.getLogger(__name__)


class PhysicsInformedAugmentor:
    """Augment perovskite solar cell data using physics-based constraints.
    
    Generates synthetic data that respects physical relationships
    derived from semiconductor physics and device principles.
    
    Attributes:
        random_state: Random seed for reproducibility.
        
    Example:
        >>> augmentor = PhysicsInformedAugmentor(random_state=42)
        >>> df_augmented = augmentor.augment(df, n_samples=1000)
        >>> df_interpolated = augmentor.interpolate(df, method="physics")
    """
    
    # Physical constants
    K_BOLTZMANN = 8.617e-5  # Boltzmann constant (eV/K)
    q = 1.602e-19  # Elementary charge (C)
    
    # Typical parameter ranges for perovskites
    TYPICAL_RANGES = {
        "efficiency": {"min": 5, "max": 28, "mean": 18, "std": 5},
        "voc": {"min": 0.8, "max": 1.3, "mean": 1.1, "std": 0.1},
        "jsc": {"min": 15, "max": 26, "mean": 22, "std": 2},
        "ff": {"min": 60, "max": 90, "mean": 78, "std": 6},
        "bandgap": {"min": 1.2, "max": 2.3, "mean": 1.55, "std": 0.2},
        "thickness": {"min": 100, "max": 1000, "mean": 500, "std": 150},
    }
    
    # Correlation structures based on physics
    # Positive correlations: efficiency↔voc, efficiency↔jsc, efficiency↔ff
    # Negative correlations: bandgap↔jsc (higher bandgap → lower Jsc)
    PHYSICAL_CORRELATIONS = {
        ("efficiency", "voc"): 0.7,
        ("efficiency", "jsc"): 0.6,
        ("efficiency", "ff"): 0.5,
        ("efficiency", "bandgap"): -0.2,  # Optimal bandgap around 1.3-1.5 eV
        ("voc", "bandgap"): 0.8,  # Higher bandgap → higher Voc
        ("jsc", "bandgap"): -0.6,  # Higher bandgap → lower Jsc
        ("voc", "ff"): 0.4,
        ("jsc", "ff"): 0.3,
    }
    
    def __init__(
        self,
        random_state: Optional[int] = None,
        preserve_correlations: bool = True,
    ):
        """Initialize the augmentor.
        
        Args:
            random_state: Random seed for reproducibility.
            preserve_correlations: Whether to preserve physical correlations.
        """
        self.random_state = random_state
        self.preserve_correlations = preserve_correlations
        
        if random_state is not None:
            np.random.seed(random_state)
    
    def augment(
        self,
        df: pd.DataFrame,
        n_samples: int = 1000,
        method: str = "gaussian_mixture",
        noise_level: float = 0.05,
    ) -> pd.DataFrame:
        """Augment dataset with synthetic samples.
        
        Args:
            df: Input DataFrame.
            n_samples: Number of synthetic samples to generate.
            method: Augmentation method ('gaussian_mixture', 'smote', 'interpolation').
            noise_level: Amount of noise to add (0-1).
            
        Returns:
            Augmented DataFrame containing original + synthetic samples.
        """
        logger.info(f"Augmenting data with {n_samples} samples using '{method}' method")
        
        if method == "gaussian_mixture":
            synthetic = self._generate_gaussian_mixture(df, n_samples)
        elif method == "smote":
            synthetic = self._generate_smote(df, n_samples)
        elif method == "interpolation":
            synthetic = self._generate_interpolated(df, n_samples)
        else:
            raise ValueError(f"Unknown augmentation method: {method}")
        
        # Add noise if specified
        if noise_level > 0:
            synthetic = self._add_noise(synthetic, noise_level)
        
        # Validate physical consistency
        synthetic = self._enforce_physical_constraints(synthetic)
        
        # Combine original and synthetic
        df_augmented = pd.concat([df, synthetic], ignore_index=True)
        
        logger.info(f"Augmented dataset shape: {df_augmented.shape}")
        
        return df_augmented
    
    def _generate_gaussian_mixture(
        self,
        df: pd.DataFrame,
        n_samples: int,
    ) -> pd.DataFrame:
        """Generate samples using Gaussian Mixture Model.
        
        Args:
            df: Input DataFrame.
            n_samples: Number of samples to generate.
            
        Returns:
            Synthetic DataFrame.
        """
        from sklearn.mixture import GaussianMixture
        
        # Select numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df_numeric = df[numeric_cols].dropna()
        
        # Fit GMM
        n_components = min(5, max(1, len(df_numeric) // 50))
        gmm = GaussianMixture(
            n_components=n_components,
            covariance_type="full",
            random_state=self.random_state,
        )
        gmm.fit(df_numeric)
        
        # Generate samples
        samples, _ = gmm.sample(n_samples)
        
        # Create DataFrame
        synthetic = pd.DataFrame(samples, columns=numeric_cols)
        
        return synthetic
    
    def _generate_smote(
        self,
        df: pd.DataFrame,
        n_samples: int,
    ) -> pd.DataFrame:
        """Generate samples using SMOTE-like approach with physical constraints.
        
        Args:
            df: Input DataFrame.
            n_samples: Number of samples to generate.
            
        Returns:
            Synthetic DataFrame.
        """
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df_numeric = df[numeric_cols].dropna()
        
        synthetic_samples = []
        
        for _ in range(n_samples):
            # Select a random sample
            idx = np.random.randint(0, len(df_numeric))
            sample = df_numeric.iloc[idx].copy()
            
            # Find k nearest neighbors
            k = min(5, len(df_numeric) - 1)
            distances = np.sqrt(((df_numeric - sample) ** 2).sum(axis=1))
            distances[idx] = np.inf  # Exclude self
            neighbor_indices = distances.nsmallest(k).index
            
            # Select a random neighbor
            neighbor_idx = np.random.choice(neighbor_indices)
            neighbor = df_numeric.loc[neighbor_idx]
            
            # Interpolate with random weight
            alpha = np.random.uniform(0, 1)
            new_sample = sample + alpha * (neighbor - sample)
            
            synthetic_samples.append(new_sample)
        
        synthetic = pd.DataFrame(synthetic_samples, columns=numeric_cols)
        
        return synthetic
    
    def _generate_interpolated(
        self,
        df: pd.DataFrame,
        n_samples: int,
    ) -> pd.DataFrame:
        """Generate samples through physics-informed interpolation.
        
        Args:
            df: Input DataFrame.
            n_samples: Number of samples to generate.
            
        Returns:
            Synthetic DataFrame.
        """
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df_numeric = df[numeric_cols].dropna()
        
        synthetic_samples = []
        
        for _ in range(n_samples):
            # Select two random samples
            idx1, idx2 = np.random.choice(len(df_numeric), 2, replace=False)
            sample1 = df_numeric.iloc[idx1]
            sample2 = df_numeric.iloc[idx2]
            
            # Interpolate with random weight
            alpha = np.random.uniform(0, 1)
            new_sample = self._physics_aware_interpolation(sample1, sample2, alpha)
            
            synthetic_samples.append(new_sample)
        
        synthetic = pd.DataFrame(synthetic_samples, columns=numeric_cols)
        
        return synthetic
    
    def _physics_aware_interpolation(
        self,
        sample1: pd.Series,
        sample2: pd.Series,
        alpha: float,
    ) -> pd.Series:
        """Perform physics-aware interpolation between two samples.
        
        Args:
            sample1: First sample.
            sample2: Second sample.
            alpha: Interpolation weight (0-1).
            
        Returns:
            Interpolated sample respecting physical constraints.
        """
        # Start with linear interpolation
        interpolated = sample1 + alpha * (sample2 - sample1)
        
        # If preserving correlations, adjust based on physics
        if self.preserve_correlations:
            # Ensure efficiency = voc × jsc × ff / 100 (approximate)
            if all(col in interpolated.index for col in ["efficiency", "voc", "jsc", "ff"]):
                theoretical_eff = (interpolated["voc"] * interpolated["jsc"] * interpolated["ff"]) / 100
                
                # Blend actual with theoretical
                interpolated["efficiency"] = (
                    0.7 * interpolated["efficiency"] + 
                    0.3 * theoretical_eff
                )
            
            # Ensure Voc is consistent with bandgap
            if all(col in interpolated.index for col in ["voc", "bandgap"]):
                # Voc should be roughly Eg - 0.4V (with losses)
                max_voc = interpolated["bandgap"] - 0.3
                if interpolated["voc"] > max_voc:
                    interpolated["voc"] = max_voc * np.random.uniform(0.85, 0.95)
        
        return interpolated
    
    def interpolate(
        self,
        df: pd.DataFrame,
        n_points: int = 100,
        method: str = "physics",
        target_col: Optional[str] = None,
    ) -> pd.DataFrame:
        """Interpolate data using physics-based methods.
        
        Args:
            df: Input DataFrame.
            n_points: Number of points to interpolate.
            method: Interpolation method ('physics', 'spline', 'linear').
            target_col: Target column for interpolation (if None, interpolate all).
            
        Returns:
            Interpolated DataFrame.
        """
        logger.info(f"Interpolating data with {n_points} points using '{method}' method")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if target_col and target_col in numeric_cols:
            # Interpolate based on target column
            return self._interpolate_by_target(df, target_col, n_points, method)
        else:
            # General interpolation
            return self._general_interpolation(df, n_points, method)
    
    def _interpolate_by_target(
        self,
        df: pd.DataFrame,
        target_col: str,
        n_points: int,
        method: str,
    ) -> pd.DataFrame:
        """Interpolate data based on a target column.
        
        Args:
            df: Input DataFrame.
            target_col: Target column name.
            n_points: Number of points to interpolate.
            method: Interpolation method.
            
        Returns:
            Interpolated DataFrame.
        """
        df_sorted = df.sort_values(target_col).reset_index(drop=True)
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Create new target values
        target_min = df_sorted[target_col].min()
        target_max = df_sorted[target_col].max()
        new_target = np.linspace(target_min, target_max, n_points)
        
        interpolated_data = {target_col: new_target}
        
        for col in numeric_cols:
            if col == target_col:
                continue
            
            if method == "spline":
                # Cubic spline interpolation
                try:
                    spline = interpolate.CubicSpline(
                        df_sorted[target_col],
                        df_sorted[col],
                    )
                    interpolated_data[col] = spline(new_target)
                except Exception as e:
                    logger.warning(f"Spline interpolation failed for {col}: {e}")
                    # Fall back to linear
                    interp_func = np.interp
                    interpolated_data[col] = interp_func(
                        new_target,
                        df_sorted[target_col],
                        df_sorted[col],
                    )
            elif method == "physics":
                # Physics-aware interpolation
                interpolated_data[col] = self._physics_interpolate_column(
                    df_sorted, target_col, col, new_target
                )
            else:  # linear
                interpolated_data[col] = np.interp(
                    new_target,
                    df_sorted[target_col],
                    df_sorted[col],
                )
        
        return pd.DataFrame(interpolated_data)
    
    def _physics_interpolate_column(
        self,
        df: pd.DataFrame,
        x_col: str,
        y_col: str,
        new_x: np.ndarray,
    ) -> np.ndarray:
        """Perform physics-aware interpolation for a column.
        
        Args:
            df: Input DataFrame.
            x_col: X column name.
            y_col: Y column name.
            new_x: New X values.
            
        Returns:
            Interpolated Y values.
        """
        # Use linear interpolation as base
        y_interp = np.interp(new_x, df[x_col], df[y_col])
        
        # Apply physics-based corrections for known relationships
        if x_col == "bandgap" and y_col == "voc":
            # Voc ≈ Eg - 0.4V (with thermal losses)
            theoretical_voc = new_x - 0.4
            y_interp = np.maximum(y_interp, theoretical_voc * 0.8)
            y_interp = np.minimum(y_interp, theoretical_voc * 0.95)
        
        elif x_col == "bandgap" and y_col == "jsc":
            # Jsc decreases with bandgap (fewer photons absorbed)
            # Approximate relationship
            if "voc" in df.columns:
                # Use correlation with Voc for better estimate
                y_interp = y_interp * (1 - 0.1 * (new_x - df[x_col].mean()) / df[x_col].std())
        
        return y_interp
    
    def _general_interpolation(
        self,
        df: pd.DataFrame,
        n_points: int,
        method: str,
    ) -> pd.DataFrame:
        """Perform general interpolation across all dimensions.
        
        Args:
            df: Input DataFrame.
            n_points: Number of points to interpolate.
            method: Interpolation method.
            
        Returns:
            Interpolated DataFrame.
        """
        # Use multivariate interpolation
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df_numeric = df[numeric_cols].dropna()
        
        # Sample points for interpolation
        indices = np.linspace(0, len(df_numeric) - 1, n_points).astype(int)
        
        if method == "spline":
            # Use spline interpolation for each column
            interpolated = pd.DataFrame()
            for col in numeric_cols:
                spline = interpolate.CubicSpline(
                    np.arange(len(df_numeric)),
                    df_numeric[col],
                )
                interpolated[col] = spline(indices)
        else:
            # Simple linear interpolation
            interpolated = df_numeric.iloc[indices].reset_index(drop=True)
        
        return interpolated
    
    def generate_synthetic_dataset(
        self,
        n_samples: int = 1000,
        composition: Optional[str] = None,
        include_derived: bool = True,
    ) -> pd.DataFrame:
        """Generate a purely synthetic dataset based on physical principles.
        
        Args:
            n_samples: Number of samples to generate.
            composition: Chemical composition hint (e.g., "MAPbI3", "FAPbBr3").
            include_derived: Whether to include derived columns.
            
        Returns:
            Synthetic DataFrame.
        """
        logger.info(f"Generating {n_samples} synthetic samples")
        
        # Adjust ranges based on composition if provided
        ranges = self._get_composition_ranges(composition)
        
        # Generate correlated samples
        if self.preserve_correlations:
            df = self._generate_correlated_samples(n_samples, ranges)
        else:
            df = self._generate_independent_samples(n_samples, ranges)
        
        # Add derived columns
        if include_derived:
            df = self._add_derived_columns(df)
        
        # Enforce physical constraints
        df = self._enforce_physical_constraints(df)
        
        return df
    
    def _get_composition_ranges(
        self,
        composition: Optional[str],
    ) -> Dict[str, Dict]:
        """Get parameter ranges for a specific composition.
        
        Args:
            composition: Chemical composition.
            
        Returns:
            Dictionary of parameter ranges.
        """
        ranges = self.TYPICAL_RANGES.copy()
        
        if composition is None:
            return ranges
        
        composition = composition.upper()
        
        # Adjust ranges based on composition
        if "MAPBI3" in composition or "MA" in composition:
            # Methylammonium lead iodide
            ranges["bandgap"]["mean"] = 1.55
            ranges["bandgap"]["std"] = 0.05
            ranges["voc"]["mean"] = 1.05
            ranges["efficiency"]["max"] = 25
            
        elif "FAPBI3" in composition or "FA" in composition:
            # Formamidinium lead iodide
            ranges["bandgap"]["mean"] = 1.48
            ranges["bandgap"]["std"] = 0.05
            ranges["voc"]["mean"] = 1.10
            ranges["efficiency"]["max"] = 26
            
        elif "CS" in composition:
            # Cesium-based
            ranges["bandgap"]["mean"] = 1.73
            ranges["bandgap"]["std"] = 0.05
            ranges["voc"]["mean"] = 1.15
            ranges["efficiency"]["max"] = 21
            
        elif "BR" in composition:
            # Bromine-based (higher bandgap)
            ranges["bandgap"]["mean"] = 2.0
            ranges["bandgap"]["std"] = 0.1
            ranges["voc"]["mean"] = 1.35
            ranges["jsc"]["mean"] = 10
            ranges["efficiency"]["max"] = 15
        
        return ranges
    
    def _generate_correlated_samples(
        self,
        n_samples: int,
        ranges: Dict[str, Dict],
    ) -> pd.DataFrame:
        """Generate correlated samples respecting physical relationships.
        
        Args:
            n_samples: Number of samples.
            ranges: Parameter ranges.
            
        Returns:
            DataFrame with correlated samples.
        """
        # Build correlation matrix
        params = list(ranges.keys())
        n_params = len(params)
        
        corr_matrix = np.eye(n_params)
        for i, param1 in enumerate(params):
            for j, param2 in enumerate(params):
                if i < j:
                    key = (param1, param2) if param1 < param2 else (param2, param1)
                    if key in self.PHYSICAL_CORRELATIONS:
                        corr = self.PHYSICAL_CORRELATIONS[key]
                        corr_matrix[i, j] = corr
                        corr_matrix[j, i] = corr
        
        # Generate correlated Gaussian samples
        means = np.array([ranges[p]["mean"] for p in params])
        stds = np.array([ranges[p]["std"] for p in params])
        
        # Create covariance matrix
        cov_matrix = np.outer(stds, stds) * corr_matrix
        
        # Generate samples
        samples = np.random.multivariate_normal(means, cov_matrix, n_samples)
        
        # Create DataFrame
        df = pd.DataFrame(samples, columns=params)
        
        # Clip to valid ranges
        for param in params:
            df[param] = df[param].clip(
                lower=ranges[param]["min"],
                upper=ranges[param]["max"],
            )
        
        return df
    
    def _generate_independent_samples(
        self,
        n_samples: int,
        ranges: Dict[str, Dict],
    ) -> pd.DataFrame:
        """Generate independent samples (no correlation preservation).
        
        Args:
            n_samples: Number of samples.
            ranges: Parameter ranges.
            
        Returns:
            DataFrame with independent samples.
        """
        data = {}
        
        for param, params in ranges.items():
            # Use truncated normal distribution
            lower = params["min"]
            upper = params["max"]
            mu = params["mean"]
            sigma = params["std"]
            
            # Generate truncated normal samples
            a = (lower - mu) / sigma
            b = (upper - mu) / sigma
            samples = truncnorm.rvs(a, b, loc=mu, scale=sigma, size=n_samples)
            
            data[param] = samples
        
        return pd.DataFrame(data)
    
    def _add_derived_columns(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Add derived columns based on physical relationships.
        
        Args:
            df: Input DataFrame.
            
        Returns:
            DataFrame with derived columns.
        """
        df_derived = df.copy()
        
        # Calculate theoretical efficiency if not present
        if "efficiency" not in df_derived.columns and all(
            col in df_derived.columns for col in ["voc", "jsc", "ff"]
        ):
            df_derived["efficiency_theoretical"] = (
                df_derived["voc"] * df_derived["jsc"] * df_derived["ff"]
            ) / 100
        
        # Add ideality factor estimate (from Voc vs bandgap)
        if all(col in df_derived.columns for col in ["voc", "bandgap"]):
            # n ≈ qVoc / (kT * ln(Jsc/J0))
            # Simplified estimate
            df_derived["ideality_factor_estimate"] = (
                df_derived["voc"] / (self.K_BOLTZMANN * 300 * np.log(1e6))
            )
        
        # Add defect density estimate (from Voc loss)
        if all(col in df_derived.columns for col in ["voc", "bandgap"]):
            # Voc loss = Eg/q - Voc
            df_derived["voc_loss"] = df_derived["bandgap"] - df_derived["voc"]
        
        return df_derived
    
    def _enforce_physical_constraints(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Enforce physical constraints on the data.
        
        Args:
            df: Input DataFrame.
            
        Returns:
            DataFrame with enforced constraints.
        """
        df_constrained = df.copy()
        
        # Ensure positive values for key parameters
        for col in ["efficiency", "voc", "jsc", "ff", "bandgap", "thickness"]:
            if col in df_constrained.columns:
                df_constrained[col] = np.maximum(df_constrained[col], 0)
        
        # Ensure efficiency doesn't exceed 100%
        if "efficiency" in df_constrained.columns:
            df_constrained["efficiency"] = np.minimum(df_constrained["efficiency"], 100)
        
        # Ensure FF doesn't exceed 100%
        if "ff" in df_constrained.columns:
            df_constrained["ff"] = np.minimum(df_constrained["ff"], 100)
        
        # Ensure Voc is less than bandgap (thermodynamic limit)
        if "voc" in df_constrained.columns and "bandgap" in df_constrained.columns:
            max_voc = df_constrained["bandgap"] * 0.95  # Account for losses
            df_constrained["voc"] = np.minimum(df_constrained["voc"], max_voc)
        
        # Ensure consistency between efficiency, Voc, Jsc, and FF
        if all(col in df_constrained.columns for col in ["efficiency", "voc", "jsc", "ff"]):
            theoretical_eff = (
                df_constrained["voc"] * df_constrained["jsc"] * df_constrained["ff"]
            ) / 100
            
            # If efficiency is way off, adjust it
            ratio = df_constrained["efficiency"] / (theoretical_eff + 1e-6)
            unreasonable = (ratio < 0.5) | (ratio > 2.0)
            
            if unreasonable.any():
                logger.warning(f"Adjusting {unreasonable.sum()} samples with unreasonable efficiency")
                df_constrained.loc[unreasonable, "efficiency"] = (
                    0.8 * theoretical_eff[unreasonable] + 
                    0.2 * df_constrained.loc[unreasonable, "efficiency"]
                )
        
        return df_constrained
    
    def add_noise(
        self,
        df: pd.DataFrame,
        noise_level: float = 0.05,
        noise_type: str = "gaussian",
    ) -> pd.DataFrame:
        """Add realistic measurement noise to the data.
        
        Args:
            df: Input DataFrame.
            noise_level: Relative noise level (0-1).
            noise_type: Type of noise ('gaussian', 'uniform', 'poisson').
            
        Returns:
            DataFrame with added noise.
        """
        return self._add_noise(df, noise_level, noise_type)
    
    def _add_noise(
        self,
        df: pd.DataFrame,
        noise_level: float = 0.05,
        noise_type: str = "gaussian",
    ) -> pd.DataFrame:
        """Internal method to add noise.
        
        Args:
            df: Input DataFrame.
            noise_level: Relative noise level.
            noise_type: Type of noise.
            
        Returns:
            DataFrame with added noise.
        """
        df_noisy = df.copy()
        numeric_cols = df_noisy.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            std = df_noisy[col].std()
            
            if noise_type == "gaussian":
                noise = np.random.normal(0, noise_level * std, len(df_noisy))
            elif noise_type == "uniform":
                noise = np.random.uniform(
                    -noise_level * std,
                    noise_level * std,
                    len(df_noisy),
                )
            elif noise_type == "poisson":
                # For count-like data
                noise = np.random.poisson(std * noise_level, len(df_noisy)) - std * noise_level
            else:
                raise ValueError(f"Unknown noise type: {noise_type}")
            
            df_noisy[col] = df_noisy[col] + noise
        
        return df_noisy
    
    def balance_dataset(
        self,
        df: pd.DataFrame,
        target_col: str,
        method: str = "oversample",
        balance_threshold: float = 0.1,
    ) -> pd.DataFrame:
        """Balance dataset by target column.
        
        Args:
            df: Input DataFrame.
            target_col: Target column to balance.
            method: Balancing method ('oversample', 'undersample', 'smote').
            balance_threshold: Threshold for considering dataset imbalanced.
            
        Returns:
            Balanced DataFrame.
        """
        # Discretize target if continuous
        if df[target_col].dtype in [np.float64, np.float32]:
            df["_target_bin"] = pd.qcut(df[target_col], q=5, labels=False, duplicates="drop")
            target_for_balance = "_target_bin"
        else:
            target_for_balance = target_col
        
        # Get class distribution
        class_counts = df[target_for_balance].value_counts()
        max_count = class_counts.max()
        min_count = class_counts.min()
        
        # Check if balancing is needed
        imbalance_ratio = (max_count - min_count) / max_count
        if imbalance_ratio < balance_threshold:
            logger.info("Dataset is already balanced")
            return df
        
        logger.info(f"Balancing dataset (imbalance ratio: {imbalance_ratio:.2f})")
        
        if method == "oversample":
            df_balanced = self._oversample(df, target_for_balance, max_count)
        elif method == "undersample":
            df_balanced = self._undersample(df, target_for_balance, min_count)
        elif method == "smote":
            df_balanced = self._smote_balance(df, target_for_balance, max_count)
        else:
            raise ValueError(f"Unknown balancing method: {method}")
        
        # Clean up temporary column
        if "_target_bin" in df_balanced.columns:
            df_balanced = df_balanced.drop(columns=["_target_bin"])
        
        return df_balanced
    
    def _oversample(
        self,
        df: pd.DataFrame,
        target_col: str,
        target_count: int,
    ) -> pd.DataFrame:
        """Oversample minority classes.
        
        Args:
            df: Input DataFrame.
            target_col: Target column.
            target_count: Target count per class.
            
        Returns:
            Oversampled DataFrame.
        """
        dfs = []
        
        for class_val in df[target_col].unique():
            df_class = df[df[target_col] == class_val]
            
            if len(df_class) < target_count:
                # Oversample with replacement
                n_samples = target_count - len(df_class)
                df_oversampled = df_class.sample(
                    n=n_samples,
                    replace=True,
                    random_state=self.random_state,
                )
                df_class = pd.concat([df_class, df_oversampled])
            
            dfs.append(df_class)
        
        return pd.concat(dfs, ignore_index=True)
    
    def _undersample(
        self,
        df: pd.DataFrame,
        target_col: str,
        target_count: int,
    ) -> pd.DataFrame:
        """Undersample majority classes.
        
        Args:
            df: Input DataFrame.
            target_col: Target column.
            target_count: Target count per class.
            
        Returns:
            Undersampled DataFrame.
        """
        dfs = []
        
        for class_val in df[target_col].unique():
            df_class = df[df[target_col] == class_val]
            
            if len(df_class) > target_count:
                df_class = df_class.sample(
                    n=target_count,
                    random_state=self.random_state,
                )
            
            dfs.append(df_class)
        
        return pd.concat(dfs, ignore_index=True)
    
    def _smote_balance(
        self,
        df: pd.DataFrame,
        target_col: str,
        target_count: int,
    ) -> pd.DataFrame:
        """Balance using SMOTE-like approach.
        
        Args:
            df: Input DataFrame.
            target_col: Target column.
            target_count: Target count per class.
            
        Returns:
            Balanced DataFrame.
        """
        dfs = []
        
        for class_val in df[target_col].unique():
            df_class = df[df[target_col] == class_val]
            
            if len(df_class) < target_count:
                # Generate synthetic samples
                n_synthetic = target_count - len(df_class)
                df_synthetic = self.augment(
                    df_class,
                    n_samples=n_synthetic,
                    method="smote",
                    noise_level=0.02,
                )
                df_class = pd.concat([df_class, df_synthetic.tail(n_synthetic)])
            
            dfs.append(df_class)
        
        return pd.concat(dfs, ignore_index=True)
