"""Data loader for perovskite solar cells datasets.

Supports multiple data sources including CSV, Excel, and APIs.
Includes automatic download from public databases and data validation.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Union
from urllib.parse import urljoin

import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)


class PerovskiteDataLoader:
    """Load and validate perovskite solar cell data from various sources.
    
    Attributes:
        cache_dir: Directory for caching downloaded data.
        timeout: Request timeout in seconds.
        
    Example:
        >>> loader = PerovskiteDataLoader(cache_dir="./data")
        >>> df = loader.load_csv("perovskite_data.csv")
        >>> df = loader.load_from_api("https://api.example.com/perovskite")
    """
    
    # Known public databases for perovskite solar cell data
    PUBLIC_DATABASES = {
        "nrel": "https://developer.nrel.gov/api/pvwatts/v6",
        "perovskitedatabase": "https://www.perovsitedatabase.com/api/v1",
        "materials_project": "https://materialsproject.org/rest/v2",
    }
    
    # Required columns for perovskite solar cell data
    REQUIRED_COLUMNS = [
        "efficiency",  # Power conversion efficiency (%)
        "voc",         # Open-circuit voltage (V)
        "jsc",         # Short-circuit current density (mA/cm²)
        "ff",          # Fill factor (%)
    ]
    
    # Optional but recommended columns
    OPTIONAL_COLUMNS = [
        "bandgap",           # Band gap (eV)
        "thickness",         # Active layer thickness (nm)
        "temperature",       # Measurement temperature (K)
        "light_intensity",   # Light intensity (mW/cm²)
        "active_area",       # Active area (cm²)
        "composition",       # Chemical composition
        "deposition_method", # Deposition method
        "annealing_temp",    # Annealing temperature (°C)
        "annealing_time",    # Annealing time (min)
    ]
    
    def __init__(
        self,
        cache_dir: Optional[Union[str, Path]] = None,
        timeout: int = 30,
        max_retries: int = 3,
    ):
        """Initialize the data loader.
        
        Args:
            cache_dir: Directory for caching downloaded data.
            timeout: Request timeout in seconds.
            max_retries: Maximum number of retries for failed requests.
        """
        self.cache_dir = Path(cache_dir) if cache_dir else Path("./data/cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.timeout = timeout
        
        # Setup session with retry strategy
        self.session = requests.Session()
        retry_strategy = Retry(
            total=max_retries,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
    
    def load_csv(
        self,
        filepath: Union[str, Path],
        validate: bool = True,
        **kwargs,
    ) -> pd.DataFrame:
        """Load data from a CSV file.
        
        Args:
            filepath: Path to the CSV file.
            validate: Whether to validate the loaded data.
            **kwargs: Additional arguments passed to pd.read_csv.
            
        Returns:
            Loaded DataFrame.
            
        Raises:
            FileNotFoundError: If file does not exist.
            ValueError: If validation fails.
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        
        logger.info(f"Loading CSV file: {filepath}")
        df = pd.read_csv(filepath, **kwargs)
        
        if validate:
            self.validate_data(df)
        
        return df
    
    def load_excel(
        self,
        filepath: Union[str, Path],
        sheet_name: Optional[Union[str, int]] = None,
        validate: bool = True,
        **kwargs,
    ) -> pd.DataFrame:
        """Load data from an Excel file.
        
        Args:
            filepath: Path to the Excel file.
            sheet_name: Name or index of the sheet to load.
            validate: Whether to validate the loaded data.
            **kwargs: Additional arguments passed to pd.read_excel.
            
        Returns:
            Loaded DataFrame.
            
        Raises:
            FileNotFoundError: If file does not exist.
            ValueError: If validation fails.
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        
        logger.info(f"Loading Excel file: {filepath}")
        df = pd.read_excel(filepath, sheet_name=sheet_name, **kwargs)
        
        if validate:
            self.validate_data(df)
        
        return df
    
    def load_from_api(
        self,
        url: str,
        params: Optional[Dict] = None,
        headers: Optional[Dict] = None,
        validate: bool = True,
        cache_key: Optional[str] = None,
    ) -> pd.DataFrame:
        """Load data from an API endpoint.
        
        Args:
            url: API endpoint URL.
            params: Query parameters.
            headers: Request headers.
            validate: Whether to validate the loaded data.
            cache_key: Key for caching the response. If None, no caching.
            
        Returns:
            Loaded DataFrame.
            
        Raises:
            requests.RequestException: If request fails.
            ValueError: If response cannot be parsed or validation fails.
        """
        # Check cache first
        if cache_key:
            cache_file = self.cache_dir / f"{cache_key}.pkl"
            if cache_file.exists():
                logger.info(f"Loading from cache: {cache_file}")
                return pd.read_pickle(cache_file)
        
        logger.info(f"Fetching data from API: {url}")
        
        response = self.session.get(
            url,
            params=params,
            headers=headers,
            timeout=self.timeout,
        )
        response.raise_for_status()
        
        # Try to parse as JSON
        try:
            data = response.json()
            
            # Handle different response formats
            if isinstance(data, list):
                df = pd.DataFrame(data)
            elif isinstance(data, dict):
                # Common patterns: {"data": [...]}, {"results": [...]}
                for key in ["data", "results", "items", "records"]:
                    if key in data and isinstance(data[key], list):
                        df = pd.DataFrame(data[key])
                        break
                else:
                    # Try to create DataFrame from dict
                    df = pd.DataFrame([data])
            else:
                raise ValueError(f"Unexpected response format: {type(data)}")
                
        except ValueError:
            # Try to parse as CSV
            from io import StringIO
            df = pd.read_csv(StringIO(response.text))
        
        # Cache the result
        if cache_key:
            cache_file = self.cache_dir / f"{cache_key}.pkl"
            df.to_pickle(cache_file)
            logger.info(f"Cached data to: {cache_file}")
        
        if validate:
            self.validate_data(df)
        
        return df
    
    def load_from_database(
        self,
        database: str,
        api_key: Optional[str] = None,
        query: Optional[Dict] = None,
        **kwargs,
    ) -> pd.DataFrame:
        """Load data from a known public database.
        
        Args:
            database: Database identifier ('nrel', 'perovskitedatabase', 'materials_project').
            api_key: API key for authentication.
            query: Query parameters specific to the database.
            **kwargs: Additional arguments.
            
        Returns:
            Loaded DataFrame.
            
        Raises:
            ValueError: If database is not supported.
            requests.RequestException: If request fails.
        """
        database = database.lower()
        
        if database not in self.PUBLIC_DATABASES:
            raise ValueError(
                f"Unknown database: {database}. "
                f"Supported: {list(self.PUBLIC_DATABASES.keys())}"
            )
        
        base_url = self.PUBLIC_DATABASES[database]
        
        # Database-specific handling
        if database == "nrel":
            return self._load_from_nrel(base_url, api_key, query, **kwargs)
        elif database == "perovskitedatabase":
            return self._load_from_perovskite_db(base_url, query, **kwargs)
        elif database == "materials_project":
            return self._load_from_materials_project(base_url, api_key, query, **kwargs)
        
        # Generic API call
        return self.load_from_api(base_url, params=query, headers={"API-Key": api_key} if api_key else None)
    
    def _load_from_nrel(
        self,
        base_url: str,
        api_key: Optional[str],
        query: Optional[Dict],
        **kwargs,
    ) -> pd.DataFrame:
        """Load data from NREL API."""
        if not api_key:
            raise ValueError("NREL API requires an API key")
        
        endpoint = urljoin(base_url, "output.json")
        params = {"api_key": api_key, **(query or {})}
        
        return self.load_from_api(
            endpoint,
            params=params,
            cache_key=f"nrel_{hash(str(params))}",
            **kwargs,
        )
    
    def _load_from_perovskite_db(
        self,
        base_url: str,
        query: Optional[Dict],
        **kwargs,
    ) -> pd.DataFrame:
        """Load data from Perovskite Database."""
        endpoint = urljoin(base_url, "solar_cells")
        return self.load_from_api(
            endpoint,
            params=query,
            cache_key=f"perovskite_{hash(str(query))}",
            **kwargs,
        )
    
    def _load_from_materials_project(
        self,
        base_url: str,
        api_key: Optional[str],
        query: Optional[Dict],
        **kwargs,
    ) -> pd.DataFrame:
        """Load data from Materials Project API."""
        if not api_key:
            raise ValueError("Materials Project API requires an API key")
        
        endpoint = urljoin(base_url, "materials/perovskite")
        headers = {"X-API-KEY": api_key}
        
        return self.load_from_api(
            endpoint,
            params=query,
            headers=headers,
            cache_key=f"materials_{hash(str(query))}",
            **kwargs,
        )
    
    def validate_data(
        self,
        df: pd.DataFrame,
        strict: bool = False,
    ) -> bool:
        """Validate loaded data for required columns and data quality.
        
        Args:
            df: DataFrame to validate.
            strict: If True, require all optional columns as well.
            
        Returns:
            True if validation passes.
            
        Raises:
            ValueError: If validation fails.
        """
        issues = []
        
        # Check required columns
        missing_required = set(self.REQUIRED_COLUMNS) - set(df.columns)
        if missing_required:
            issues.append(f"Missing required columns: {missing_required}")
        
        # Check optional columns
        missing_optional = set(self.OPTIONAL_COLUMNS) - set(df.columns)
        if missing_optional:
            logger.warning(f"Missing optional columns: {missing_optional}")
        
        if strict and missing_optional:
            issues.append(f"Missing optional columns (strict mode): {missing_optional}")
        
        # Check for empty DataFrame
        if df.empty:
            issues.append("DataFrame is empty")
        
        # Check for duplicate rows
        duplicates = df.duplicated().sum()
        if duplicates > 0:
            logger.warning(f"Found {duplicates} duplicate rows")
        
        # Check data types for required columns
        for col in self.REQUIRED_COLUMNS:
            if col in df.columns:
                if not np.issubdtype(df[col].dtype, np.number):
                    issues.append(f"Column '{col}' should be numeric, got {df[col].dtype}")
        
        # Basic quality checks
        self._check_data_quality(df, issues)
        
        if issues:
            raise ValueError("Data validation failed:\n" + "\n".join(f"  - {issue}" for issue in issues))
        
        logger.info("Data validation passed")
        return True
    
    def _check_data_quality(self, df: pd.DataFrame, issues: List[str]) -> None:
        """Perform data quality checks specific to perovskite solar cells.
        
        Args:
            df: DataFrame to check.
            issues: List to append issues to.
        """
        # Check efficiency range (typically 0-30% for perovskites)
        if "efficiency" in df.columns:
            eff = df["efficiency"]
            if (eff < 0).any():
                issues.append("Negative efficiency values found")
            if (eff > 100).any():
                issues.append("Efficiency values > 100% found")
            elif (eff > 30).any():
                logger.warning("Some efficiency values > 30% (unusual for perovskites)")
        
        # Check Voc range (typically 0.5-1.5V for perovskites)
        if "voc" in df.columns:
            voc = df["voc"]
            if (voc < 0).any():
                issues.append("Negative Voc values found")
            if (voc > 5).any():
                logger.warning("Some Voc values > 5V (unusual for perovskites)")
        
        # Check Jsc range (typically 0-30 mA/cm²)
        if "jsc" in df.columns:
            jsc = df["jsc"]
            if (jsc < 0).any():
                issues.append("Negative Jsc values found")
            if (jsc > 50).any():
                logger.warning("Some Jsc values > 50 mA/cm² (unusual for perovskites)")
        
        # Check FF range (typically 0-100%)
        if "ff" in df.columns:
            ff = df["ff"]
            if (ff < 0).any():
                issues.append("Negative FF values found")
            if (ff > 100).any():
                issues.append("FF values > 100% found")
        
        # Check bandgap range (typically 1-4 eV for perovskites)
        if "bandgap" in df.columns:
            bg = df["bandgap"]
            if (bg < 0).any():
                issues.append("Negative bandgap values found")
            if (bg > 10).any():
                logger.warning("Some bandgap values > 10 eV (unusual for perovskites)")
    
    def get_data_summary(self, df: pd.DataFrame) -> Dict:
        """Get a summary of the loaded data.
        
        Args:
            df: DataFrame to summarize.
            
        Returns:
            Dictionary containing data summary.
        """
        summary = {
            "shape": df.shape,
            "columns": list(df.columns),
            "dtypes": df.dtypes.to_dict(),
            "missing_values": df.isnull().sum().to_dict(),
            "missing_percentage": (df.isnull().sum() / len(df) * 100).to_dict(),
        }
        
        # Add statistics for numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            summary["statistics"] = df[numeric_cols].describe().to_dict()
        
        # Check for required columns
        summary["has_required_columns"] = all(
            col in df.columns for col in self.REQUIRED_COLUMNS
        )
        
        return summary
    
    def clear_cache(self) -> None:
        """Clear all cached data files."""
        import shutil
        
        if self.cache_dir.exists():
            logger.info(f"Clearing cache directory: {self.cache_dir}")
            shutil.rmtree(self.cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
