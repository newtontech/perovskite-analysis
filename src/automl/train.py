#!/usr/bin/env python3
"""
Perovskite Solar Cells - AutoML Pipeline
Target: Predict PCE from material properties
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import joblib
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class PerovskiteAutoML:
    def __init__(self, data_path):
        """Initialize AutoML pipeline"""
        self.data_path = Path(data_path)
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.models = {}
        self.best_model = None
        self.model_dir = Path("models")
        self.model_dir.mkdir(exist_ok=True)
        
    def load_and_preprocess(self):
        """Load and preprocess data"""
        print("Loading and preprocessing data...")
        self.df = pd.read_csv(self.data_path)
        print(f"✓ Loaded {len(self.df):,} records")
        
        # Select features (numeric + categorical)
        numeric_features = [
            'perovskite_band_gap',
            'jv_reverse_scan_j_sc',
            'jv_reverse_scan_v_oc',
            'jv_reverse_scan_ff',
            'cell_area_measured'
        ]
        
        # Filter available features
        available_features = [f for f in numeric_features if f in self.df.columns]
        print(f"Using {len(available_features)} features: {available_features}")
        
        # Prepare data
        df_clean = self.df.dropna(subset=['jv_reverse_scan_pce'] + available_features)
        X = df_clean[available_features]
        y = df_clean['jv_reverse_scan_pce']
        
        print(f"✓ Clean dataset: {len(df_clean):,} samples")
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        print(f"✓ Train: {len(self.X_train):,}, Test: {len(self.X_test):,}")
        
        return self
    
    def train_models(self):
        """Train multiple models"""
        print("\n" + "="*60)
        print("TRAINING MODELS")
        print("="*60)
        
        # Define models
        models_config = {
            'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
            'GradientBoosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
        }
        
        results = {}
        
        for name, model in models_config.items():
            print(f"\nTraining {name}...")
            
            # Train
            model.fit(self.X_train, self.y_train)
            
            # Predict
            y_pred = model.predict(self.X_test)
            
            # Evaluate
            r2 = r2_score(self.y_test, y_pred)
            mae = mean_absolute_error(self.y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(self.y_test, y_pred))
            
            results[name] = {
                'model': model,
                'r2': r2,
                'mae': mae,
                'rmse': rmse
            }
            
            print(f"  ✓ R²: {r2:.4f}")
            print(f"  ✓ MAE: {mae:.4f}%")
            print(f"  ✓ RMSE: {rmse:.4f}%")
            
            # Save model
            model_path = self.model_dir / f"{name.lower()}_model.joblib"
            joblib.dump(model, model_path)
            print(f"  ✓ Saved: {model_path}")
        
        # Select best model
        best_name = max(results, key=lambda x: results[x]['r2'])
        self.best_model = results[best_name]['model']
        print(f"\n✓ Best Model: {best_name} (R² = {results[best_name]['r2']:.4f})")
        
        return results
    
    def generate_report(self, results):
        """Generate performance report"""
        print("\n" + "="*60)
        print("MODEL COMPARISON")
        print("="*60)
        
        report_data = []
        for name, metrics in results.items():
            report_data.append({
                'Model': name,
                'R²': f"{metrics['r2']:.4f}",
                'MAE (%)': f"{metrics['mae']:.4f}",
                'RMSE (%)': f"{metrics['rmse']:.4f}"
            })
        
        report_df = pd.DataFrame(report_data)
        print(report_df.to_string(index=False))
        
        # Save report
        report_path = self.model_dir / "model_comparison.csv"
        report_df.to_csv(report_path, index=False)
        print(f"\n✓ Saved: {report_path}")
        
        return report_df
    
    def run_pipeline(self):
        """Run complete AutoML pipeline"""
        print("\n" + "="*80)
        print(" PEROVSKITE SOLAR CELLS - AUTOML PIPELINE ".center(80, "="))
        print("="*80 + "\n")
        
        # Load and preprocess
        self.load_and_preprocess()
        
        # Train models
        results = self.train_models()
        
        # Generate report
        self.generate_report(results)
        
        print("\n" + "="*80)
        print(" AUTOML COMPLETE - Models saved to models/ ".center(80, "="))
        print("="*80 + "\n")

if __name__ == "__main__":
    automl = PerovskiteAutoML("data/raw/perovskite_raw_data_for_import.csv")
    automl.run_pipeline()
