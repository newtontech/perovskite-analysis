#!/usr/bin/env python3
"""
Perovskite Solar Cells - Exploratory Data Analysis
Inspired by Nature Energy battery technology study
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class PerovskiteEDA:
    def __init__(self, data_path):
        """Initialize EDA with data path"""
        self.data_path = Path(data_path)
        self.df = None
        self.report_dir = Path("reports/figures")
        self.report_dir.mkdir(parents=True, exist_ok=True)
        
    def load_data(self):
        """Load perovskite dataset"""
        print("Loading dataset...")
        self.df = pd.read_csv(self.data_path)
        print(f"✓ Loaded {len(self.df):,} records with {len(self.df.columns)} features")
        return self.df
    
    def basic_stats(self):
        """Generate basic statistics"""
        print("\n" + "="*60)
        print("BASIC STATISTICS")
        print("="*60)
        
        # Data overview
        print(f"\nDataset Shape: {self.df.shape}")
        print(f"Memory Usage: {self.df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        # Missing values
        missing = self.df.isnull().sum()
        missing_pct = (missing / len(self.df) * 100).sort_values(ascending=False)
        print(f"\nTop 10 Features with Missing Values:")
        print(missing_pct[missing_pct > 0].head(10))
        
        # Data types
        print(f"\nData Types:")
        print(self.df.dtypes.value_counts())
        
        return missing_pct
    
    def target_analysis(self, target='jv_reverse_scan_pce'):
        """Analyze target variable (PCE)"""
        print("\n" + "="*60)
        print("TARGET VARIABLE ANALYSIS (PCE)")
        print("="*60)
        
        if target not in self.df.columns:
            print(f"Warning: {target} not found in dataset")
            return
        
        # Statistics
        pce = self.df[target].dropna()
        print(f"\nPCE Statistics:")
        print(f"  Count: {len(pce):,}")
        print(f"  Mean: {pce.mean():.2f}%")
        print(f"  Median: {pce.median():.2f}%")
        print(f"  Std: {pce.std():.2f}%")
        print(f"  Min: {pce.min():.2f}%")
        print(f"  Max: {pce.max():.2f}%")
        print(f"  Range: {pce.max() - pce.min():.2f}%")
        
        # Distribution plot
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Histogram
        axes[0].hist(pce, bins=50, edgecolor='black', alpha=0.7)
        axes[0].set_xlabel('PCE (%)', fontsize=12)
        axes[0].set_ylabel('Frequency', fontsize=12)
        axes[0].set_title('PCE Distribution', fontsize=14, fontweight='bold')
        axes[0].axvline(pce.mean(), color='red', linestyle='--', label=f'Mean: {pce.mean():.2f}%')
        axes[0].legend()
        
        # Box plot
        axes[1].boxplot(pce, vert=True)
        axes[1].set_ylabel('PCE (%)', fontsize=12)
        axes[1].set_title('PCE Box Plot', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.report_dir / 'pce_distribution.png', dpi=300, bbox_inches='tight')
        print(f"\n✓ Saved: {self.report_dir / 'pce_distribution.png'}")
        plt.close()
        
        return pce
    
    def time_evolution(self, date_col='publication_date'):
        """Analyze performance evolution over time"""
        print("\n" + "="*60)
        print("TIME EVOLUTION ANALYSIS")
        print("="*60)
        
        if date_col not in self.df.columns:
            print(f"Warning: {date_col} not found")
            return
        
        # Convert to datetime
        self.df[date_col] = pd.to_datetime(self.df[date_col], errors='coerce')
        df_time = self.df.dropna(subset=[date_col, 'jv_reverse_scan_pce'])
        
        # Extract year
        df_time['year'] = df_time[date_col].dt.year
        
        # Yearly statistics
        yearly_stats = df_time.groupby('year')['jv_reverse_scan_pce'].agg(['count', 'mean', 'max', 'std'])
        print(f"\nYearly PCE Evolution:")
        print(yearly_stats)
        
        # Plot
        fig, ax = plt.subplots(figsize=(12, 6))
        
        ax.plot(yearly_stats.index, yearly_stats['mean'], 'o-', label='Mean PCE', linewidth=2)
        ax.fill_between(yearly_stats.index, 
                        yearly_stats['mean'] - yearly_stats['std'],
                        yearly_stats['mean'] + yearly_stats['std'],
                        alpha=0.3, label='± 1 Std')
        ax.plot(yearly_stats.index, yearly_stats['max'], 's--', label='Max PCE', linewidth=2)
        
        ax.set_xlabel('Year', fontsize=12)
        ax.set_ylabel('PCE (%)', fontsize=12)
        ax.set_title('Perovskite Solar Cell Efficiency Evolution (2015-2025)', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.report_dir / 'pce_evolution.png', dpi=300, bbox_inches='tight')
        print(f"\n✓ Saved: {self.report_dir / 'pce_evolution.png'}")
        plt.close()
        
        return yearly_stats
    
    def material_analysis(self):
        """Analyze perovskite material compositions"""
        print("\n" + "="*60)
        print("MATERIAL COMPOSITION ANALYSIS")
        print("="*60)
        
        if 'perovskite_composition' not in self.df.columns:
            print("Warning: perovskite_composition not found")
            return
        
        # Top compositions
        top_compositions = self.df['perovskite_composition'].value_counts().head(10)
        print(f"\nTop 10 Perovskite Compositions:")
        print(top_compositions)
        
        # Plot
        fig, ax = plt.subplots(figsize=(12, 6))
        top_compositions.plot(kind='bar', ax=ax, color='skyblue', edgecolor='black')
        ax.set_xlabel('Composition', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.set_title('Top 10 Perovskite Compositions', fontsize=14, fontweight='bold')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        plt.savefig(self.report_dir / 'material_compositions.png', dpi=300, bbox_inches='tight')
        print(f"\n✓ Saved: {self.report_dir / 'material_compositions.png'}")
        plt.close()
        
        return top_compositions
    
    def run_full_eda(self):
        """Run complete EDA pipeline"""
        print("\n" + "="*80)
        print(" PEROVSKITE SOLAR CELLS - EXPLORATORY DATA ANALYSIS ".center(80, "="))
        print("="*80 + "\n")
        
        # Load data
        self.load_data()
        
        # Run analyses
        self.basic_stats()
        self.target_analysis()
        self.time_evolution()
        self.material_analysis()
        
        print("\n" + "="*80)
        print(" EDA COMPLETE - All figures saved to reports/figures/ ".center(80, "="))
        print("="*80 + "\n")

if __name__ == "__main__":
    eda = PerovskiteEDA("data/raw/perovskite_raw_data_for_import.csv")
    eda.run_full_eda()
