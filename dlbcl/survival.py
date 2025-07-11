import os
import re
import warnings
from pathlib import Path

from glob import glob
from tqdm import tqdm
from pydantic import Field
from PIL import Image, ImageDraw
import cv2
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt, colors as mcolors
from matplotlib.colors import LinearSegmentedColormap, Normalize
import seaborn as sns
import h5py
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from scipy import stats
from scipy.stats import pearsonr, spearmanr, chi2_contingency, mannwhitneyu, kruskal, f_oneway, ttest_ind
from scipy.cluster.hierarchy import dendrogram, linkage, leaves_list
from scipy.spatial.distance import pdist
from pydantic_autocli import param, AutoCLI
import hdbscan
import torch
import timm
from umap import UMAP
from gigapath import slide_encoder
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
import matplotlib.patches as patches
import scanpy as sc
import anndata as ad

from .utils import BaseMLCLI
from .utils.data_loader import load_common_data

warnings.filterwarnings('ignore', category=FutureWarning, message='.*force_all_finite.*')



class CLI(BaseMLCLI):
    class CommonArgs(BaseMLCLI.CommonArgs):
        pass

    class SurvivalAnalysisArgs(CommonArgs):
        output_dir: str = param('', description="Output directory (defaults to out/{dataset}/survival)")
        time_col: str = param('OS', choices=['OS', 'PFS'], description="Survival time column")
        event_col: str = param('Follow-up Status', description="Event column (1=event, 0=censored)")

    def run_survival_analysis(self, a: SurvivalAnalysisArgs):
        """Generate survival curves for clinical data and Leiden clusters"""

        if not a.output_dir:
            a.output_dir = f'out/{a.dataset}/survival'

        os.makedirs(a.output_dir, exist_ok=True)

        print(f"Generating survival analysis for {a.dataset} dataset...")

        # Use merged data
        df = self.merged_data.copy()

        # Add leiden clusters
        leiden_file = str(self.dataset_dir / 'leiden_clusters_with_ids.csv')
        if os.path.exists(leiden_file):
            leiden_df = pd.read_csv(leiden_file, index_col=0)
            # Create patient_order mapping for leiden results
            df['patient_order_key'] = df.apply(lambda row: f"{row['patient_id']}__{row.get('order', 0)}", axis=1)

            # Map leiden clusters
            leiden_mapping = leiden_df['leiden_cluster'].to_dict()
            df['leiden_cluster'] = df['patient_order_key'].map(leiden_mapping)
            df['leiden_cluster'] = df['leiden_cluster'].fillna(-1).astype(int)
        else:
            print(f"Warning: Leiden clustering results not found: {leiden_file}")
            df['leiden_cluster'] = -1

        # Check required columns
        if a.time_col not in df.columns:
            raise ValueError(f"Time column '{a.time_col}' not found in dataset")
        if a.event_col not in df.columns:
            raise ValueError(f"Event column '{a.event_col}' not found in dataset")

        # Clean data for survival analysis - include all IHC markers
        survival_cols = [a.time_col, a.event_col, 'leiden_cluster', 'HANS', 'CD10_binary']
        ihc_markers = ['CD10 IHC', 'MUM1 IHC', 'BCL2 IHC', 'BCL6 IHC', 'MYC IHC']
        age_ldh = ['Age', 'LDH']

        # Add columns that exist in the dataset
        all_cols = survival_cols + [col for col in ihc_markers + age_ldh if col in df.columns]
        survival_df = df[all_cols].copy()
        survival_df = survival_df.dropna(subset=[a.time_col, a.event_col])

        if len(survival_df) == 0:
            print("No valid survival data found")
            return

        print(f"Survival analysis data: {len(survival_df)} samples")

        # Import survival analysis libraries
        try:
            from lifelines import KaplanMeierFitter
            from lifelines.statistics import logrank_test
        except ImportError:
            print("Installing lifelines for survival analysis...")
            import subprocess
            subprocess.run(['uv', 'add', 'lifelines'], check=True)
            from lifelines import KaplanMeierFitter
            from lifelines.statistics import logrank_test

        # Generate survival curves
        self._create_survival_curves(survival_df, a)

    def _create_survival_curves(self, df, a):
        """Create Kaplan-Meier survival curves"""
        from lifelines import KaplanMeierFitter
        from lifelines.statistics import logrank_test

        # Survival factors to analyze
        factors = {
            'leiden_cluster': 'Leiden Cluster',
            'HANS': 'HANS Classification',
            'CD10_binary': 'CD10 Status',
            'CD10 IHC': 'CD10 IHC',
            'MUM1 IHC': 'MUM1 IHC',
            'BCL2 IHC': 'BCL2 IHC',
            'BCL6 IHC': 'BCL6 IHC',
            'MYC IHC': 'MYC IHC',
            'Age': 'Age',
            'LDH': 'LDH'
        }

        for factor_col, factor_name in factors.items():
            if factor_col not in df.columns:
                continue

            # Remove missing values
            factor_df = df.dropna(subset=[factor_col])
            if len(factor_df) < 10:
                print(f"Insufficient data for {factor_name} survival analysis")
                continue

            print(f"\nGenerating survival curves for {factor_name}...")

            plt.figure(figsize=(10, 6))

            kmf = KaplanMeierFitter()

            # Get unique groups
            groups = sorted(factor_df[factor_col].unique())
            p_values = []

            # Handle continuous variables (IHC markers, Age, LDH) by creating binary groups
            if factor_col in ['CD10 IHC', 'MUM1 IHC', 'BCL2 IHC', 'BCL6 IHC', 'MYC IHC', 'Age', 'LDH']:
                # Create binary groups - use different strategies based on data distribution
                unique_values = sorted(factor_df[factor_col].dropna().unique())

                # Special handling for LDH (binary variable: 0 vs 1)
                if factor_col == 'LDH':
                    # LDH is already binary (0=normal, 1=high), use as-is
                    if len(unique_values) == 2 and set(unique_values) == {0, 1}:
                        factor_df['binary_group'] = factor_df[factor_col].astype(int)
                        group_0_size = (factor_df['binary_group'] == 0).sum()
                        group_1_size = (factor_df['binary_group'] == 1).sum()

                        if group_0_size >= 5 and group_1_size >= 5:
                            binary_labels = {0: f'{factor_name} Normal (=0)',
                                           1: f'{factor_name} High (=1)'}
                            groups = [0, 1]
                        else:
                            print(f"  Insufficient samples for LDH binary split: (normal: {group_0_size}, high: {group_1_size})")
                            continue
                    else:
                        print(f"  LDH values are not binary as expected: {unique_values}")
                        continue
                # If many values are 0, use 0 vs >0 split (for IHC markers)
                elif 0 in unique_values and (factor_df[factor_col] == 0).sum() / len(factor_df) > 0.3:
                    factor_df['binary_group'] = (factor_df[factor_col] > 0).astype(int)
                    # Check if both groups have sufficient samples
                    group_0_size = (factor_df['binary_group'] == 0).sum()
                    group_1_size = (factor_df['binary_group'] == 1).sum()

                    if group_0_size >= 5 and group_1_size >= 5:
                        binary_labels = {0: f'{factor_name} Negative (=0)',
                                       1: f'{factor_name} Positive (>0)'}
                        groups = [0, 1]
                    else:
                        print(f"  Insufficient samples for binary split: {factor_name} (negative: {group_0_size}, positive: {group_1_size})")
                        continue
                else:
                    # Use median split for normally distributed data (Age etc.)
                    median_val = factor_df[factor_col].median()
                    factor_df['binary_group'] = (factor_df[factor_col] >= median_val).astype(int)

                    # Check if both groups have sufficient samples
                    group_0_size = (factor_df['binary_group'] == 0).sum()
                    group_1_size = (factor_df['binary_group'] == 1).sum()

                    if group_0_size >= 5 and group_1_size >= 5:
                        binary_labels = {0: f'{factor_name} Low (<{median_val:.1f})',
                                       1: f'{factor_name} High (≥{median_val:.1f})'}
                        groups = [0, 1]
                    else:
                        print(f"  Insufficient samples for median split: {factor_name} (low: {group_0_size}, high: {group_1_size})")
                        continue
            else:
                binary_labels = None

            # Plot survival curves for each group
            for i, group in enumerate(groups):
                if binary_labels:
                    group_mask = factor_df['binary_group'] == group
                    label = binary_labels[group]
                else:
                    group_mask = factor_df[factor_col] == group
                    label = f'{factor_name} {group}'

                group_df = factor_df[group_mask]

                if len(group_df) < 5:  # Skip groups with too few samples
                    continue

                kmf.fit(group_df[a.time_col], group_df[a.event_col], label=label)
                kmf.plot_survival_function(ci_show=True)

                # Calculate log-rank test against other groups
                for other_group in groups:
                    if other_group != group:
                        if binary_labels:
                            other_mask = factor_df['binary_group'] == other_group
                        else:
                            other_mask = factor_df[factor_col] == other_group
                        other_df = factor_df[other_mask]

                        if len(other_df) >= 5:
                            try:
                                results = logrank_test(group_df[a.time_col], other_df[a.time_col],
                                                     group_df[a.event_col], other_df[a.event_col])
                                p_values.append(results.p_value)
                            except:
                                pass

            plt.title(f'Kaplan-Meier Survival Curves by {factor_name}\n({a.dataset.upper()} dataset, {a.time_col})')
            plt.xlabel(f'Time ({a.time_col})')
            plt.ylabel('Survival Probability')
            plt.legend()
            plt.grid(True, alpha=0.3)

            # Add p-value if calculated
            if p_values:
                min_p = min(p_values)
                plt.text(0.02, 0.02, f'Min p-value: {min_p:.4f}', transform=plt.gca().transAxes,
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

            # Save plot
            plot_file = f"{a.output_dir}/survival_{factor_col}_{a.time_col}.png"
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()

            print(f"  Saved: {plot_file}")

            # Generate summary statistics
            summary_file = f"{a.output_dir}/survival_{factor_col}_{a.time_col}_summary.txt"
            with open(summary_file, 'w') as f:
                f.write(f"Survival Analysis Summary: {factor_name}\n")
                f.write(f"Dataset: {a.dataset.upper()}\n")
                f.write(f"Time variable: {a.time_col}\n")
                f.write(f"Event variable: {a.event_col}\n\n")

                for group in groups:
                    if factor_col in ['CD10 IHC', 'MUM1 IHC', 'BCL2 IHC', 'BCL6 IHC', 'MYC IHC', 'Age', 'LDH']:
                        group_mask = factor_df['binary_group'] == group
                        group_label = binary_labels[group] if binary_labels else f"{factor_name} {group}"
                    else:
                        group_mask = factor_df[factor_col] == group
                        group_label = f"{factor_name} {group}"

                    group_df = factor_df[group_mask]

                    if len(group_df) >= 5:
                        median_survival = group_df[a.time_col].median()
                        events = group_df[a.event_col].sum()
                        f.write(f"{group_label}:\n")
                        f.write(f"  Sample size: {len(group_df)}\n")
                        f.write(f"  Events: {events}\n")
                        f.write(f"  Median {a.time_col}: {median_survival:.1f}\n\n")

                if p_values:
                    f.write(f"Statistical significance:\n")
                    f.write(f"  Minimum p-value: {min(p_values):.4f}\n")
                    f.write(f"  Number of comparisons: {len(p_values)}\n")

            print(f"  Summary: {summary_file}")


if __name__ == '__main__':
    cli = CLI()
    cli.run()
