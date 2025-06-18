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
from .utils.correlation import (
    load_both_datasets, get_clinical_mapping, compute_correlation_matrix,
    prepare_comparison_matrices, create_unified_dendrogram
)

warnings.filterwarnings('ignore', category=FutureWarning, message='.*force_all_finite.*')



class CLI(BaseMLCLI):
    class CommonArgs(BaseMLCLI.CommonArgs):
        pass


    # Legacy comprehensive_heatmap functionality has been integrated into run_compare_correlation

    class CompareCorrelationArgs(CommonArgs):
        output_dir: str = param('', description="Output directory (defaults to out/compare_correlation)")
        correlation_method: str = param('pearson', choices=['pearson', 'spearman'], description="Correlation method to use")
        figsize_width: int = param(24, description="Figure width in inches")
        figsize_height: int = param(12, description="Figure height in inches")

    def run_compare_correlation(self, a: CompareCorrelationArgs):
        """Compare correlation heatmaps between Morph and Patho2 datasets with unified feature ordering"""

        if not a.output_dir:
            a.output_dir = 'out/correlation/compare_correlation'

        os.makedirs(a.output_dir, exist_ok=True)

        print("Computing correlation matrices for both datasets...")

        # Load data for both datasets
        morph_merged, patho2_merged = load_both_datasets()

        print(f"Loaded morph data: {morph_merged.shape}")
        print(f"Loaded patho2 data: {patho2_merged.shape}")

        # Compute correlation matrices
        morph_corr = compute_correlation_matrix(morph_merged, a.correlation_method)
        patho2_corr = compute_correlation_matrix(patho2_merged, a.correlation_method)

        print(f"Morph correlation matrix: {morph_corr.shape}")
        print(f"Patho2 correlation matrix: {patho2_corr.shape}")

        # Save correlation matrices for future reference
        os.makedirs(f"{a.output_dir}/morph", exist_ok=True)
        os.makedirs(f"{a.output_dir}/patho2", exist_ok=True)
        morph_corr.to_csv(f"{a.output_dir}/morph/correlation_matrix.csv")
        patho2_corr.to_csv(f"{a.output_dir}/patho2/correlation_matrix.csv")

        # Get clinical mapping and prepare data matrices
        clinical_mapping = get_clinical_mapping(patho2_merged.columns)
        morph_subset, patho2_subset, variable_labels = prepare_comparison_matrices(
            morph_corr, patho2_corr, clinical_mapping)

        # Create dendrogram for feature ordering
        feature_linkage, feature_order = create_unified_dendrogram(morph_subset, patho2_subset)

        # Reorder both datasets using the same feature order
        morph_ordered = morph_subset.iloc[feature_order]
        patho2_ordered = patho2_subset.iloc[feature_order]

        # Create unified heatmap with dendrogram
        fig = plt.figure(figsize=(a.figsize_width, a.figsize_height))

        # Grid: dendrogram | morph | patho2 | colorbar
        gs = fig.add_gridspec(1, 4, width_ratios=[0.1, 0.4, 0.4, 0.1],
                             hspace=0.02, wspace=0.1)

        # Plot dendrogram
        ax_dendro = fig.add_subplot(gs[0, 0])
        dendro = dendrogram(feature_linkage, ax=ax_dendro, orientation='left',
                           no_labels=True, color_threshold=0)
        ax_dendro.set_xticks([])
        ax_dendro.set_yticks([])
        ax_dendro.set_title('Dendrogram', fontsize=12)
        for spine in ax_dendro.spines.values():
            spine.set_visible(False)

        # Morph heatmap
        ax_morph = fig.add_subplot(gs[0, 1])
        im1 = ax_morph.imshow(morph_ordered.values, cmap=plt.cm.RdBu_r,
                             aspect='auto', vmin=-1, vmax=1, interpolation='nearest')
        ax_morph.set_title('MORPH Dataset', fontsize=14)
        ax_morph.set_xticks(range(len(variable_labels)))
        ax_morph.set_xticklabels(variable_labels, rotation=45, ha='right', fontsize=9)
        ax_morph.set_yticks(range(0, len(morph_ordered), 50))
        ax_morph.set_yticklabels([f"F{feature_order[i]}" for i in range(0, len(morph_ordered), 50)], fontsize=8)
        ax_morph.set_ylabel('Feature Index (Ordered by Dendrogram)')

        # Patho2 heatmap with NaN masking
        ax_patho2 = fig.add_subplot(gs[0, 2])

        # Create masked array for NaN values
        patho2_values = patho2_ordered.values
        mask = np.isnan(patho2_values)

        im2 = ax_patho2.imshow(patho2_values, cmap=plt.cm.RdBu_r,
                              aspect='auto', vmin=-1, vmax=1, interpolation='nearest')

        # Overlay gray for missing data
        import matplotlib.patches as patches
        for i in range(patho2_values.shape[0]):
            for j in range(patho2_values.shape[1]):
                if mask[i, j]:
                    ax_patho2.add_patch(patches.Rectangle((j-0.5, i-0.5), 1, 1,
                                                         facecolor='lightgray', edgecolor='darkgray', alpha=0.8))

        ax_patho2.set_title('PATHO2 Dataset', fontsize=14)
        ax_patho2.set_xticks(range(len(variable_labels)))
        ax_patho2.set_xticklabels(variable_labels, rotation=45, ha='right', fontsize=9)
        ax_patho2.set_yticks([])

        # Add colorbar
        ax_colorbar = fig.add_subplot(gs[0, 3])
        cbar = plt.colorbar(im2, cax=ax_colorbar)
        cbar.set_label(f'{a.correlation_method.capitalize()}\nCorrelation', rotation=270, labelpad=20)

        plt.suptitle(f'Unified Feature-Clinical Correlation Comparison\n' +
                    f'Features ordered by Morph dendrogram clustering', fontsize=16, y=0.98)

        # Adjust layout to prevent cutoff
        plt.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.1)

        # Save the unified comparison plot
        plt.savefig(f"{a.output_dir}/unified_correlation_comparison.png", dpi=300, bbox_inches='tight')
        plt.savefig(f"{a.output_dir}/unified_correlation_comparison.pdf", dpi=300, bbox_inches='tight')
        plt.close()

        # Create difference analysis
        print("Creating correlation difference analysis...")

        # Calculate differences (align columns properly)
        morph_aligned = morph_ordered.values
        patho2_aligned = patho2_ordered.values
        diff_matrix = morph_aligned - patho2_aligned

        # Create difference heatmap with dendrogram (matching the main comparison layout)
        fig_diff = plt.figure(figsize=(16, 10))

        # Grid: dendrogram | difference heatmap | colorbar
        gs_diff = fig_diff.add_gridspec(1, 3, width_ratios=[0.15, 0.75, 0.1],
                                       hspace=0.02, wspace=0.1)

        # Plot same dendrogram on left
        ax_dendro_diff = fig_diff.add_subplot(gs_diff[0, 0])
        dendro_diff = dendrogram(feature_linkage, ax=ax_dendro_diff, orientation='left',
                               no_labels=True, color_threshold=0)
        ax_dendro_diff.set_xticks([])
        ax_dendro_diff.set_yticks([])
        ax_dendro_diff.set_title('Dendrogram', fontsize=12)
        for spine in ax_dendro_diff.spines.values():
            spine.set_visible(False)

        # Difference heatmap in center
        ax_diff = fig_diff.add_subplot(gs_diff[0, 1])
        im_diff = ax_diff.imshow(diff_matrix, cmap=plt.cm.RdBu_r, aspect='auto',
                               vmin=-1, vmax=1, interpolation='nearest')
        ax_diff.set_title('Correlation Differences (Morph - Patho2)', fontsize=14)
        ax_diff.set_xticks(range(len(variable_labels)))
        ax_diff.set_xticklabels(variable_labels, rotation=45, ha='right', fontsize=9)
        ax_diff.set_yticks(range(0, len(morph_ordered), 50))
        ax_diff.set_yticklabels([f"F{feature_order[i]}" for i in range(0, len(morph_ordered), 50)], fontsize=8)
        ax_diff.set_ylabel('Feature Index (Dendrogram Order)')

        # Colorbar on right
        ax_cbar_diff = fig_diff.add_subplot(gs_diff[0, 2])
        cbar_diff = plt.colorbar(im_diff, cax=ax_cbar_diff)
        cbar_diff.set_label('Correlation Difference\n(Morph - Patho2)', rotation=270, labelpad=25)

        plt.suptitle(f'Correlation Differences Between Datasets\nFeatures ordered by Morph dendrogram clustering',
                    fontsize=16, y=0.98)

        # Adjust layout
        plt.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.1)

        plt.savefig(f"{a.output_dir}/unified_correlation_difference.png", dpi=300, bbox_inches='tight')
        plt.close()

        # Summary statistics
        summary = {
            'total_features': len(morph_ordered),
            'mapped_clinical_vars': len(morph_clinical_vars),
            'morph_vars': morph_clinical_vars,
            'patho2_vars': patho2_clinical_vars,
            'mean_abs_diff': np.nanmean(np.abs(diff_matrix)),
            'max_abs_diff': np.nanmax(np.abs(diff_matrix)),
            'highly_different_pairs': np.sum(np.abs(diff_matrix) > 0.3),
            'correlation_method': a.correlation_method
        }

        with open(f"{a.output_dir}/unified_comparison_summary.txt", 'w') as f:
            f.write("Unified Dataset Correlation Comparison Summary\n")
            f.write("="*50 + "\n\n")
            f.write(f"Features: {summary['total_features']} (ordered by Morph dendrogram)\n")
            f.write(f"Clinical variables mapped: {summary['mapped_clinical_vars']}\n\n")
            f.write("Variable mapping:\n")
            for i, (m, p) in enumerate(zip(morph_clinical_vars, patho2_clinical_vars)):
                f.write(f"  {i+1}. {m} (Morph) <-> {p} (Patho2)\n")
            f.write(f"\nCorrelation differences:\n")
            f.write(f"  Mean absolute difference: {summary['mean_abs_diff']:.3f}\n")
            f.write(f"  Maximum absolute difference: {summary['max_abs_diff']:.3f}\n")
            f.write(f"  Highly different pairs (|diff| > 0.3): {summary['highly_different_pairs']}\n")
            f.write(f"  Correlation method: {summary['correlation_method']}\n")

        print(f"Unified comparison completed! Results saved to {a.output_dir}")
        print(f"Features analyzed: {len(morph_ordered)} (dendrogram ordered)")
        print(f"Clinical variables mapped: {len(morph_clinical_vars)}")
        print(f"Mean absolute correlation difference: {summary['mean_abs_diff']:.3f}")

        # Create concatenated unified correlation heatmap
        print("\n=== Creating Concatenated Unified Correlation ===")
        self._create_concatenated_correlation(common_mapping, feature_linkage, feature_order, a)

        # Return None to avoid warning about unexpected return type
        return None

    class ChannelCorrelationArgs(CommonArgs):
        output_dir: str = param('', description="Output directory (defaults to out/channel_correlation)")
        correlation_method: str = param('pearson', choices=['pearson', 'spearman'], description="Correlation method to use")
        figsize_width: int = param(16, description="Figure width in inches")
        figsize_height: int = param(12, description="Figure height in inches")

    def run_channel_correlation(self, a: ChannelCorrelationArgs):
        """Compare channel-clinical correlations between datasets and identify consistent markers"""

        if not a.output_dir:
            a.output_dir = 'out/correlation/channel_correlation'

        os.makedirs(a.output_dir, exist_ok=True)

        print("Channel correlation analysis...")

        # Load data for both datasets
        morph_merged, patho2_merged = load_both_datasets()

        print(f"Morph data: {morph_merged.shape}, Patho2 data: {patho2_merged.shape}")

        # Get feature columns
        morph_features = [col for col in morph_merged.columns if col.startswith('feature_')]
        patho2_features = [col for col in patho2_merged.columns if col.startswith('feature_')]

        # Use existing mapping logic from run_compare_correlation
        clinical_mapping = get_clinical_mapping(patho2_merged.columns)

        # Calculate correlations for each dataset
        print("Computing feature-clinical correlations...")
        results = {}

        for morph_col, patho2_col in clinical_mapping.items():
            if morph_col not in morph_merged.columns or patho2_col not in patho2_merged.columns:
                continue

            print(f"Processing {morph_col} vs {patho2_col}...")

            # Compute correlations using refactored helper
            morph_corrs = self._compute_feature_clinical_correlations(
                morph_merged, morph_features, morph_col, a.correlation_method)
            patho2_corrs = self._compute_feature_clinical_correlations(
                patho2_merged, patho2_features, patho2_col, a.correlation_method)

            results[morph_col] = {
                'morph_corrs': np.array(morph_corrs),
                'patho2_corrs': np.array(patho2_corrs),
                'patho2_name': patho2_col
            }

        # Create visualization and analysis
        consistency_scores = self._create_correlation_scatter_plots(results, a)
        self._save_correlation_results(results, consistency_scores, a)

        print(f"\nResults saved to {a.output_dir}/")
        print(f"- Scatter plots: channel_correlation_scatter.png")
        print(f"- Consistency ranking: consistency_ranking.csv")
        print(f"- Detailed correlations: feature_clinical_correlations.csv")


    def _compute_feature_clinical_correlations(self, merged_data, feature_cols, clinical_col, correlation_method):
        """Compute correlations between all features and a single clinical variable"""
        correlations = []

        for feat_col in feature_cols:
            feature_data = merged_data[feat_col]
            clinical_data = merged_data[clinical_col]

            valid_mask = ~(feature_data.isna() | clinical_data.isna())

            if valid_mask.sum() >= 10:  # Minimum 10 samples for correlation
                try:
                    if correlation_method == 'pearson':
                        corr, _ = pearsonr(feature_data[valid_mask], clinical_data[valid_mask])
                    else:
                        corr, _ = spearmanr(feature_data[valid_mask], clinical_data[valid_mask])
                    correlations.append(corr)
                except:
                    correlations.append(np.nan)
            else:
                correlations.append(np.nan)

        return correlations

    def _create_correlation_scatter_plots(self, results, a):
        """Create scatter plots comparing correlations between datasets"""
        n_markers = len(results)
        n_cols = 3
        n_rows = (n_markers + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(a.figsize_width, a.figsize_height))
        if n_rows == 1:
            axes = axes.reshape(1, -1)

        plot_idx = 0
        consistency_scores = {}

        for marker, data in results.items():
            row = plot_idx // n_cols
            col = plot_idx % n_cols
            ax = axes[row, col]

            morph_corrs = data['morph_corrs']
            patho2_corrs = data['patho2_corrs']

            # Remove NaN values for plotting and correlation calculation
            valid_mask = ~(np.isnan(morph_corrs) | np.isnan(patho2_corrs))
            morph_valid = morph_corrs[valid_mask]
            patho2_valid = patho2_corrs[valid_mask]

            if len(morph_valid) > 0:
                ax.scatter(morph_valid, patho2_valid, alpha=0.6, s=1)

                # Calculate consistency (correlation between dataset correlations)
                if len(morph_valid) > 1:
                    try:
                        consistency, _ = pearsonr(morph_valid, patho2_valid)
                        consistency_scores[marker] = consistency
                    except:
                        consistency_scores[marker] = np.nan
                        consistency = np.nan
                else:
                    consistency_scores[marker] = np.nan
                    consistency = np.nan

                # Add diagonal line
                min_val = min(np.min(morph_valid), np.min(patho2_valid))
                max_val = max(np.max(morph_valid), np.max(patho2_valid))
                ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5)

                ax.set_xlabel(f'Morph {marker} Correlation')
                ax.set_ylabel(f'Patho2 {data["patho2_name"]} Correlation')
                ax.set_title(f'{marker}\nConsistency: {consistency:.3f}' if not np.isnan(consistency) else f'{marker}\nConsistency: N/A')
                ax.grid(True, alpha=0.3)
            else:
                ax.text(0.5, 0.5, 'No valid data', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'{marker}\nNo data')
                consistency_scores[marker] = np.nan

            plot_idx += 1

        # Hide unused subplots
        for i in range(plot_idx, n_rows * n_cols):
            row = i // n_cols
            col = i % n_cols
            axes[row, col].set_visible(False)

        plt.tight_layout()
        plt.savefig(f"{a.output_dir}/channel_correlation_scatter.png", dpi=300, bbox_inches='tight')
        plt.close()

        return consistency_scores

    def _save_correlation_results(self, results, consistency_scores, a):
        """Save correlation analysis results to files"""
        # Create consistency ranking
        print("\n=== Dataset Consistency Ranking ===")
        consistency_df = pd.DataFrame(list(consistency_scores.items()),
                                    columns=['Marker', 'Consistency'])
        consistency_df = consistency_df.sort_values('Consistency', ascending=False)

        print(consistency_df.round(3))
        consistency_df.to_csv(f"{a.output_dir}/consistency_ranking.csv", index=False)

        # Save detailed correlation data
        correlation_data = {}
        for marker, data in results.items():
            correlation_data[f'{marker}_morph'] = data['morph_corrs']
            correlation_data[f'{marker}_patho2'] = data['patho2_corrs']

        correlation_df = pd.DataFrame(correlation_data)
        correlation_df.to_csv(f"{a.output_dir}/feature_clinical_correlations.csv")

    def _create_concatenated_correlation(self, common_mapping, feature_linkage, feature_order, a):
        """Create unified correlation heatmap by concatenating both datasets (common columns only)"""

        print("Loading and concatenating datasets...")

        # Load both datasets using the same approach as in _pre_common
        morph_data_dict = load_common_data('morph')
        patho2_data_dict = load_common_data('patho2')

        if morph_data_dict is None or patho2_data_dict is None:
            raise RuntimeError("Failed to load dataset(s). Check data availability.")

        return morph_data_dict['merged_data'], patho2_data_dict['merged_data']

    def _prepare_comparison_matrices(self, morph_corr, patho2_corr, clinical_mapping):
        """Prepare data matrices for comparison analysis"""
        # Create comprehensive variable list combining both datasets
        all_variables = []

        # Add common variables
        for morph_var, patho2_var in clinical_mapping.items():
            if morph_var in morph_corr.columns and patho2_var in patho2_corr.columns:
                all_variables.append((morph_var, patho2_var, 'common'))

        # Add Morph-only variables
        morph_only = ['OS', 'PFS', 'Follow-up Status', 'Age', 'LDH', 'ECOG PS',
                     'Stage', 'IPI Score', 'IPI Risk Group (4 Class)', 'RIPI Risk Group']
        for var in morph_only:
            if var in morph_corr.columns:
                all_variables.append((var, None, 'morph_only'))

        # Add Patho2-only variables
        patho2_only = ['EBV']
        for var in patho2_only:
            if var in patho2_corr.columns:
                all_variables.append((None, var, 'patho2_only'))

        # Prepare data matrices
        morph_clinical_vars = []
        patho2_clinical_vars = []
        variable_labels = []

        for morph_var, patho2_var, var_type in all_variables:
            if var_type == 'common':
                morph_clinical_vars.append(morph_var)
                patho2_clinical_vars.append(patho2_var)
                variable_labels.append(morph_var)
            elif var_type == 'morph_only':
                morph_clinical_vars.append(morph_var)
                patho2_clinical_vars.append(None)
                variable_labels.append(f"{morph_var}\n(Morph only)")
            elif var_type == 'patho2_only':
                morph_clinical_vars.append(None)
                patho2_clinical_vars.append(patho2_var)
                variable_labels.append(f"{patho2_var}\n(Patho2 only)")

        # Create data matrices handling None values
        morph_data = {}
        patho2_data = {}

        for i, (morph_var, patho2_var) in enumerate(zip(morph_clinical_vars, patho2_clinical_vars)):
            col_name = f"col_{i}"

            if morph_var:
                morph_data[col_name] = morph_corr[morph_var]
            else:
                morph_data[col_name] = pd.Series([np.nan] * len(morph_corr), index=morph_corr.index)

            if patho2_var:
                patho2_data[col_name] = patho2_corr[patho2_var]
            else:
                patho2_data[col_name] = pd.Series([np.nan] * len(patho2_corr), index=patho2_corr.index)

        morph_subset = pd.DataFrame(morph_data)
        patho2_subset = pd.DataFrame(patho2_data)

        return morph_subset, patho2_subset, variable_labels

    def _create_unified_dendrogram(self, morph_subset, patho2_subset):
        """Create unified dendrogram using common variables from both datasets"""
        from scipy.cluster.hierarchy import linkage, leaves_list
        from scipy.spatial.distance import pdist

        # Use common variables for fair dendrogram creation
        n_common = len([col for col in morph_subset.columns if not morph_subset[col].isna().all()])

        # Use both datasets' common variables for more robust clustering
        morph_common_only = morph_subset.iloc[:, :n_common].fillna(0)
        patho2_common_only = patho2_subset.iloc[:, :n_common].fillna(0)

        # Average correlation patterns from both datasets for fairness
        combined_common = (morph_common_only + patho2_common_only) / 2

        feature_distance = pdist(combined_common.values, metric='euclidean')
        feature_linkage = linkage(feature_distance, method='ward')
        feature_order = leaves_list(feature_linkage)

        return feature_linkage, feature_order

    def _compute_correlation_matrix(self, merged_data, correlation_method='pearson'):
        """Compute correlation matrix between features and clinical variables"""

        # Get clinical columns (maintaining original order)
        clinical_cols = [col for col in merged_data.columns if not col.startswith('feature_') and col != 'patient_id']
        feature_cols = [col for col in merged_data.columns if col.startswith('feature_')]

        print(f"Computing correlations: {len(feature_cols)} features x {len(clinical_cols)} clinical variables")

        # Create correlation matrix
        correlation_matrix = np.full((len(feature_cols), len(clinical_cols)), np.nan)

        for i, feature_col in enumerate(feature_cols):
            for j, clinical_col in enumerate(clinical_cols):
                feature_data = merged_data[feature_col]
                clinical_data = merged_data[clinical_col]

                valid_mask = ~(feature_data.isna() | clinical_data.isna())

                if valid_mask.sum() >= 10:  # Minimum 10 samples for correlation
                    feature_valid = feature_data[valid_mask]
                    clinical_valid = clinical_data[valid_mask]

                    try:
                        if correlation_method == 'pearson':
                            corr, _ = pearsonr(feature_valid, clinical_valid)
                        else:
                            corr, _ = spearmanr(feature_valid, clinical_valid)
                        correlation_matrix[i, j] = corr
                    except:
                        pass  # Keep as NaN if correlation fails

        # Convert to DataFrame
        corr_df = pd.DataFrame(correlation_matrix,
                              index=feature_cols,
                              columns=clinical_cols)

        return corr_df

    def _create_concatenated_correlation(self, common_mapping, feature_linkage, feature_order, a):
        """Create unified correlation heatmap by concatenating both datasets (common columns only)"""

        print("Loading and concatenating datasets...")

        # Load both datasets using the same approach as in _pre_common
        morph_data_dict = load_common_data('morph')
        if morph_data_dict is None:
            print("Warning: Could not load morph data")
            return
        morph_merged = morph_data_dict['merged_data']

        patho2_data_dict = load_common_data('patho2')
        if patho2_data_dict is None:
            print("Warning: Could not load patho2 data")
            return
        patho2_merged = patho2_data_dict['merged_data']

        print(f"Morph data shape: {morph_merged.shape}")
        print(f"Patho2 data shape: {patho2_merged.shape}")

        # Extract common columns only
        common_clinical_cols = list(common_mapping.keys())  # Use morph naming as standard
        feature_cols = [col for col in morph_merged.columns if col.startswith('feature_')]

        # Map patho2 column names to morph names for consistency
        patho2_renamed = patho2_merged.copy()
        for morph_col, patho2_col in common_mapping.items():
            if patho2_col in patho2_merged.columns and morph_col != patho2_col:
                patho2_renamed[morph_col] = patho2_merged[patho2_col]
                patho2_renamed.drop(patho2_col, axis=1, inplace=True)

        # Select common columns + features for concatenation
        common_cols = feature_cols + common_clinical_cols
        morph_subset = morph_merged[common_cols]
        patho2_subset = patho2_renamed[common_cols]

        # Concatenate datasets
        concatenated_data = pd.concat([morph_subset, patho2_subset], ignore_index=True)
        print(f"Concatenated data shape: {concatenated_data.shape}")
        print(f"Total samples: {len(concatenated_data)} (morph: {len(morph_subset)}, patho2: {len(patho2_subset)})")

        # Compute correlation matrix for concatenated data
        print("Computing unified correlation matrix...")
        concat_corr_matrix = np.full((len(feature_cols), len(common_clinical_cols)), np.nan)

        for i, feature_col in enumerate(feature_cols):
            for j, clinical_col in enumerate(common_clinical_cols):
                feature_data = concatenated_data[feature_col]
                clinical_data = concatenated_data[clinical_col]

                valid_mask = ~(feature_data.isna() | clinical_data.isna())

                if valid_mask.sum() >= 10:  # Minimum 10 samples for correlation
                    feature_valid = feature_data[valid_mask]
                    clinical_valid = clinical_data[valid_mask]

                    try:
                        if a.correlation_method == 'pearson':
                            corr, _ = pearsonr(feature_valid, clinical_valid)
                        else:
                            corr, _ = spearmanr(feature_valid, clinical_valid)
                        concat_corr_matrix[i, j] = corr
                    except:
                        pass  # Keep as NaN if correlation fails

        # Create DataFrame
        concat_corr_df = pd.DataFrame(concat_corr_matrix,
                                     index=feature_cols,
                                     columns=common_clinical_cols)

        # Apply the same feature ordering from dendrogram
        concat_ordered = concat_corr_df.iloc[feature_order]

        print("Creating concatenated correlation heatmap...")

        # Create figure with dendrogram
        fig = plt.figure(figsize=(14, a.figsize_height))

        # Grid: dendrogram | heatmap | colorbar
        gs = fig.add_gridspec(1, 3, width_ratios=[0.15, 0.75, 0.1],
                             hspace=0.02, wspace=0.1)

        # Plot dendrogram
        ax_dendro = fig.add_subplot(gs[0, 0])
        dendro = dendrogram(feature_linkage, ax=ax_dendro, orientation='left',
                           no_labels=True, color_threshold=0)
        ax_dendro.set_xticks([])
        ax_dendro.set_yticks([])
        ax_dendro.set_title('Dendrogram', fontsize=12)
        for spine in ax_dendro.spines.values():
            spine.set_visible(False)

        # Main heatmap
        ax_heatmap = fig.add_subplot(gs[0, 1])
        im = ax_heatmap.imshow(concat_ordered.values, cmap=plt.cm.RdBu_r,
                              aspect='auto', vmin=-1, vmax=1, interpolation='nearest')

        # Overlay gray for missing values
        mask = concat_ordered.isna()
        import matplotlib.patches as patches
        for i in range(len(concat_ordered)):
            for j in range(len(common_clinical_cols)):
                if mask.iloc[i, j]:
                    ax_heatmap.add_patch(patches.Rectangle((j-0.5, i-0.5), 1, 1,
                                                          facecolor='lightgray', edgecolor='darkgray', alpha=0.8))

        ax_heatmap.set_title(f'Concatenated Unified Correlation\n(Total: {len(concatenated_data)} samples)', fontsize=14)
        ax_heatmap.set_xticks(range(len(common_clinical_cols)))
        ax_heatmap.set_xticklabels(common_clinical_cols, rotation=45, ha='right', fontsize=10)
        ax_heatmap.set_yticks(range(0, len(concat_ordered), 50))
        ax_heatmap.set_yticklabels([f"F{feature_order[i]}" for i in range(0, len(concat_ordered), 50)], fontsize=8)
        ax_heatmap.set_ylabel('Feature Index (Dendrogram Order)')

        # Add colorbar
        ax_colorbar = fig.add_subplot(gs[0, 2])
        cbar = plt.colorbar(im, cax=ax_colorbar)
        cbar.set_label(f'{a.correlation_method.capitalize()}\nCorrelation', rotation=270, labelpad=20)

        plt.suptitle(f'Unified Feature-Clinical Correlation (Concatenated Data)\n' +
                    f'Morph + Patho2 Combined Analysis', fontsize=16, y=0.95)

        # Save plot
        plt.savefig(f"{a.output_dir}/unified_correlation_concatenated.png", dpi=300, bbox_inches='tight')
        plt.savefig(f"{a.output_dir}/unified_correlation_concatenated.pdf", dpi=300, bbox_inches='tight')
        plt.close()

        # Save correlation matrix
        concat_corr_df.to_csv(f"{a.output_dir}/concatenated_correlation_matrix.csv")

        print(f"Concatenated correlation analysis completed!")
        print(f"Results saved to: {a.output_dir}/unified_correlation_concatenated.png")
        print(f"Correlation matrix saved to: {a.output_dir}/concatenated_correlation_matrix.csv")


if __name__ == '__main__':
    cli = CLI()
    cli.run()
