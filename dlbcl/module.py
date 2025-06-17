import os
import json
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import fcluster, dendrogram, linkage, leaves_list
from scipy.spatial.distance import pdist
from scipy.stats import pearsonr, spearmanr
from pydantic_autocli import param

from .utils import BaseMLCLI
from .utils.data_loader import load_common_data
from .utils.correlation import (
    load_both_datasets, get_clinical_mapping, compute_correlation_matrix,
    prepare_comparison_matrices, create_unified_dendrogram
)


class CLI(BaseMLCLI):
    class CommonArgs(BaseMLCLI.CommonArgs):
        pass

    class ModuleAnalysisArgs(CommonArgs):
        output_dir: str = param('', description="Output directory (defaults to out/module)")
        correlation_method: str = param('pearson', choices=['pearson', 'spearman'], description="Correlation method to use")
        figsize_width: int = param(24, description="Figure width in inches")
        figsize_height: int = param(12, description="Figure height in inches")

    def run_module_analysis(self, a: ModuleAnalysisArgs):
        """Analyze feature modules from dendrogram clusters and their biological meaning"""
        
        if not a.output_dir:
            a.output_dir = 'out/module'
        
        os.makedirs(a.output_dir, exist_ok=True)
        
        print("Feature module analysis...")
        
        # Load data for both datasets - using inherited method
        morph_merged, patho2_merged = load_both_datasets()
        
        print(f"Morph data: {morph_merged.shape}, Patho2 data: {patho2_merged.shape}")
        
        # Compute correlation matrices - using inherited method
        morph_corr = compute_correlation_matrix(morph_merged, a.correlation_method)
        patho2_corr = compute_correlation_matrix(patho2_merged, a.correlation_method)
        
        # Get clinical mapping and prepare data matrices - using inherited methods
        clinical_mapping = get_clinical_mapping(patho2_merged.columns)
        morph_subset, patho2_subset, variable_labels = prepare_comparison_matrices(
            morph_corr, patho2_corr, clinical_mapping)
        
        # Create dendrogram for feature ordering - using inherited method
        feature_linkage, feature_order = create_unified_dendrogram(morph_subset, patho2_subset)
        
        # Reorder data using dendrogram
        morph_ordered = morph_subset.iloc[feature_order]
        patho2_ordered = patho2_subset.iloc[feature_order]
        
        # Create feature modules from dendrogram clusters
        print("\n=== Feature Module Creation ===")
        self._create_feature_modules(feature_linkage, feature_order, morph_ordered, patho2_ordered,
                                   variable_labels, a)
        
        print(f"\nModule analysis completed! Results saved to {a.output_dir}")

    def _create_feature_modules(self, feature_linkage, feature_order, morph_ordered, patho2_ordered,
                               variable_labels, a):
        """Create feature modules from dendrogram clusters and analyze their biological meaning"""

        print("Creating feature modules from dendrogram clusters...")

        # Extract modules at different hierarchy levels
        module_configs = [
            {'n_clusters': 5, 'name': 'major'},
            {'n_clusters': 10, 'name': 'moderate'},
            {'n_clusters': 20, 'name': 'fine'}
        ]

        for config in module_configs:
            n_clusters = config['n_clusters']
            module_name = config['name']

            print(f"\n--- {module_name.upper()} MODULES ({n_clusters} clusters) ---")

            # Get cluster assignments
            clusters = fcluster(feature_linkage, n_clusters, criterion='maxclust')

            # Map back to original feature order
            cluster_mapping = {}
            for i, original_idx in enumerate(feature_order):
                cluster_mapping[original_idx] = clusters[i]

            # Analyze each module
            module_analysis = {}
            for cluster_id in range(1, n_clusters + 1):
                # Get features in this cluster
                cluster_features = [i for i, c in enumerate(clusters) if c == cluster_id]
                cluster_size = len(cluster_features)

                if cluster_size < 5:  # Skip tiny clusters
                    continue

                # Extract correlation patterns for this module
                morph_module = morph_ordered.iloc[cluster_features]
                patho2_module = patho2_ordered.iloc[cluster_features]

                # Calculate module statistics
                morph_means = morph_module.mean()
                patho2_means = patho2_module.mean()
                module_consistency = np.corrcoef(morph_means.fillna(0), patho2_means.fillna(0))[0,1]

                # Find dominant clinical associations
                combined_means = (morph_means + patho2_means) / 2
                combined_means = combined_means.fillna(0)

                # Get top positive and negative associations
                top_pos = combined_means.nlargest(3)
                top_neg = combined_means.nsmallest(3)

                module_analysis[cluster_id] = {
                    'size': cluster_size,
                    'consistency': module_consistency,
                    'top_positive': top_pos.to_dict(),
                    'top_negative': top_neg.to_dict(),
                    'feature_indices': cluster_features
                }

                print(f"Module {cluster_id}: {cluster_size} features, consistency={module_consistency:.3f}")
                print(f"  Top positive: {list(top_pos.index[:2])}")
                print(f"  Top negative: {list(top_neg.index[:2])}")

            # Save module analysis
            module_file = f"{a.output_dir}/feature_modules_{module_name}.json"

            # Convert numpy types to Python native types for JSON serialization
            serializable_analysis = {}
            for k, v in module_analysis.items():
                serializable_analysis[k] = {
                    'size': int(v['size']),
                    'consistency': float(v['consistency']) if not np.isnan(v['consistency']) else 0.0,
                    'top_positive': {str(kk): float(vv) for kk, vv in v['top_positive'].items()},
                    'top_negative': {str(kk): float(vv) for kk, vv in v['top_negative'].items()},
                    'feature_indices': [int(x) for x in v['feature_indices']]
                }

            with open(module_file, 'w') as f:
                json.dump(serializable_analysis, f, indent=2)
            print(f"Saved {module_name} module analysis to: {module_file}")

        # Create module-based reduced features for further analysis
        self._create_module_features(module_analysis, morph_ordered, patho2_ordered, variable_labels, a)

    def _create_module_features(self, module_analysis, morph_ordered, patho2_ordered, variable_labels, a):
        """Create reduced feature representation based on modules"""

        print("\n--- MODULE-BASED FEATURE REDUCTION ---")

        # Create module features by averaging within each module
        morph_module_features = {}
        patho2_module_features = {}

        for module_id, module_info in module_analysis.items():
            if module_info['size'] < 10:  # Only use substantial modules
                continue

            feature_indices = module_info['feature_indices']

            # Average features within module
            morph_module_avg = morph_ordered.iloc[feature_indices].mean()
            patho2_module_avg = patho2_ordered.iloc[feature_indices].mean()

            morph_module_features[f'module_{module_id}'] = morph_module_avg
            patho2_module_features[f'module_{module_id}'] = patho2_module_avg

        morph_modules_df = pd.DataFrame(morph_module_features).T
        patho2_modules_df = pd.DataFrame(patho2_module_features).T

        print(f"Created {len(morph_modules_df)} module features from {len(morph_ordered)} original features")

        # Save module features
        morph_modules_df.to_csv(f"{a.output_dir}/morph_module_features.csv")
        patho2_modules_df.to_csv(f"{a.output_dir}/patho2_module_features.csv")

        # Create module comparison heatmap
        self._plot_module_comparison(morph_modules_df, patho2_modules_df, variable_labels, a)

    def _plot_module_comparison(self, morph_modules_df, patho2_modules_df, variable_labels, a):
        """Create comparison plot for module features"""

        # Calculate dynamic color range based on actual data
        all_values = np.concatenate([morph_modules_df.values.flatten(), patho2_modules_df.values.flatten()])
        valid_values = all_values[~np.isnan(all_values)]
        vmin, vmax = np.percentile(valid_values, [5, 95])  # Use 5-95 percentile for better contrast

        # Make sure range is symmetric around 0 for better interpretation
        abs_max = max(abs(vmin), abs(vmax))
        vmin, vmax = -abs_max, abs_max

        print(f"Using dynamic color range: [{vmin:.3f}, {vmax:.3f}]")

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

        # Morph modules
        im1 = ax1.imshow(morph_modules_df.values, cmap=plt.cm.RdBu_r, aspect='auto',
                        vmin=vmin, vmax=vmax, interpolation='nearest')
        ax1.set_title('MORPH Module Features', fontsize=14)
        ax1.set_xticks(range(len(variable_labels)))
        ax1.set_xticklabels(variable_labels, rotation=45, ha='right', fontsize=9)
        ax1.set_yticks(range(len(morph_modules_df)))
        ax1.set_yticklabels(morph_modules_df.index, fontsize=9)

        # Patho2 modules
        im2 = ax2.imshow(patho2_modules_df.values, cmap=plt.cm.RdBu_r, aspect='auto',
                        vmin=vmin, vmax=vmax, interpolation='nearest')
        ax2.set_title('PATHO2 Module Features', fontsize=14)
        ax2.set_xticks(range(len(variable_labels)))
        ax2.set_xticklabels(variable_labels, rotation=45, ha='right', fontsize=9)
        ax2.set_yticks(range(len(patho2_modules_df)))
        ax2.set_yticklabels(patho2_modules_df.index, fontsize=9)

        # Difference
        diff_modules = morph_modules_df.values - patho2_modules_df.values
        diff_vmin, diff_vmax = np.nanpercentile(diff_modules, [5, 95])
        diff_abs_max = max(abs(diff_vmin), abs(diff_vmax))
        diff_vmin, diff_vmax = -diff_abs_max, diff_abs_max

        im3 = ax3.imshow(diff_modules, cmap=plt.cm.RdBu_r, aspect='auto',
                        vmin=diff_vmin, vmax=diff_vmax, interpolation='nearest')
        ax3.set_title('Module Differences\n(Morph - Patho2)', fontsize=14)
        ax3.set_xticks(range(len(variable_labels)))
        ax3.set_xticklabels(variable_labels, rotation=45, ha='right', fontsize=9)
        ax3.set_yticks(range(len(morph_modules_df)))
        ax3.set_yticklabels(morph_modules_df.index, fontsize=9)

        # Shared colorbar
        fig.subplots_adjust(right=0.9)
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        cbar = plt.colorbar(im3, cax=cbar_ax)
        cbar.set_label('Correlation', rotation=270, labelpad=20)

        plt.suptitle('ViT Feature Modules: Biological Pattern Abstraction', fontsize=16, y=0.98)
        plt.tight_layout()

        plt.savefig(f"{a.output_dir}/vit_module_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Module comparison plot saved to: {a.output_dir}/vit_module_analysis.png")


if __name__ == '__main__':
    cli = CLI()
    cli.run()