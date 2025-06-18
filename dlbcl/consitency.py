
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

    class FeatureConsistencyArgs(CommonArgs):
        output_dir: str = param('', description="Output directory (defaults to out/{dataset}/feature_consistency)")
        target_features: list[str] = param(['CD10 IHC', 'MUM1 IHC', 'BCL2 IHC', 'BCL6 IHC', 'HANS'], description="Clinical features to analyze")
        top_n: int = param(50, description="Number of top correlating features to analyze")
        correlation_threshold: float = param(0.15, description="Minimum correlation threshold for analysis")

    def run_feature_consistency(self, a: FeatureConsistencyArgs):
        """Analyze consistency of feature channels across datasets"""

        if not a.output_dir:
            a.output_dir = f'out/{a.dataset}/feature_consistency'

        os.makedirs(a.output_dir, exist_ok=True)

        print(f"Analyzing feature consistency for dataset: {a.dataset}")

        # Use merged data
        df = self.merged_data
        feature_cols = [col for col in df.columns if col.startswith('feature_')]

        # Results storage
        results = {}

        for target_feature in a.target_features:
            if target_feature not in df.columns:
                print(f"Warning: {target_feature} not found in {a.dataset} dataset")
                continue

            print(f"\nAnalyzing {target_feature}...")

            # Calculate correlations
            correlations = []
            for feature_col in feature_cols:
                valid_mask = ~(df[feature_col].isna() | df[target_feature].isna())
                if valid_mask.sum() >= 10:
                    try:
                        corr, p_val = pearsonr(df[feature_col][valid_mask], df[target_feature][valid_mask])
                        correlations.append({
                            'feature_channel': int(feature_col.replace('feature_', '')),
                            'correlation': corr,
                            'p_value': p_val,
                            'abs_correlation': abs(corr)
                        })
                    except:
                        continue

            # Sort by absolute correlation
            correlations = sorted(correlations, key=lambda x: x['abs_correlation'], reverse=True)

            # Filter by threshold and take top N
            significant_corrs = [c for c in correlations if c['abs_correlation'] >= a.correlation_threshold]
            top_corrs = significant_corrs[:a.top_n]

            if top_corrs:
                results[target_feature] = top_corrs
                print(f"Found {len(top_corrs)} significant correlations (|r| >= {a.correlation_threshold})")
                print(f"Top 5 correlations:")
                for i, corr_info in enumerate(top_corrs[:5]):
                    print(f"  {i+1}. Channel {corr_info['feature_channel']}: r={corr_info['correlation']:.3f}, p={corr_info['p_value']:.3e}")
            else:
                print(f"No significant correlations found for {target_feature}")

        # Save results
        import json
        results_file = f"{a.output_dir}/feature_correlations_{a.dataset}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {results_file}")

        # Create visualization
        self._create_feature_consistency_plots(results, a)

        return results

    def _create_feature_consistency_plots(self, results, a):
        """Create visualization plots for feature consistency analysis"""

        if not results:
            return

        # Create heatmap of top correlating channels
        fig, axes = plt.subplots(len(results), 1, figsize=(12, 4 * len(results)))
        if len(results) == 1:
            axes = [axes]

        for idx, (target_feature, correlations) in enumerate(results.items()):
            ax = axes[idx]

            if correlations:
                channels = [c['feature_channel'] for c in correlations[:20]]  # Top 20
                corr_values = [c['correlation'] for c in correlations[:20]]

                # Create bar plot
                colors = ['red' if x < 0 else 'blue' for x in corr_values]
                bars = ax.bar(range(len(channels)), corr_values, color=colors, alpha=0.7)

                # Customize plot
                ax.set_title(f'{target_feature} - Top Correlating Feature Channels ({a.dataset})', fontsize=12)
                ax.set_xlabel('Rank')
                ax.set_ylabel('Correlation Coefficient')
                ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
                ax.grid(True, alpha=0.3)

                # Add channel numbers as x-tick labels
                ax.set_xticks(range(len(channels)))
                ax.set_xticklabels([f'Ch{ch}' for ch in channels], rotation=45)

                # Add correlation values on bars
                for bar, corr_val in zip(bars, corr_values):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + (0.01 if height >= 0 else -0.01),
                           f'{corr_val:.2f}', ha='center', va='bottom' if height >= 0 else 'top', fontsize=8)
            else:
                ax.text(0.5, 0.5, f'No significant correlations\nfound for {target_feature}',
                       ha='center', va='center', transform=ax.transAxes, fontsize=12)
                ax.set_title(f'{target_feature} ({a.dataset})', fontsize=12)

        plt.tight_layout()
        plot_file = f"{a.output_dir}/feature_correlations_{a.dataset}.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Visualization saved to: {plot_file}")

    class ChannelConsistencyArgs(CommonArgs):
        output_dir: str = param('', description="Output directory (defaults to out/channel_consistency)")
        correlation_threshold: float = param(0.15, description="Minimum correlation threshold for analysis")
        consistency_threshold: float = param(0.1, description="Threshold for considering correlations consistent across datasets")
        top_n: int = param(100, description="Number of top correlating channels to analyze per target")

    def run_channel_consistency(self, a: ChannelConsistencyArgs):
        """統計的にチャンネル一貫性を評価し、共通傾向を持つチャンネル数をカウント"""

        if not a.output_dir:
            a.output_dir = 'out/channel_consistency'

        os.makedirs(a.output_dir, exist_ok=True)

        print("Analyzing channel consistency across morph and patho2 datasets...")

        # Load results from both datasets
        morph_file = 'out/morph/feature_consistency/feature_correlations_morph.json'
        patho2_file = 'out/patho2/feature_consistency/feature_correlations_patho2.json'

        if not (os.path.exists(morph_file) and os.path.exists(patho2_file)):
            raise FileNotFoundError("Please run feature-consistency for both datasets first")

        import json
        with open(morph_file, 'r') as f:
            morph_results = json.load(f)
        with open(patho2_file, 'r') as f:
            patho2_results = json.load(f)

        # Map morph IHC names to patho2 names
        feature_mapping = {
            'CD10 IHC': 'CD10 IHC',
            'MUM1 IHC': 'MUM1 IHC',
            'BCL2 IHC': 'BCL2 IHC',
            'BCL6 IHC': 'BCL6 IHC',
            'HANS': 'HANS'
        }

        consistency_results = {}
        all_channels = set(range(768))  # All possible channels

        for morph_feature, patho2_feature in feature_mapping.items():
            if morph_feature not in morph_results or patho2_feature not in patho2_results:
                continue

            print(f"\nAnalyzing {morph_feature} (morph) vs {patho2_feature} (patho2)...")

            # Get top correlating channels from both datasets
            morph_channels = {item['feature_channel']: item['correlation']
                             for item in morph_results[morph_feature][:a.top_n]}
            patho2_channels = {item['feature_channel']: item['correlation']
                              for item in patho2_results[patho2_feature][:a.top_n]}

            # Find overlapping channels
            common_channels = set(morph_channels.keys()) & set(patho2_channels.keys())

            # Analyze consistency
            consistent_positive = 0  # Same positive direction
            consistent_negative = 0  # Same negative direction
            inconsistent = 0         # Opposite directions

            consistent_channels = []
            inconsistent_channels = []

            for channel in common_channels:
                morph_corr = morph_channels[channel]
                patho2_corr = patho2_channels[channel]

                # Check if correlations are in same direction and both above threshold
                if (abs(morph_corr) >= a.correlation_threshold and
                    abs(patho2_corr) >= a.correlation_threshold):

                    if morph_corr * patho2_corr > 0:  # Same sign
                        if abs(abs(morph_corr) - abs(patho2_corr)) <= a.consistency_threshold:
                            if morph_corr > 0:
                                consistent_positive += 1
                            else:
                                consistent_negative += 1
                            consistent_channels.append({
                                'channel': channel,
                                'morph_corr': morph_corr,
                                'patho2_corr': patho2_corr,
                                'diff': abs(abs(morph_corr) - abs(patho2_corr))
                            })
                    else:  # Opposite signs
                        inconsistent += 1
                        inconsistent_channels.append({
                            'channel': channel,
                            'morph_corr': morph_corr,
                            'patho2_corr': patho2_corr
                        })

            total_consistent = consistent_positive + consistent_negative
            consistency_rate = total_consistent / len(common_channels) if common_channels else 0

            consistency_results[f"{morph_feature}_vs_{patho2_feature}"] = {
                'total_common_channels': len(common_channels),
                'consistent_positive': consistent_positive,
                'consistent_negative': consistent_negative,
                'total_consistent': total_consistent,
                'inconsistent': inconsistent,
                'consistency_rate': consistency_rate,
                'consistent_channels': sorted(consistent_channels, key=lambda x: x['diff']),
                'inconsistent_channels': inconsistent_channels
            }

            print(f"  Common channels: {len(common_channels)}")
            print(f"  Consistent (same direction): {total_consistent} ({consistency_rate:.3f})")
            print(f"    - Positive correlations: {consistent_positive}")
            print(f"    - Negative correlations: {consistent_negative}")
            print(f"  Inconsistent (opposite direction): {inconsistent}")

            if consistent_channels:
                print(f"  Top 3 most consistent channels:")
                for i, ch in enumerate(consistent_channels[:3]):
                    print(f"    {i+1}. Channel {ch['channel']}: morph={ch['morph_corr']:.3f}, patho2={ch['patho2_corr']:.3f}, diff={ch['diff']:.3f}")

        # Save detailed results
        results_file = f"{a.output_dir}/channel_consistency_analysis.json"
        with open(results_file, 'w') as f:
            json.dump(consistency_results, f, indent=2)
        print(f"\nDetailed results saved to: {results_file}")

        # Create visualization
        self._create_consistency_plots(consistency_results, a)

        # Generate summary statistics
        self._generate_consistency_summary(consistency_results, a)

        return consistency_results

    def _create_consistency_plots(self, results, a):
        """Create visualization for channel consistency analysis"""

        features = list(results.keys())
        if not features:
            return

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # Plot 1: Consistency rates
        ax1 = axes[0, 0]
        consistency_rates = [results[f]['consistency_rate'] for f in features]
        ax1.bar(range(len(features)), consistency_rates, alpha=0.7, color='skyblue')
        ax1.set_title('Channel Consistency Rates Across Datasets')
        ax1.set_ylabel('Consistency Rate')
        ax1.set_xticks(range(len(features)))
        ax1.set_xticklabels([f.replace('_vs_', '\nvs\n') for f in features], rotation=0, fontsize=9)
        ax1.grid(True, alpha=0.3)

        # Plot 2: Common channels count
        ax2 = axes[0, 1]
        common_counts = [results[f]['total_common_channels'] for f in features]
        ax2.bar(range(len(features)), common_counts, alpha=0.7, color='lightgreen')
        ax2.set_title('Number of Common Significant Channels')
        ax2.set_ylabel('Common Channels Count')
        ax2.set_xticks(range(len(features)))
        ax2.set_xticklabels([f.replace('_vs_', '\nvs\n') for f in features], rotation=0, fontsize=9)
        ax2.grid(True, alpha=0.3)

        # Plot 3: Consistent vs Inconsistent breakdown
        ax3 = axes[1, 0]
        consistent_counts = [results[f]['total_consistent'] for f in features]
        inconsistent_counts = [results[f]['inconsistent'] for f in features]

        x = range(len(features))
        width = 0.35
        ax3.bar([i - width/2 for i in x], consistent_counts, width, label='Consistent', alpha=0.7, color='green')
        ax3.bar([i + width/2 for i in x], inconsistent_counts, width, label='Inconsistent', alpha=0.7, color='red')
        ax3.set_title('Consistent vs Inconsistent Channels')
        ax3.set_ylabel('Channel Count')
        ax3.set_xticks(x)
        ax3.set_xticklabels([f.replace('_vs_', '\nvs\n') for f in features], rotation=0, fontsize=9)
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # Plot 4: Positive vs Negative consistent correlations
        ax4 = axes[1, 1]
        positive_counts = [results[f]['consistent_positive'] for f in features]
        negative_counts = [results[f]['consistent_negative'] for f in features]

        ax4.bar([i - width/2 for i in x], positive_counts, width, label='Positive Corr', alpha=0.7, color='blue')
        ax4.bar([i + width/2 for i in x], negative_counts, width, label='Negative Corr', alpha=0.7, color='orange')
        ax4.set_title('Direction of Consistent Correlations')
        ax4.set_ylabel('Channel Count')
        ax4.set_xticks(x)
        ax4.set_xticklabels([f.replace('_vs_', '\nvs\n') for f in features], rotation=0, fontsize=9)
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plot_file = f"{a.output_dir}/channel_consistency_analysis.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Consistency visualization saved to: {plot_file}")

    def _generate_consistency_summary(self, results, a):
        """Generate summary report of consistency analysis"""

        summary_file = f"{a.output_dir}/consistency_summary.txt"

        with open(summary_file, 'w') as f:
            f.write("=== CHANNEL CONSISTENCY ANALYSIS SUMMARY ===\n\n")
            f.write(f"Analysis Parameters:\n")
            f.write(f"- Correlation threshold: {a.correlation_threshold}\n")
            f.write(f"- Consistency threshold: {a.consistency_threshold}\n")
            f.write(f"- Top N channels analyzed: {a.top_n}\n\n")

            # Overall statistics
            total_common = sum(results[f]['total_common_channels'] for f in results)
            total_consistent = sum(results[f]['total_consistent'] for f in results)
            overall_rate = total_consistent / total_common if total_common > 0 else 0

            f.write(f"Overall Statistics:\n")
            f.write(f"- Total common channels across all comparisons: {total_common}\n")
            f.write(f"- Total consistent channels: {total_consistent}\n")
            f.write(f"- Overall consistency rate: {overall_rate:.3f}\n\n")

            # Per-feature analysis
            for feature, data in results.items():
                f.write(f"{feature}:\n")
                f.write(f"  Common channels: {data['total_common_channels']}\n")
                f.write(f"  Consistent: {data['total_consistent']} ({data['consistency_rate']:.3f})\n")
                f.write(f"  Inconsistent: {data['inconsistent']}\n")

                if data['consistent_channels']:
                    f.write(f"  Top consistent channels:\n")
                    for i, ch in enumerate(data['consistent_channels'][:5]):
                        f.write(f"    {i+1}. Ch{ch['channel']}: morph={ch['morph_corr']:.3f}, patho2={ch['patho2_corr']:.3f}\n")
                f.write("\n")

        print(f"Summary report saved to: {summary_file}")


if __name__ == '__main__':
    cli = CLI()
    cli.run()
