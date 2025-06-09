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

from .utils import BaseMLCLI, BaseMLArgs

warnings.filterwarnings('ignore', category=FutureWarning, message='.*force_all_finite.*')



class CLI(BaseMLCLI):
    class CommonArgs(BaseMLCLI.CommonArgs):
        pass

    def run_gather_slide_features(self, a):
        data = []
        features = []
        dirs = sorted(glob(str(self.dataset_dir / 'dataset/*')))
        assert len(dirs) > 0
        for dir in tqdm(dirs):
            name = os.path.basename(dir)
            for i, h5_path in enumerate(sorted(glob(f'{dir}/*.h5'))):
                h5_filename = Path(h5_path).name
                # Support both Morph (12345_0.h5) and Patho2 (20-0494_1_1.h5) patterns
                if a.dataset == 'morph':
                    m = re.match(r'^\d\d\d\d\d_\d\.h5$', h5_filename)
                else:  # patho2
                    m = re.match(r'^[\d\-]+_\d+_\d+\.h5$', h5_filename)

                if not m:
                    print('skip', h5_path)
                    continue
                print('loading', h5_path)
                with h5py.File(h5_path, 'r') as f:
                    features.append(f['gigapath/slide_feature'][:])
                data.append({
                    'name': name,
                    'order': i,
                    'filename': os.path.basename(h5_path),
                })

        df = pd.DataFrame(data)
        features = np.array(features)
        print('features', features.shape)

        o = str(self.dataset_dir / 'slide_features.h5')
        with h5py.File(o, 'w') as f:
            f.create_dataset('features', data=features)
            f.create_dataset('names', data=df['name'].values)
            f.create_dataset('orders', data=df['order'].values)
            f.create_dataset('filenames', data=df['filename'].values)
        print(f'wrote {o}')

    class LeidenArgs(CommonArgs):
        resolution: float = param(0.5, description="Leiden clustering resolution")
        n_neighbors: int = param(15, description="Number of neighbors for UMAP")
        min_dist: float = param(0.1, description="Minimum distance for UMAP")
        save_clusters: bool = param(True, description="Save cluster results to CSV")
        noshow: bool = False

    def run_leiden(self, a: LeidenArgs):
        """Leiden clustering with UMAP visualization"""

        # Load features
        with h5py.File(str(self.dataset_dir / 'slide_features.h5'), 'r') as f:
            features = f['features'][:]
            names = f['names'][:]
            orders = f['orders'][:]

        # Create AnnData object for scanpy
        adata = ad.AnnData(features)
        adata.obs['sample_id'] = [f'{name.decode()}__{order}' for name, order in zip(names, orders)]

        # Preprocessing
        sc.pp.neighbors(adata, n_neighbors=a.n_neighbors)

        # Leiden clustering
        sc.tl.leiden(adata, resolution=a.resolution, key_added='leiden_cluster')

        # UMAP embedding
        sc.tl.umap(adata, min_dist=a.min_dist)

        # Extract results
        clusters = adata.obs['leiden_cluster'].astype(int).values
        umap_coords = adata.obsm['X_umap']

        # Create DataFrame with results
        results_df = pd.DataFrame({
            'sample_id': adata.obs['sample_id'],
            'leiden_cluster': clusters,
            'umap_1': umap_coords[:, 0],
            'umap_2': umap_coords[:, 1]
        })

        # Save cluster results
        if a.save_clusters:
            clusters_file = str(self.dataset_dir / 'leiden_clusters_with_ids.csv')
            results_df.set_index('sample_id').to_csv(clusters_file)
            print(f'Saved cluster results to: {clusters_file}')

        # Visualization
        from .utils import plot_umap
        fig = plot_umap(umap_coords, clusters, title=f'Leiden Clustering (resolution={a.resolution})')

        # Save plot
        os.makedirs(f'out/{a.dataset}/leiden', exist_ok=True)
        plot_file = f'out/{a.dataset}/leiden/leiden_clustering_res{a.resolution}.png'
        fig.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f'Saved plot to: {plot_file}')

        if not a.noshow:
            plt.show()
        else:
            plt.close()

        return results_df

    class UMAPArgs(CommonArgs):
        keys: list[str] = param(['leiden_cluster'], description="Keys to visualize (use leiden_cluster for latest clustering)")
        cluster_file: str = param('', description="Cluster file to use (defaults to leiden_clusters_with_ids.csv)")

    def run_umap(self, a:UMAPArgs):
        # Use latest clustering results (prioritize Leiden over legacy DBSCAN from Murakami)
        if not a.cluster_file:
            leiden_file = str(self.dataset_dir / 'leiden_clusters_with_ids.csv')
            murakami_dbscan_file = str(self.dataset_dir / 'DBSCAN_clusters_with_ids.csv')  # Legacy Murakami results

            if os.path.exists(leiden_file):
                a.cluster_file = leiden_file
                print(f'Using Leiden clustering results: {leiden_file}')
            elif os.path.exists(murakami_dbscan_file):
                a.cluster_file = murakami_dbscan_file
                print(f'Using Murakami HDBSCAN clustering results: {murakami_dbscan_file}')
                print(f'Note: This preserves Murakami-san\'s results for reproducibility. Run leiden clustering for updated results.')
            else:
                raise FileNotFoundError(f'No clustering results found. Run leiden clustering first.')

        df = pd.read_csv(a.cluster_file, index_col=0)
        keys = a.keys
        if 'ALL' in a.keys:
            keys = list(df.columns)
        with h5py.File(str(self.dataset_dir / 'slide_features.h5'), 'r') as f:
            features = f['features'][()]
            names = f['names'][()]
            orders = f['orders'][()]
            data = []
            for name, order in zip(names, orders):
                label = f'{name.decode("utf-8")}_{order}'
                leiden_label = f'{name.decode("utf-8")}__{order}'  # Double underscore for Leiden format
                v = {}
                for key in keys:
                    try:
                        v[key] = df.loc[label, key]
                    except KeyError:
                        # Try leiden format
                        try:
                            v[key] = df.loc[leiden_label, key]
                        except KeyError:
                            v[key] = -1  # Default value if not found
                data.append(v)

        df_labels = pd.DataFrame(data)

        for key in keys:
            plt.close()
            labels = df_labels[key]
            umap = UMAP()
            embs = umap.fit_transform(features)
            unique_labels = sorted(np.unique(labels))
            cmap = plt.get_cmap('tab20')
            for i, label in enumerate(unique_labels):
                mask = labels == label
                if isinstance(label, (int, np.integer)):
                    c = 'gray' if label < 0 else cmap(label)
                else:
                    c = cmap(i)
                plt.scatter(embs[mask,0], embs[mask,1], c=c, label=f'{key} {label}')
            plt.legend()
            os.makedirs(f'out/{a.dataset}/umap', exist_ok=True)
            plt.savefig(f'./out/{a.dataset}/umap/umap_{key.replace(" ", "_")}.png')
        plt.show()



    class ClusterArgs(CommonArgs):
        target: str = Field('cluster', s='-T')
        noshow: bool = False

    def run_cluster(self, a):
        with h5py.File(str(self.dataset_dir / 'slide_features.h5'), 'r') as f:
            features = f['features'][:]
            # Handle different patient ID formats (numeric for morph, string for patho2)
            names_decoded = [v.decode('utf-8') for v in f['names'][:]]
            if a.dataset == 'morph':
                patient_names = [int(name) for name in names_decoded]
            else:  # patho2
                patient_names = names_decoded

            df = pd.DataFrame({
                'name': patient_names,
                'filename': [v.decode('utf-8') for v in f['filenames'][:]],
                'order': f['orders'][:],
            })

        # Use clinical data from shared data
        df_clinical = self.clinical_data.copy()
        if a.dataset == 'morph':
            df_clinical['patient_id'] = df_clinical['patient_id'].astype(int)
        # For patho2, patient_id is already string, no conversion needed
        df_clinical = df_clinical.set_index('patient_id')
        df = pd.merge(
            df,
            df_clinical,
            left_on='name',
            right_index=True,
            how='left'
        )

        print('Loaded features', features.shape)
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(features)
        # scaled_features = features

        print('UMAP fitting...')
        reducer = UMAP(
                n_neighbors=10,
                min_dist=0.05,
                n_components=2,
                metric='cosine',
                random_state=a.seed,
                n_jobs=1,
            )
        embedding = reducer.fit_transform(scaled_features)
        print('Loaded features', features.shape)

        # Add leiden_cluster if available
        if a.target == 'leiden_cluster':
            leiden_file = str(self.dataset_dir / 'leiden_clusters_with_ids.csv')
            if os.path.exists(leiden_file):
                leiden_df = pd.read_csv(leiden_file, index_col=0)
                # Map leiden clusters to current df
                df['leiden_cluster'] = df.apply(lambda row: leiden_df.loc[f"{row['name']}__{row['order']}", 'leiden_cluster']
                                               if f"{row['name']}__{row['order']}" in leiden_df.index else -1, axis=1)
            else:
                raise FileNotFoundError(f'Leiden clustering results not found: {leiden_file}')

        if a.target in [
                'HDBSCAN', 'leiden_cluster',
                'CD10 IHC', 'MUM1 IHC', 'HANS', 'BCL6 FISH', 'MYC FISH', 'BCL2 FISH',
                'CD10', 'MUM1', 'BCL2', 'BCL6', 'MYC', 'EBV',  # patho2 columns
                'ECOG PS', 'LDH', 'EN', 'Stage', 'IPI Score',
                'IPI Risk Group (4 Class)', 'RIPI Risk Group', 'Follow-up Status',
                ]:
            mode = 'categorical'
        elif a.target in ['MYC IHC', 'BCL2 IHC', 'BCL6 IHC', 'Age', 'OS', 'PFS']:
            mode = 'numeric'
        else:
            raise RuntimeError('invalid target', a.target)


        plt.figure(figsize=(10, 8))
        marker_size = 15

        if mode == 'categorical':
            labels = df[a.target].fillna(-1)
            n_labels = len(set(labels))
            cmap = plt.cm.viridis

            noise_mask = labels == -1
            valid_labels = sorted(list(set(labels[~noise_mask])))
            norm = plt.Normalize(min(valid_labels or [0]), max(valid_labels or [1]))
            for label in valid_labels:
                mask = labels == label
                color = cmap(norm(label))
                plt.scatter(
                    embedding[mask, 0], embedding[mask, 1], c=[color],
                    s=marker_size, label=f'{a.target} {label}'
                )

            if np.any(noise_mask):
                plt.scatter(
                    embedding[noise_mask, 0], embedding[noise_mask, 1], c='gray',
                    s=marker_size, marker='x', label='Noise/NaN',
                )

        else:
            values = df[a.target]
            norm = Normalize(vmin=values.min(), vmax=values.max())
            values = values.fillna(-1)
            has_value = values > 0
            cmap = plt.cm.viridis
            scatter = plt.scatter(embedding[has_value, 0], embedding[has_value, 1], c=values[has_value],
                                  s=marker_size, cmap=cmap, norm=norm, label=a.target,)
            if np.any(has_value):
                plt.scatter(embedding[~has_value, 0], embedding[~has_value, 1], c='gray',
                            s=marker_size, marker='x', label='NaN')
            cbar = plt.colorbar(scatter)
            cbar.set_label(a.target)

        plt.title(f'UMAP + {a.target}')
        plt.xlabel('UMAP Dimension 1')
        plt.ylabel('UMAP Dimension 2')
        # plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.legend()
        plt.tight_layout()
        os.makedirs(f'out/{a.dataset}/cluster', exist_ok=True)
        name = a.target.replace(' ', '_')
        plt.savefig(f'out/{a.dataset}/cluster/umap_{name}.png')
        if not a.noshow:
            plt.show()



    class ComprehensiveHeatmapArgs(CommonArgs):
        output_dir: str = param('', description="Directory to save comprehensive heatmap results (defaults to out/{dataset}/comprehensive_heatmap)")
        correlation_method: str = param('pearson', choices=['pearson', 'spearman'], description="Correlation method to use")
        cluster_method: str = param('ward', choices=['ward', 'complete', 'average', 'single'], description="Clustering method for features")
        figsize_width: int = param(20, description="Figure width in inches")
        figsize_height: int = param(16, description="Figure height in inches")

    def run_comprehensive_heatmap(self, a: ComprehensiveHeatmapArgs):
        """Comprehensive heatmap analysis of all features and clinical data (with dendrogram)"""

        if not a.output_dir:
            a.output_dir = f'out/{a.dataset}/comprehensive_heatmap'

        os.makedirs(a.output_dir, exist_ok=True)

        # Use shared data
        df = self.merged_data
        print(f"Using merged dataset shape: {df.shape}")

        # Get clinical columns (maintaining original order)
        clinical_cols = [col for col in self.clinical_data.columns if col != 'patient_id']
        feature_cols = [col for col in df.columns if col.startswith('feature_')]

        print(f"Number of features: {len(feature_cols)}")
        print(f"Number of clinical variables: {len(clinical_cols)}")

        # Create correlation matrix
        print("Computing correlation matrix...")
        correlation_matrix = np.full((len(feature_cols), len(clinical_cols)), np.nan)

        for i, feature_col in enumerate(feature_cols):
            for j, clinical_col in enumerate(clinical_cols):
                feature_data = df[feature_col]
                clinical_data = df[clinical_col]

                valid_mask = ~(feature_data.isna() | clinical_data.isna())

                if valid_mask.sum() >= 10:  # Minimum 10 samples for correlation
                    feature_valid = feature_data[valid_mask]
                    clinical_valid = clinical_data[valid_mask]

                    try:
                        if a.correlation_method == 'pearson':
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

        # Save correlation matrix
        corr_df.to_csv(f"{a.output_dir}/correlation_matrix.csv")

        # Hierarchical clustering for features (rows)
        print("Performing hierarchical clustering on features...")

        # Calculate distance matrix for features (handle NaN values)
        feature_corr_valid = corr_df.fillna(0)  # Fill NaN with 0 for clustering
        feature_distance = pdist(feature_corr_valid.values, metric='euclidean')
        feature_linkage = linkage(feature_distance, method=a.cluster_method)

        # Get the order of features after clustering
        feature_order = leaves_list(feature_linkage)
        ordered_features = [feature_cols[i] for i in feature_order]

        # Reorder correlation matrix
        corr_df_ordered = corr_df.reindex(ordered_features)

        # Create the comprehensive heatmap with dendrogram
        print("Creating comprehensive heatmap...")

        # Create figure with subplots
        fig = plt.figure(figsize=(a.figsize_width, a.figsize_height))

        # Create grid for dendrogram and heatmap (dendrogram on left)
        gs = fig.add_gridspec(1, 3,
                             width_ratios=[0.2, 0.7, 0.1],
                             hspace=0.02, wspace=0.02)

        # Plot dendrogram on left
        ax_dendro = fig.add_subplot(gs[0, 0])
        dendro = dendrogram(feature_linkage, ax=ax_dendro, orientation='left',
                           no_labels=True, color_threshold=0)
        ax_dendro.set_xticks([])
        ax_dendro.set_yticks([])
        for spine in ax_dendro.spines.values():
            spine.set_visible(False)

        # Main heatmap in center
        ax_heatmap = fig.add_subplot(gs[0, 1])

        # Base colormap for correlations
        base_cmap = plt.cm.RdBu_r

        # Create mask for missing values
        mask = corr_df_ordered.isna()

        # Plot heatmap
        im = ax_heatmap.imshow(corr_df_ordered.values,
                              cmap=base_cmap,
                              aspect='auto',
                              vmin=-1, vmax=1)

        # Overlay gray for missing values
        for i in range(len(ordered_features)):
            for j in range(len(clinical_cols)):
                if mask.iloc[i, j]:
                    ax_heatmap.add_patch(patches.Rectangle((j-0.5, i-0.5), 1, 1,
                                                          facecolor='lightgray', edgecolor='none'))

        # Set ticks and labels
        ax_heatmap.set_xticks(range(len(clinical_cols)))
        ax_heatmap.set_xticklabels(clinical_cols, rotation=45, ha='right')
        ax_heatmap.set_yticks(range(0, len(ordered_features), 50))  # Show every 50th feature
        ax_heatmap.set_yticklabels([f"F{i}" for i in range(0, len(ordered_features), 50)], fontsize=8)

        # Add colorbar on right
        ax_colorbar = fig.add_subplot(gs[0, 2])
        cbar = plt.colorbar(im, cax=ax_colorbar)
        cbar.set_label(f'{a.correlation_method.capitalize()} Correlation', rotation=270, labelpad=20)

        # Add title
        fig.suptitle(f'Comprehensive Feature-Clinical Correlation Heatmap\n'
                    f'Dataset: {a.dataset.upper()}, Method: {a.correlation_method.capitalize()}, '
                    f'Clustering: {a.cluster_method}', fontsize=14, y=0.98)

        # Save the plot
        plt.savefig(f"{a.output_dir}/comprehensive_heatmap.png", dpi=300, bbox_inches='tight')
        plt.savefig(f"{a.output_dir}/comprehensive_heatmap.pdf", dpi=300, bbox_inches='tight')
        plt.close()

        # Create significant correlations overlay if statistical results exist
        print("Creating significant correlations overlay...")

        stats_file = f"out/{a.dataset}/statistics/correlation_analysis_corrected.csv"
        if os.path.exists(stats_file):
            stats_df = pd.read_csv(stats_file)

            # Create significance mask
            sig_mask = np.zeros_like(corr_df_ordered.values, dtype=bool)

            for _, row in stats_df.iterrows():
                if row['pearson_p_corrected'] < 0.05:  # Significant after correction
                    try:
                        feature_idx = ordered_features.index(row['feature'])
                        clinical_idx = clinical_cols.index(row['clinical_variable'])
                        sig_mask[feature_idx, clinical_idx] = True
                    except ValueError:
                        continue  # Skip if feature/clinical variable not found

            # Create figure with significance overlay
            fig, ax = plt.subplots(figsize=(a.figsize_width, a.figsize_height * 0.8))

            # Plot base heatmap
            im = ax.imshow(corr_df_ordered.values,
                          cmap=base_cmap,
                          aspect='auto',
                          vmin=-1, vmax=1)

            # Overlay gray for missing values and highlight significant correlations
            for i in range(len(ordered_features)):
                for j in range(len(clinical_cols)):
                    if mask.iloc[i, j]:
                        ax.add_patch(patches.Rectangle((j-0.5, i-0.5), 1, 1,
                                                      facecolor='lightgray', edgecolor='none'))
                    elif sig_mask[i, j]:
                        ax.add_patch(patches.Rectangle((j-0.5, i-0.5), 1, 1,
                                                      facecolor='none', edgecolor='black', linewidth=2))

            # Set labels
            ax.set_xticks(range(len(clinical_cols)))
            ax.set_xticklabels(clinical_cols, rotation=45, ha='right')
            ax.set_yticks(range(0, len(ordered_features), 50))  # Show every 50th feature
            ax.set_yticklabels([f"F{i}" for i in range(0, len(ordered_features), 50)])

            # Add colorbar
            cbar = plt.colorbar(im, ax=ax, shrink=0.8)
            cbar.set_label(f'{a.correlation_method.capitalize()} Correlation', rotation=270, labelpad=20)

            plt.title(f'Significant Correlations Highlighted (FDR < 0.05)\n'
                     f'Dataset: {a.dataset.upper()}, Black borders indicate significance', fontsize=14)

            plt.tight_layout()
            plt.savefig(f"{a.output_dir}/significant_correlations_heatmap.png", dpi=300, bbox_inches='tight')
            plt.close()

        # Generate summary statistics
        summary = {
            'dataset': a.dataset,
            'n_features': len(feature_cols),
            'n_clinical_vars': len(clinical_cols),
            'correlation_method': a.correlation_method,
            'clustering_method': a.cluster_method,
            'total_correlations_computed': np.sum(~np.isnan(correlation_matrix)),
            'missing_correlations': np.sum(np.isnan(correlation_matrix)),
            'mean_abs_correlation': np.nanmean(np.abs(correlation_matrix)),
            'max_correlation': np.nanmax(correlation_matrix),
            'min_correlation': np.nanmin(correlation_matrix)
        }

        with open(f"{a.output_dir}/summary.txt", 'w') as f:
            for key, value in summary.items():
                f.write(f"{key}: {value}\n")

        print(f"Comprehensive heatmap analysis completed. Results saved in {a.output_dir}")
        for key, value in summary.items():
            print(f"{key}: {value}")

        return corr_df_ordered

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

        return survival_df

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

    class CompareCorrelationArgs(CommonArgs):
        output_dir: str = param('', description="Output directory (defaults to out/compare_correlation)")
        correlation_method: str = param('pearson', choices=['pearson', 'spearman'], description="Correlation method to use")
        figsize_width: int = param(24, description="Figure width in inches")
        figsize_height: int = param(12, description="Figure height in inches")

    def run_compare_correlation(self, a: CompareCorrelationArgs):
        """Compare correlation heatmaps between Morph and Patho2 datasets with unified feature ordering"""
        
        if not a.output_dir:
            a.output_dir = 'out/compare_correlation'
        
        os.makedirs(a.output_dir, exist_ok=True)
        
        print("Loading correlation matrices from both datasets...")
        
        # Load correlation matrices from both datasets
        morph_corr_file = "out/morph/comprehensive_heatmap/correlation_matrix.csv"
        patho2_corr_file = "out/patho2/comprehensive_heatmap/correlation_matrix.csv"
        
        if not (os.path.exists(morph_corr_file) and os.path.exists(patho2_corr_file)):
            raise FileNotFoundError("Please run comprehensive-heatmap for both datasets first")
        
        morph_corr = pd.read_csv(morph_corr_file, index_col=0)
        patho2_corr = pd.read_csv(patho2_corr_file, index_col=0)
        
        print(f"Morph correlation matrix: {morph_corr.shape}")
        print(f"Patho2 correlation matrix: {patho2_corr.shape}")
        
        # Create comprehensive variable list combining both datasets
        all_variables = []
        
        # Check which naming convention Patho2 uses in the correlation matrix
        patho2_columns = list(patho2_corr.columns)
        print(f"Patho2 columns in correlation matrix: {patho2_columns}")
        
        # Determine mapping based on actual column names
        if 'CD10 IHC' in patho2_columns:
            # New naming convention
            common_mapping = {
                'CD10 IHC': 'CD10 IHC',
                'MUM1 IHC': 'MUM1 IHC', 
                'BCL2 IHC': 'BCL2 IHC',
                'BCL6 IHC': 'BCL6 IHC',
                'MYC IHC': 'MYC IHC',
                'HANS': 'HANS'
            }
        else:
            # Old naming convention (short names)
            common_mapping = {
                'CD10 IHC': 'CD10',
                'MUM1 IHC': 'MUM1', 
                'BCL2 IHC': 'BCL2',
                'BCL6 IHC': 'BCL6',
                'MYC IHC': 'MYC',
                'HANS': 'HANS'
            }
        
        # Add common variables
        for morph_var, patho2_var in common_mapping.items():
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
        
        print(f"Total variables to display: {len(all_variables)}")
        for i, (m, p, t) in enumerate(all_variables):
            print(f"  {i+1}. {m} <-> {p} ({t})")
        
        # Prepare data matrices
        morph_clinical_vars = []
        patho2_clinical_vars = []
        variable_labels = []
        
        for morph_var, patho2_var, var_type in all_variables:
            if var_type == 'common':
                morph_clinical_vars.append(morph_var)
                patho2_clinical_vars.append(patho2_var)
                # Use Morph naming for display (longer, more descriptive)
                display_name = morph_var
                variable_labels.append(display_name)
            elif var_type == 'morph_only':
                morph_clinical_vars.append(morph_var)
                patho2_clinical_vars.append(None)
                variable_labels.append(f"{morph_var}\n(Morph only)")
            elif var_type == 'patho2_only':
                morph_clinical_vars.append(None)
                patho2_clinical_vars.append(patho2_var)
                variable_labels.append(f"{patho2_var}\n(Patho2 only)")
        
        print(f"Mapped clinical variables: {len(morph_clinical_vars)}")
        print(f"Morph variables: {morph_clinical_vars}")
        print(f"Patho2 variables: {patho2_clinical_vars}")
        
        if len(morph_clinical_vars) == 0:
            raise ValueError("No mapped clinical variables found between datasets")
        
        # Create data matrices handling None values
        morph_data = {}
        patho2_data = {}
        
        for i, (morph_var, patho2_var) in enumerate(zip(morph_clinical_vars, patho2_clinical_vars)):
            col_name = f"col_{i}"  # Use index as column name
            
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
        
        # Use Morph's dendrogram ordering (from comprehensive_heatmap)
        print("Loading feature ordering from Morph dendrogram...")
        
        # Calculate hierarchical clustering using the same method as comprehensive_heatmap
        from scipy.cluster.hierarchy import linkage, leaves_list
        from scipy.spatial.distance import pdist
        
        # Use ONLY common variables for fair dendrogram creation
        print("Creating dendrogram using common variables only...")
        
        # Extract only common variables (first 6 variables are common)
        n_common = len([v for v in all_variables if v[2] == 'common'])
        print(f"Using {n_common} common variables for dendrogram")
        
        # Use both datasets' common variables for more robust clustering
        morph_common_only = morph_subset.iloc[:, :n_common].fillna(0)
        patho2_common_only = patho2_subset.iloc[:, :n_common].fillna(0)
        
        # Average correlation patterns from both datasets for fairness
        combined_common = (morph_common_only + patho2_common_only) / 2
        
        feature_distance = pdist(combined_common.values, metric='euclidean')
        feature_linkage = linkage(feature_distance, method='ward')
        feature_order = leaves_list(feature_linkage)
        
        print(f"Dendrogram created using combined patterns from {n_common} common variables")
        
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
        
        # Advanced analysis: Create feature modules from dendrogram clusters
        print("\n=== Advanced Analysis: Feature Module Creation ===")
        self._create_feature_modules(feature_linkage, feature_order, morph_ordered, patho2_ordered, 
                                   variable_labels, a)
        
        # Return None to avoid warning about unexpected return type
        return None

    def _create_feature_modules(self, feature_linkage, feature_order, morph_ordered, patho2_ordered, 
                               variable_labels, a):
        """Create feature modules from dendrogram clusters and analyze their biological meaning"""
        from scipy.cluster.hierarchy import fcluster
        
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
            import json
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
        self._create_module_features(module_analysis, morph_ordered, patho2_ordered, a)
    
    def _create_module_features(self, module_analysis, morph_ordered, patho2_ordered, a):
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
        self._plot_module_comparison(morph_modules_df, patho2_modules_df, a)
    
    def _plot_module_comparison(self, morph_modules_df, patho2_modules_df, a):
        """Create comparison plot for module features"""
        
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
        
        # Morph modules
        im1 = ax1.imshow(morph_modules_df.values, cmap=plt.cm.RdBu_r, aspect='auto', 
                        vmin=-1, vmax=1, interpolation='nearest')
        ax1.set_title('MORPH Module Features', fontsize=14)
        ax1.set_xticks(range(len(morph_modules_df.columns)))
        ax1.set_xticklabels(morph_modules_df.columns, rotation=45, ha='right', fontsize=9)
        ax1.set_yticks(range(len(morph_modules_df)))
        ax1.set_yticklabels(morph_modules_df.index, fontsize=9)
        
        # Patho2 modules
        im2 = ax2.imshow(patho2_modules_df.values, cmap=plt.cm.RdBu_r, aspect='auto',
                        vmin=-1, vmax=1, interpolation='nearest')
        ax2.set_title('PATHO2 Module Features', fontsize=14)
        ax2.set_xticks(range(len(patho2_modules_df.columns)))
        ax2.set_xticklabels(patho2_modules_df.columns, rotation=45, ha='right', fontsize=9)
        ax2.set_yticks(range(len(patho2_modules_df)))
        ax2.set_yticklabels(patho2_modules_df.index, fontsize=9)
        
        # Difference
        diff_modules = morph_modules_df.values - patho2_modules_df.values
        im3 = ax3.imshow(diff_modules, cmap=plt.cm.RdBu_r, aspect='auto',
                        vmin=-1, vmax=1, interpolation='nearest')
        ax3.set_title('Module Differences\n(Morph - Patho2)', fontsize=14)
        ax3.set_xticks(range(len(morph_modules_df.columns)))
        ax3.set_xticklabels(morph_modules_df.columns, rotation=45, ha='right', fontsize=9)
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

