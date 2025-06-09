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
from .utils.data_loader import load_common_data

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
        """Leiden clustering (pure clustering without visualization)"""

        # Load features
        with h5py.File(str(self.dataset_dir / 'slide_features.h5'), 'r') as f:
            features = f['features'][:]
            names = f['names'][:]
            orders = f['orders'][:]

        print(f'Loaded features: {features.shape}')

        # Create AnnData object for scanpy
        adata = ad.AnnData(features)
        adata.obs['sample_id'] = [f'{name.decode()}__{order}' for name, order in zip(names, orders)]

        # Preprocessing
        sc.pp.neighbors(adata, n_neighbors=a.n_neighbors)

        # Leiden clustering
        sc.tl.leiden(adata, resolution=a.resolution, key_added='leiden_cluster')

        # Extract clustering results
        clusters = adata.obs['leiden_cluster'].astype(int).values

        # Create DataFrame with results
        results_df = pd.DataFrame({
            'sample_id': adata.obs['sample_id'],
            'leiden_cluster': clusters,
        })

        # Save cluster results
        if a.save_clusters:
            clusters_file = str(self.dataset_dir / 'leiden_clusters_with_ids.csv')
            results_df.set_index('sample_id').to_csv(clusters_file)
            print(f'Saved cluster results to: {clusters_file}')

        print(f'Leiden clustering completed with {len(set(clusters))} clusters')
        print(f'Use "run_visualize --target leiden_cluster" to visualize results')

    class VisualizeArgs(CommonArgs):
        target: str = param('leiden_cluster', description="Target to visualize (leiden_cluster, clinical variables, etc.)")
        n_neighbors: int = param(10, description="Number of neighbors for UMAP")
        min_dist: float = param(0.05, description="Minimum distance for UMAP")
        metric: str = param('cosine', choices=['cosine', 'euclidean', 'manhattan'], description="Distance metric for UMAP")
        noshow: bool = param(False, description="Skip showing plots")

    def run_visualize(self, a: VisualizeArgs):
        """Unified visualization for any target using UMAP embedding"""

        # Load features and metadata
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

        print(f'Loaded features: {features.shape}')

        # Merge with clinical data
        df_clinical = self.clinical_data.copy()
        if a.dataset == 'morph':
            df_clinical['patient_id'] = df_clinical['patient_id'].astype(int)
        df_clinical = df_clinical.set_index('patient_id')
        df = pd.merge(df, df_clinical, left_on='name', right_index=True, how='left')

        # Add leiden_cluster if target requests it
        if a.target == 'leiden_cluster':
            leiden_file = str(self.dataset_dir / 'leiden_clusters_with_ids.csv')
            if os.path.exists(leiden_file):
                leiden_df = pd.read_csv(leiden_file, index_col=0)
                df['leiden_cluster'] = df.apply(
                    lambda row: leiden_df.loc[f"{row['name']}__{row['order']}", 'leiden_cluster']
                    if f"{row['name']}__{row['order']}" in leiden_df.index else -1, axis=1)
            else:
                raise FileNotFoundError(f'Leiden clustering results not found: {leiden_file}')

        # Check if target exists
        if a.target not in df.columns:
            available_targets = [col for col in df.columns if col not in ['name', 'filename', 'order']]
            raise ValueError(f"Target '{a.target}' not found. Available targets: {available_targets}")

        # Feature scaling and UMAP embedding
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(features)

        print('Computing UMAP embedding...')
        reducer = UMAP(
            n_neighbors=a.n_neighbors,
            min_dist=a.min_dist,
            n_components=2,
            metric=a.metric,
            random_state=a.seed,
            n_jobs=1,
        )
        embedding = reducer.fit_transform(scaled_features)

        # Auto-detect visualization mode
        target_data = df[a.target]
        unique_values = target_data.dropna().unique()

        if len(unique_values) <= 20 and not np.issubdtype(target_data.dtype, np.floating):
            mode = 'categorical'
        else:
            mode = 'numeric'

        print(f'Visualizing {a.target} in {mode} mode')

        # Create visualization
        plt.figure(figsize=(10, 8))
        marker_size = 15

        if mode == 'categorical':
            labels = df[a.target].fillna(-1)
            cmap = plt.cm.tab20 if len(unique_values) <= 20 else plt.cm.viridis

            noise_mask = labels == -1
            valid_labels = sorted(list(set(labels[~noise_mask])))

            for i, label in enumerate(valid_labels):
                mask = labels == label
                color = cmap(i / len(valid_labels)) if len(valid_labels) > 1 else cmap(0)
                plt.scatter(
                    embedding[mask, 0], embedding[mask, 1], c=[color],
                    s=marker_size, label=f'{a.target} {label}', alpha=0.7
                )

            if np.any(noise_mask):
                plt.scatter(
                    embedding[noise_mask, 0], embedding[noise_mask, 1], c='gray',
                    s=marker_size, marker='x', label='Missing/NaN', alpha=0.7
                )

        else:  # numeric mode
            values = df[a.target]
            valid_mask = ~values.isna()

            if valid_mask.sum() > 0:
                norm = Normalize(vmin=values[valid_mask].min(), vmax=values[valid_mask].max())
                scatter = plt.scatter(
                    embedding[valid_mask, 0], embedding[valid_mask, 1],
                    c=values[valid_mask], s=marker_size, cmap=plt.cm.viridis,
                    norm=norm, alpha=0.7
                )
                cbar = plt.colorbar(scatter)
                cbar.set_label(a.target)

            # Plot missing values
            if (~valid_mask).sum() > 0:
                plt.scatter(
                    embedding[~valid_mask, 0], embedding[~valid_mask, 1],
                    c='gray', s=marker_size, marker='x', label='Missing/NaN', alpha=0.7
                )

        plt.title(f'UMAP Embedding: {a.target} ({a.dataset.upper()})')
        plt.xlabel('UMAP Dimension 1')
        plt.ylabel('UMAP Dimension 2')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()

        # Save plot
        output_dir = f'out/{a.dataset}/visualize'
        os.makedirs(output_dir, exist_ok=True)
        target_name = a.target.replace(' ', '_')
        plot_file = f'{output_dir}/umap_{target_name}.png'
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f'Saved plot to: {plot_file}')

        if not a.noshow:
            plt.show()
        else:
            plt.close()





if __name__ == '__main__':
    cli = CLI()
    cli.run()
