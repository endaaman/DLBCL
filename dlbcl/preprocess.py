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





if __name__ == '__main__':
    cli = CLI()
    cli.run()
