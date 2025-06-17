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

    def run_gather_features(self, a):
        data = []
        slide_features = []
        center_features = []
        medoid_features = []
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
                with h5py.File(h5_path, 'r') as f:
                    slide_features.append(f['gigapath/slide_feature'][:])
                    features = f['gigapath/features'][:]
                    # 中心
                    center_feature = np.mean(features, axis=0)
                    # 中心最近傍
                    medoid_idx = np.argmin(np.linalg.norm(features - center_feature, axis=1))
                    medoid_feature = features[medoid_idx]
                    center_features.append(center_feature)
                    medoid_features.append(medoid_feature)

                data.append({
                    'name': name,
                    'order': i,
                    'filename': os.path.basename(h5_path),
                })

        df = pd.DataFrame(data)
        slide_features = np.array(slide_features)
        center_features = np.array(center_features)
        medoid_features = np.array(medoid_features)
        print('slide_features', slide_features.shape)
        print('center_features', center_features.shape)
        print('medoid_features', medoid_features.shape)

        o = str(self.dataset_dir / 'global_features.h5')
        with h5py.File(o, 'w') as f:
            f.create_dataset('gigapath/slide_features', data=slide_features)
            f.create_dataset('gigapath/center_features', data=center_features)
            f.create_dataset('gigapath/medoid_features', data=medoid_features)
            f.create_dataset('names', data=df['name'].values)
            f.create_dataset('orders', data=df['order'].values)
            f.create_dataset('filenames', data=df['filename'].values)
        print(f'wrote {o}')

    class LeidenArgs(CommonArgs):
        resolution: float = param(0.5, description="Leiden clustering resolution")
        n_neighbors: int = param(15, description="Number of neighbors for UMAP")
        min_dist: float = param(0.1, description="Minimum distance for UMAP")
        save_clusters: bool = param(True, description="Save cluster results to CSV")
        feature: str = param('slide', choices=['slide', 'center', 'medoid'])
        noshow: bool = False

    def run_leiden(self, a: LeidenArgs):
        """Leiden clustering (pure clustering without visualization)"""

        # Load features
        with h5py.File(str(self.dataset_dir / 'global_features.h5'), 'r') as f:
            features = f[f'{a.feature}_features'][:]
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
            clusters_file = str(self.dataset_dir / f'leiden_clusters_{a.feature}_with_ids.csv')
            results_df.set_index('sample_id').to_csv(clusters_file)
            print(f'Saved cluster results to: {clusters_file}')

        print(f'Leiden clustering completed with {len(set(clusters))} clusters')
        print(f'Use "run_visualize --target leiden_cluster" to visualize results')

    class VisualizeArgs(CommonArgs):
        target: str = param('leiden_cluster', description="Target to visualize (leiden_cluster, clinical variables, etc.)")
        feature: str = param('slide', choices=['slide', 'center', 'medoid'], description="Feature type to use for visualization")
        n_neighbors: int = param(10, description="Number of neighbors for UMAP")
        min_dist: float = param(0.05, description="Minimum distance for UMAP")
        metric: str = param('cosine', choices=['cosine', 'euclidean', 'manhattan'], description="Distance metric for UMAP")
        noshow: bool = param(False, description="Skip showing plots")

    def run_visualize(self, a: VisualizeArgs):
        """Unified visualization for any target using UMAP embedding"""

        # Combat補正を使用する場合は、統合データを使用
        if self.use_combat:
            # Combat補正データを使用（self.featuresに既に格納されている）
            features = self.features
            # patient_idsとdfを構築
            patient_names = self.patient_ids
            df = pd.DataFrame({
                'name': patient_names,
            })
            print(f'Loaded Combat-corrected features: {features.shape}')
        else:
            # 従来の方法でfeatureファイルから読み込み
            with h5py.File(str(self.dataset_dir / 'global_features.h5'), 'r') as f:
                features = f[f'{a.feature}_features'][:]
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
            # Try feature-specific clustering first, then default
            leiden_files = [
                str(self.dataset_dir / f'leiden_clusters_{a.feature}_with_ids.csv'),
                str(self.dataset_dir / 'leiden_clusters_slide_with_ids.csv'),
                str(self.dataset_dir / 'leiden_clusters_with_ids.csv')
            ]

            leiden_file = None
            for file in leiden_files:
                if os.path.exists(file):
                    leiden_file = file
                    break

            if leiden_file:
                leiden_df = pd.read_csv(leiden_file, index_col=0)
                df['leiden_cluster'] = df.apply(
                    lambda row: leiden_df.loc[f"{row['name']}__{row['order']}", 'leiden_cluster']
                    if f"{row['name']}__{row['order']}" in leiden_df.index else -1, axis=1)
                print(f'Using clustering results from: {leiden_file}')
            else:
                raise FileNotFoundError(f'Leiden clustering results not found. Tried: {leiden_files}')

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

        # Save plot (Combat補正対応パス)
        output_dir = self.base_output_path / a.dataset / 'umap'
        output_dir.mkdir(parents=True, exist_ok=True)
        target_name = a.target.replace(' ', '_')
        plot_file = output_dir / f'umap_{target_name}.png'
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f'Saved plot to: {plot_file}')

        if not a.noshow:
            plt.show()
        else:
            plt.close()





    class CombatComparisonArgs(CommonArgs):
        target: str = param('dataset', description="Target to visualize for comparison")
        feature: str = param('slide', choices=['slide', 'center', 'medoid'], description="Feature type to use")
        n_neighbors: int = param(10, description="Number of neighbors for UMAP")
        min_dist: float = param(0.05, description="Minimum distance for UMAP")
        metric: str = param('cosine', choices=['cosine', 'euclidean', 'manhattan'], description="Distance metric for UMAP")

    def run_combat_comparison(self, a: CombatComparisonArgs):
        """Compare UMAP embeddings before and after Combat correction"""
        
        print("Combat補正前後のUMAP比較を開始...")
        
        # 両データセットの補正前データを読み込み
        morph_dict = load_common_data('morph')
        patho2_dict = load_common_data('patho2')
        
        if morph_dict is None or patho2_dict is None:
            raise RuntimeError("データセット読み込みに失敗しました")
        
        # 補正前データの結合
        morph_features_raw = morph_dict['features']
        patho2_features_raw = patho2_dict['features']
        combined_features_raw = np.vstack([morph_features_raw, patho2_features_raw])
        
        # Combat補正データの取得
        corrected_features = self._apply_combat_correction(morph_dict, patho2_dict)
        
        # データセットラベル作成
        n_morph = len(morph_dict['patient_ids'])
        n_patho2 = len(patho2_dict['patient_ids'])
        dataset_labels = ['Morph'] * n_morph + ['Patho2'] * n_patho2
        
        # 特徴量標準化
        scaler_raw = StandardScaler()
        scaled_features_raw = scaler_raw.fit_transform(combined_features_raw)
        
        scaler_corrected = StandardScaler()
        scaled_features_corrected = scaler_corrected.fit_transform(corrected_features)
        
        # UMAP埋め込み（補正前）
        print("UMAP埋め込み（補正前）...")
        reducer_raw = UMAP(
            n_neighbors=a.n_neighbors,
            min_dist=a.min_dist,
            n_components=2,
            metric=a.metric,
            random_state=a.seed,
            n_jobs=1,
        )
        embedding_raw = reducer_raw.fit_transform(scaled_features_raw)
        
        # UMAP埋め込み（補正後）
        print("UMAP埋め込み（補正後）...")
        reducer_corrected = UMAP(
            n_neighbors=a.n_neighbors,
            min_dist=a.min_dist,
            n_components=2,
            metric=a.metric,
            random_state=a.seed,
            n_jobs=1,
        )
        embedding_corrected = reducer_corrected.fit_transform(scaled_features_corrected)
        
        # 並列プロット作成
        fig, axes = plt.subplots(1, 2, figsize=(20, 8))
        
        colors = {'Morph': 'blue', 'Patho2': 'red'}
        
        # 補正前プロット
        for dataset in ['Morph', 'Patho2']:
            mask = np.array(dataset_labels) == dataset
            axes[0].scatter(
                embedding_raw[mask, 0], embedding_raw[mask, 1],
                c=colors[dataset], label=dataset, alpha=0.7, s=15
            )
        axes[0].set_title('UMAP: Before Combat Correction', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('UMAP Dimension 1')
        axes[0].set_ylabel('UMAP Dimension 2')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # 補正後プロット  
        for dataset in ['Morph', 'Patho2']:
            mask = np.array(dataset_labels) == dataset
            axes[1].scatter(
                embedding_corrected[mask, 0], embedding_corrected[mask, 1],
                c=colors[dataset], label=dataset, alpha=0.7, s=15
            )
        axes[1].set_title('UMAP: After Combat Correction', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('UMAP Dimension 1')
        axes[1].set_ylabel('UMAP Dimension 2')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存
        output_dir = self.base_output_path / 'comparison'
        output_dir.mkdir(parents=True, exist_ok=True)
        plot_file = output_dir / 'umap_combat_comparison.png'
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f'Combat比較プロットを保存: {plot_file}')
        
        # 分離度の定量評価
        self._evaluate_batch_separation(embedding_raw, embedding_corrected, dataset_labels, output_dir)
        
        plt.show()
        
    def _evaluate_batch_separation(self, embedding_raw, embedding_corrected, dataset_labels, output_dir):
        """バッチ効果の分離度を定量評価"""
        from sklearn.metrics import silhouette_score
        
        # データセットラベルを数値に変換
        label_mapping = {'Morph': 0, 'Patho2': 1}
        numeric_labels = [label_mapping[label] for label in dataset_labels]
        
        # シルエット係数計算（高いほどバッチ分離が強い = 悪い）
        sil_raw = silhouette_score(embedding_raw, numeric_labels)
        sil_corrected = silhouette_score(embedding_corrected, numeric_labels)
        
        print(f"\\n=== バッチ効果評価 ===")
        print(f"シルエット係数（補正前）: {sil_raw:.3f}")
        print(f"シルエット係数（補正後）: {sil_corrected:.3f}")
        print(f"改善度: {sil_raw - sil_corrected:.3f} ({'改善' if sil_corrected < sil_raw else '悪化'})")
        
        # 結果をCSVで保存
        evaluation_df = pd.DataFrame({
            'metric': ['silhouette_before', 'silhouette_after', 'improvement'],
            'value': [sil_raw, sil_corrected, sil_raw - sil_corrected]
        })
        
        eval_file = output_dir / 'combat_evaluation.csv'
        evaluation_df.to_csv(eval_file, index=False)
        print(f"評価結果を保存: {eval_file}")


if __name__ == '__main__':
    cli = CLI()
    cli.run()
