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
import igraph as ig
import leidenalg
import torch
import timm
from umap import UMAP
from gigapath import slide_encoder
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
import matplotlib.patches as patches
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test

from .utils import BaseMLCLI, BaseMLArgs

warnings.filterwarnings('ignore', category=FutureWarning, message='.*force_all_finite.*')



class CLI(BaseMLCLI):
    class CommonArgs(BaseMLArgs):
        # This includes `--seed` param
        device: str = 'cuda'
        dataset: str = param('morph', choices=['morph', 'patho2'])

    def pre_common(self, a:CommonArgs):
        # データセットディレクトリ設定
        if a.dataset == 'morph':
            self.dataset_dir = Path('./data/DLBCL-Morph/')
        elif a.dataset == 'patho2':
            self.dataset_dir = Path('./data/DLBCL-Patho2/')
        else:
            raise ValueError('Invalid dataset', a.dataset)
        
        # 統計解析・特徴量解析で使用する共通データの読み込み
        self._load_common_data(a.dataset)
    
    def _load_common_data(self, dataset: str):
        """共通データの読み込み"""
        try:
            # 臨床データ読み込み
            if dataset == 'patho2':
                self.clinical_data = pd.read_csv(f'data/DLBCL-{dataset.capitalize()}/clinical_data_cleaned_path2.csv')
            else:
                self.clinical_data = pd.read_csv(f'data/DLBCL-{dataset.capitalize()}/clinical_data_cleaned.csv')
            print(f"Clinical data loaded: {len(self.clinical_data)} patients")
            
            # 特徴量データ読み込み
            with h5py.File(f'data/DLBCL-{dataset.capitalize()}/slide_features.h5', 'r') as f:
                self.features = f['features'][:]
                self.feature_names = f['names'][:]
                
            # patient_IDの抽出
            self.patient_ids = [name.decode().split('_')[0] for name in self.feature_names]
            
            # 特徴量データフレーム作成
            self.feature_df = pd.DataFrame(self.features, columns=[f'feature_{i}' for i in range(self.features.shape[1])])
            self.feature_df['patient_id'] = self.patient_ids
            
            # 臨床データの patient_id を文字列に変換
            self.clinical_data['patient_id'] = self.clinical_data['patient_id'].astype(str)
            
            # データマージ
            self.merged_data = self.feature_df.merge(self.clinical_data, on='patient_id', how='inner')
            print(f"Merged data: {len(self.merged_data)} samples")
            
            # CD10のバイナリ化（0 or 1のみ）
            if 'CD10 IHC' in self.merged_data.columns:
                self.merged_data['CD10_binary'] = (self.merged_data['CD10 IHC'] > 0).astype(int)
                
        except Exception as e:
            print(f"Warning: Failed to load common data: {e}")
            # 一部のコマンド（既存のクラスタリングなど）では共通データが不要な場合があるため、
            # エラーは警告に留めて処理を続行

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

    class UMAPArgs(CommonArgs):
        keys: list[str] = param(['DBSCAN cluster'])

    def run_umap(self, a:UMAPArgs):
        df  = pd.read_csv('./data/DLBCL-Morph/DBSCAN_clusters_with_ids.csv', index_col=0)
        keys = a.keys
        if 'ALL' in a.keys:
            keys = list(df.columns)
        with h5py.File('./data/DLBCL-Morph/slide_features.h5', 'r') as f:
            features = f['features'][()]
            names = f['names'][()]
            orders = f['orders'][()]
            data = []
            for name, order in zip(names, orders):
                label = f'{name.decode("utf-8")}_{order}'
                v = {}
                for key in keys:
                    v[key] = df.loc[label, key]
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
            plt.savefig(f'./out/umap_{key.replace(" ", "_")}.png')
        plt.show()



    class ClusterArgs(CommonArgs):
        target: str = Field('cluster', s='-T')
        noshow: bool = False
        algorithm: str = param('leiden', choices=['hdbscan', 'leiden'])
        min_cluster_size: int = 5
        resolution: float = 0.5  # For Leiden clustering

    def run_cluster(self, a):
        # Use the pre-loaded merged data from _load_common_data
        if not hasattr(self, 'merged_data'):
            raise RuntimeError("Common data not loaded. Check if dataset files exist.")
        
        # Get features and merged data, ensuring they match
        merged_data = self.merged_data
        feature_cols = [col for col in merged_data.columns if col.startswith('feature_')]
        features = merged_data[feature_cols].values

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

        # Perform actual clustering if target is 'cluster'
        if a.target == 'cluster':
            print(f'Performing {a.algorithm.upper()} clustering...')
            
            if a.algorithm == 'hdbscan':
                clusterer = hdbscan.HDBSCAN(min_cluster_size=a.min_cluster_size)
                cluster_labels = clusterer.fit_predict(scaled_features)
                
            elif a.algorithm == 'leiden':
                # Build k-nearest neighbor graph
                from sklearn.neighbors import kneighbors_graph
                k = min(15, len(scaled_features) - 1)  # Ensure k < n_samples
                print(f'Building k-NN graph with k={k}...')
                
                # Create adjacency matrix
                adj_matrix = kneighbors_graph(scaled_features, n_neighbors=k, mode='connectivity')
                
                # Convert to igraph
                sources, targets = adj_matrix.nonzero()
                edges = list(zip(sources.tolist(), targets.tolist()))
                g = ig.Graph(edges=edges, directed=False)
                g.simplify()  # Remove multiple edges and self-loops
                
                # Leiden clustering
                print(f'Running Leiden clustering with resolution={a.resolution}...')
                partition = leidenalg.find_partition(g, leidenalg.RBConfigurationVertexPartition, 
                                                   resolution_parameter=a.resolution)
                cluster_labels = np.array(partition.membership)
                
            else:
                raise ValueError(f"Unknown algorithm: {a.algorithm}")
            
            # Evaluate clustering
            from sklearn.metrics import silhouette_score
            if len(set(cluster_labels)) > 1:  # Need at least 2 clusters for silhouette score
                try:
                    silhouette_avg = silhouette_score(scaled_features, cluster_labels)
                    print(f'Silhouette Score: {silhouette_avg:.3f}')
                except:
                    print('Could not compute silhouette score')
            
            n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
            n_noise = list(cluster_labels).count(-1)
            print(f'Number of clusters: {n_clusters}')
            print(f'Number of noise points: {n_noise}')
            
            # Save clustering results
            cluster_df = merged_data.copy()
            cluster_df['cluster'] = cluster_labels
            os.makedirs(f'out/{a.dataset}', exist_ok=True)
            cluster_df.to_csv(f'out/{a.dataset}/{a.algorithm}_clustering_results.csv', index=False)
            
            # Visualize clusters
            mode = 'categorical'
            labels = cluster_labels
        elif a.target in [
                'HDBSCAN',
                'CD10 IHC', 'MUM1 IHC', 'HANS', 'BCL6 FISH', 'MYC FISH', 'BCL2 FISH',
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
            if a.target == 'cluster':
                # labels already defined from clustering
                pass
            else:
                labels = merged_data[a.target].fillna(-1)
            n_labels = len(set(labels))
            cmap = plt.cm.viridis

            noise_mask = labels == -1
            valid_labels = sorted(list(set(labels[~noise_mask])))
            norm = plt.Normalize(min(valid_labels or [0]), max(valid_labels or [1]))
            for label in valid_labels:
                mask = labels == label
                color = cmap(norm(label))
                target_name = 'Cluster' if a.target == 'cluster' else a.target
                plt.scatter(
                    embedding[mask, 0], embedding[mask, 1], c=[color],
                    s=marker_size, label=f'{target_name} {label}'
                )

            if np.any(noise_mask):
                plt.scatter(
                    embedding[noise_mask, 0], embedding[noise_mask, 1], c='gray',
                    s=marker_size, marker='x', label='Noise/NaN',
                )

        else:
            if a.target == 'cluster':
                # This shouldn't happen as cluster is always categorical
                raise RuntimeError("Cluster should be categorical mode")
            values = merged_data[a.target]
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

        title = f'UMAP + {"HDBSCAN Clusters" if a.target == "cluster" else a.target}'
        plt.title(title)
        plt.xlabel('UMAP Dimension 1')
        plt.ylabel('UMAP Dimension 2')
        # plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.legend()
        plt.tight_layout()
        os.makedirs(f'out/{a.dataset}', exist_ok=True)
        name = a.target.replace(' ', '_')
        plt.savefig(f'out/{a.dataset}/umap_{name}.png')
        if not a.noshow:
            plt.show()

    class IntegratedClusterArgs(CommonArgs):
        noshow: bool = False
        min_cluster_size: int = 5
        
    def run_integrated_cluster(self, a):
        """統合クラスタリング: MorphとPatho2のデータを結合してクラスタリング"""
        print("Loading Morph dataset...")
        
        # Morphデータ読み込み
        with h5py.File('data/DLBCL-Morph/slide_features.h5', 'r') as f:
            morph_features = f['features'][:]
            morph_names = f['names'][:]
        
        morph_clinical = pd.read_csv('data/DLBCL-Morph/clinical_data_cleaned.csv')
        morph_patient_ids = [name.decode().split('_')[0] for name in morph_names]
        morph_df = pd.DataFrame(morph_features, columns=[f'feature_{i}' for i in range(morph_features.shape[1])])
        morph_df['patient_id'] = morph_patient_ids
        morph_df['dataset'] = 'Morph'
        morph_clinical['patient_id'] = morph_clinical['patient_id'].astype(str)
        morph_merged = morph_df.merge(morph_clinical, on='patient_id', how='inner')
        
        print("Loading Patho2 dataset...")
        
        # Patho2データ読み込み
        with h5py.File('data/DLBCL-Patho2/slide_features.h5', 'r') as f:
            patho2_features = f['features'][:]
            patho2_names = f['names'][:]
        
        patho2_clinical = pd.read_csv('data/DLBCL-Patho2/clinical_data_cleaned_path2.csv')
        patho2_patient_ids = [name.decode().split('_')[0] for name in patho2_names]
        patho2_df = pd.DataFrame(patho2_features, columns=[f'feature_{i}' for i in range(patho2_features.shape[1])])
        patho2_df['patient_id'] = patho2_patient_ids
        patho2_df['dataset'] = 'Patho2'
        patho2_clinical['patient_id'] = patho2_clinical['patient_id'].astype(str)
        patho2_merged = patho2_df.merge(patho2_clinical, on='patient_id', how='inner')
        
        # 共通の特徴量カラムのみ結合
        feature_cols = [col for col in morph_merged.columns if col.startswith('feature_')]
        
        # データセット結合
        combined_features = np.vstack([
            morph_merged[feature_cols].values,
            patho2_merged[feature_cols].values
        ])
        
        combined_labels = np.hstack([
            np.zeros(len(morph_merged)),  # Morph = 0
            np.ones(len(patho2_merged))   # Patho2 = 1
        ])
        
        print(f"Combined features shape: {combined_features.shape}")
        print(f"Morph samples: {len(morph_merged)}, Patho2 samples: {len(patho2_merged)}")
        
        # 特徴量正規化
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(combined_features)
        
        # UMAP次元削減
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
        
        # HDBSCANクラスタリング
        print('Performing HDBSCAN clustering...')
        clusterer = hdbscan.HDBSCAN(min_cluster_size=a.min_cluster_size)
        cluster_labels = clusterer.fit_predict(scaled_features)
        
        # 評価
        from sklearn.metrics import silhouette_score
        if len(set(cluster_labels)) > 1:
            try:
                silhouette_avg = silhouette_score(scaled_features, cluster_labels)
                print(f'Silhouette Score: {silhouette_avg:.3f}')
            except:
                print('Could not compute silhouette score')
        
        n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        n_noise = list(cluster_labels).count(-1)
        print(f'Number of clusters: {n_clusters}')
        print(f'Number of noise points: {n_noise}')
        
        # 結果保存
        combined_df = pd.concat([morph_merged, patho2_merged], ignore_index=True)
        combined_df['cluster'] = cluster_labels
        os.makedirs('out/integrated', exist_ok=True)
        combined_df.to_csv('out/integrated/integrated_clustering_results.csv', index=False)
        
        # 可視化
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # データセット別の色分け
        colors_dataset = ['blue', 'red']
        dataset_names = ['Morph', 'Patho2']
        for i, (color, dataset_name) in enumerate(zip(colors_dataset, dataset_names)):
            mask = combined_labels == i
            axes[0].scatter(embedding[mask, 0], embedding[mask, 1], 
                          c=color, s=15, alpha=0.7, label=dataset_name)
        axes[0].set_title('UMAP: Dataset Distribution')
        axes[0].set_xlabel('UMAP Dimension 1')
        axes[0].set_ylabel('UMAP Dimension 2')
        axes[0].legend()
        
        # クラスター別の色分け
        unique_clusters = sorted(set(cluster_labels))
        cmap = plt.cm.viridis
        for cluster_id in unique_clusters:
            mask = cluster_labels == cluster_id
            if cluster_id == -1:
                axes[1].scatter(embedding[mask, 0], embedding[mask, 1], 
                              c='gray', s=15, marker='x', alpha=0.7, label='Noise')
            else:
                color = cmap(cluster_id / max(1, max(unique_clusters)))
                axes[1].scatter(embedding[mask, 0], embedding[mask, 1], 
                              c=[color], s=15, alpha=0.7, label=f'Cluster {cluster_id}')
        axes[1].set_title('UMAP: HDBSCAN Clusters')
        axes[1].set_xlabel('UMAP Dimension 1')
        axes[1].set_ylabel('UMAP Dimension 2')
        axes[1].legend()
        
        plt.tight_layout()
        plt.savefig('out/integrated/integrated_clustering.png', dpi=300, bbox_inches='tight')
        if not a.noshow:
            plt.show()
        
        # データセット別クラスター分析
        print("\n=== Dataset vs Cluster Analysis ===")
        cluster_dataset_table = pd.crosstab(combined_df['dataset'], combined_df['cluster'], margins=True)
        print(cluster_dataset_table)
        
        return combined_df

    class SurvivalAnalysisArgs(CommonArgs):
        noshow: bool = False
        survival_type: str = Field('both', choices=['OS', 'PFS', 'both'])
        cluster_based: bool = False
        
    def run_survival_analysis(self, a):
        """Kaplan-Meier survival analysis for Morph dataset"""
        if a.dataset != 'morph':
            raise ValueError("Survival analysis only available for Morph dataset")
        
        data = self.merged_data.copy()
        
        # Prepare survival data
        data['OS_event'] = data['Follow-up Status']
        data['PFS_event'] = data['Follow-up Status']
        
        print(f"Survival analysis for {len(data)} patients")
        print(f"Events (deaths): {data['OS_event'].sum()}")
        
        # Create output directory
        os.makedirs(f'out/{a.dataset}/survival', exist_ok=True)
        
        survival_types = ['OS', 'PFS'] if a.survival_type == 'both' else [a.survival_type]
        
        for surv_type in survival_types:
            time_col = surv_type
            event_col = f'{surv_type}_event'
            
            print(f"\n=== {surv_type} Analysis ===")
            
            # Overall survival curve
            kmf = KaplanMeierFitter()
            valid_data = data.dropna(subset=[time_col, event_col])
            kmf.fit(valid_data[time_col], valid_data[event_col], label='All patients')
            
            plt.figure(figsize=(10, 6))
            kmf.plot_survival_function()
            plt.title(f'{surv_type} - All Patients')
            plt.xlabel('Time (months)')
            plt.ylabel(f'{surv_type} Probability')
            plt.grid(True, alpha=0.3)
            plt.savefig(f'out/{a.dataset}/survival/{surv_type}_overall.png', dpi=300, bbox_inches='tight')
            if not a.noshow:
                plt.show()
            plt.close()
            
            print(f"Median {surv_type}: {kmf.median_survival_time_:.2f} months")
            
            # Age-based analysis
            if 'Age' in data.columns:
                data['Age_high'] = (data['Age'] > data['Age'].median()).astype(int)
                self._plot_survival_by_group(data, time_col, event_col, 'Age_high', 
                                           f'{surv_type}_Age', a, ['Age low', 'Age high'])
            
            # HANS analysis
            if 'HANS' in data.columns:
                self._plot_survival_by_group(data, time_col, event_col, 'HANS', 
                                           f'{surv_type}_HANS', a)
    
    def _plot_survival_by_group(self, data, time_col, event_col, group_var, title, a, labels=None):
        """Simple survival plot by grouping variable"""
        plot_data = data.dropna(subset=[time_col, event_col, group_var])
        groups = sorted([g for g in plot_data[group_var].unique() if pd.notna(g)])
        
        if len(groups) < 2:
            return
        
        plt.figure(figsize=(10, 6))
        group_data = {}
        
        for i, group in enumerate(groups):
            subset = plot_data[plot_data[group_var] == group]
            if len(subset) < 5:
                continue
                
            kmf = KaplanMeierFitter()
            durations = subset[time_col]
            events = subset[event_col]
            
            label = labels[i] if labels and i < len(labels) else f'{group_var} {group}'
            kmf.fit(durations, events, label=label)
            kmf.plot_survival_function()
            
            group_data[group] = (durations, events)
            print(f"{label}: n={len(subset)}, median={kmf.median_survival_time_:.2f} months")
        
        # Log-rank test for 2 groups
        if len(group_data) == 2:
            groups_list = list(group_data.keys())
            durations_A, events_A = group_data[groups_list[0]]
            durations_B, events_B = group_data[groups_list[1]]
            
            results = logrank_test(durations_A, durations_B, events_A, events_B)
            p_value = results.p_value
            
            plt.text(0.05, 0.05, f'Log-rank p = {p_value:.4f}', 
                    transform=plt.gca().transAxes, 
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            print(f"Log-rank test p-value: {p_value:.4f}")
        
        plt.title(title.replace('_', ' '))
        plt.xlabel('Time (months)')
        plt.ylabel('Survival Probability')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(f'out/{a.dataset}/survival/{title}.png', dpi=300, bbox_inches='tight')
        if not a.noshow:
            plt.show()
        plt.close()

    class GlobalClusterArgs(CommonArgs):
        noshow: bool = False
        n_samples: int = Field(100, s='-N')

    def run_global_cluster(self, a):
        features = []
        images = []
        dfs = []
        for dir in sorted(glob('data/dataset/*')):
            name = os.path.basename(dir)
            for i, h5_path in enumerate(sorted(glob(f'{dir}/*.h5'))):
                with h5py.File(h5_path, 'r') as f:
                    patch_count = f['metadata/patch_count'][()]
                    ii = np.random.choice(patch_count, size=a.n_samples, replace=False)
                    ii = np.sort(ii)
                    features.append(f['features'][ii])
                    images.append(f['patches'][ii])
                    df_wsi = pd.DataFrame({'index': ii})
                df_wsi['name'] = int(os.path.basename(dir))
                df_wsi['order'] = i
                df_wsi['filename'] = os.path.basename(h5_path)
                dfs.append(df_wsi)

        df = pd.concat(dfs)
        df_clinical = pd.read_excel('./data/clinical_data_cleaned.xlsx', index_col=0)
        df = pd.merge(
            df,
            df_clinical,
            left_on='name',
            right_index=True,
            how='left'
        )

        features = np.concatenate(features)
        images = np.concatenate(images)
        # images = [Image.fromarray(i) for i in images]

        print('Loaded features', features.dtype, features.shape)
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(features)

        print('UMAP fitting...')
        reducer = UMAP(
                # n_neighbors=80,
                # min_dist=0.3,
                n_components=2,
                metric='cosine',
                min_dist=0.5,
                spread=2.0
                # random_state=a.seed
            )
        embedding = reducer.fit_transform(scaled_features)
        print('UMAP ok')

        # scatter = plt.scatter(embedding[:, 0], embedding[:, 1], s=5, c=df['LDH'].values)
        # hover_images_on_scatters([scatter], [images])

        target = 'HANS'

        fig, ax = plt.subplots(figsize=(10, 8))
        plt.scatter(embedding[:, 0], embedding[:, 1], alpha=0)
        for (x, y), image, (_idx, row) in zip(embedding, images, df.iterrows()):
            img = OffsetImage(image, zoom=.125)
            value = row[target]

            text = TextArea(row['name'], textprops=dict(color='#000', ha='center'))
            vpack = VPacker(children=[text, img], align='center', pad=1)

            cmap = plt.cm.viridis
            color = '#333' if value < 0 else cmap(value)
            bbox_props = dict(boxstyle='square,pad=0.1', edgecolor=color, linewidth=1, facecolor='none')

            ab = AnnotationBbox(vpack, (x, y), frameon=True, bboxprops=bbox_props)
            ax.add_artist(ab)

        plt.title(f'UMAP')
        plt.xlabel('UMAP Dimension 1')
        plt.ylabel('UMAP Dimension 2')
        # plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.show()


    class StatisticalAnalysisArgs(CommonArgs):
        output_dir: str = param('out/statistics', description="Directory to save statistical analysis results")
        alpha: float = param(0.05, description="Significance level for statistical tests")

    def run_statistical_analysis(self, a: StatisticalAnalysisArgs):
        """臨床データとslide特徴量の統計解析を実行する"""
        
        # ... existing code ...

    class ComprehensiveHeatmapArgs(CommonArgs):
        output_dir: str = param('out/comprehensive_heatmap', description="Directory to save comprehensive heatmap results")
        correlation_method: str = param('pearson', choices=['pearson', 'spearman'], description="Correlation method to use")
        cluster_method: str = param('ward', choices=['ward', 'complete', 'average', 'single'], description="Clustering method for features")
        figsize_width: int = param(20, description="Figure width in inches")
        figsize_height: int = param(16, description="Figure height in inches")
        
    def run_comprehensive_heatmap(self, a: ComprehensiveHeatmapArgs):
        """Comprehensive heatmap analysis of all features and clinical data (with dendrogram)"""
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
        
        stats_file = f"out/statistics_debug/correlation_analysis_corrected.csv"
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

    class FeatureClinicalAnalysisArgs(CommonArgs):
        """注目特徴量と臨床データの詳細解析・可視化引数"""
        pass

    class SpuriousCorrelationDetectionArgs(CommonArgs):
        """疑似相関検出・可視化引数"""
        output_dir: str = param('out/spurious_correlation_detection', description="出力ディレクトリ")
        top_features: int = param(20, description="分析対象の上位特徴量数")
        min_correlation: float = param(0.15, description="表示する最小相関係数の絶対値")
        figsize_width: int = param(16, description="図の幅（インチ）")
        figsize_height: int = param(12, description="図の高さ（インチ）")

    class AnalysisValidationArgs(CommonArgs):
        """これまでの解析の妥当性評価引数"""
        output_dir: str = param('out/analysis_validation', description="出力ディレクトリ")
        n_permutations: int = param(1000, description="順列検定の回数")
        effect_size_threshold: float = param(0.3, description="実用的に意味のある効果サイズの閾値")
        fdr_alpha: float = param(0.05, description="False Discovery Rateの閾値")

    def run_analysis_validation(self, a: AnalysisValidationArgs):
        """これまでの解析結果の妥当性を包括的に評価"""
        
        print(f"解析妥当性評価開始: dataset={a.dataset}")
        
        # 出力ディレクトリ作成
        output_dir = Path(a.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 共有データ使用
        merged_data = self.merged_data
        print(f"使用データ: {len(merged_data)} サンプル")
        
        # 臨床変数定義
        clinical_vars = ['PFS', 'OS', 'LDH', 'Age', 'Stage', 'CD10_binary', 'HANS']
        available_clinical = [var for var in clinical_vars if var in merged_data.columns]
        feature_cols = [col for col in merged_data.columns if col.startswith('feature_')]
        
        print(f"特徴量数: {len(feature_cols)}")
        print(f"臨床変数数: {len(available_clinical)}")
        print(f"総検定数: {len(feature_cols) * len(available_clinical)}")
        
        # 1. 多重検定問題の評価
        self._evaluate_multiple_testing(merged_data, feature_cols, available_clinical, a, output_dir)
        
        # 2. 効果サイズの評価
        self._evaluate_effect_sizes(merged_data, feature_cols, available_clinical, a, output_dir)
        
        # 3. 順列検定による偶然性評価
        self._permutation_test_evaluation(merged_data, feature_cols, available_clinical, a, output_dir)
        
        # 4. 結果の頑健性評価
        self._robustness_evaluation(merged_data, feature_cols, available_clinical, a, output_dir)
        
        # 5. 包括的評価レポート作成
        self._create_validation_report(merged_data, feature_cols, available_clinical, a, output_dir)
        
        print(f"\n解析妥当性評価完了！結果は {output_dir} に保存されました")

    def _evaluate_multiple_testing(self, data, feature_cols, clinical_vars, a, output_dir):
        """多重検定問題の評価"""
        
        print("\n=== 多重検定問題の評価 ===")
        
        from statsmodels.stats.multitest import multipletests
        
        # 多重検定問題を評価するための統計量を計算
        p_values = []
        correlations = []
        feature_names = []
        clinical_names = []
        
        for feature in feature_cols:
            for clinical_var in clinical_vars:
                subset = data[[feature, clinical_var]].dropna()
                if len(subset) >= 10:
                    r, p = pearsonr(subset[feature], subset[clinical_var])
                    p_values.append(p)
                    correlations.append(r)
                    feature_names.append(feature)
                    clinical_names.append(clinical_var)
        
        # 多重検定補正後のp値を計算
        corrected_p_values = multipletests(p_values, method='fdr_bh')[1]
        
        # 期待される偶然有意数を計算
        total_tests = len(p_values)
        expected_false_positives = total_tests * 0.05
        actual_significant = sum([p < 0.05 for p in p_values])
        corrected_significant = sum([p < a.fdr_alpha for p in corrected_p_values])
        
        print(f"総検定数: {total_tests}")
        print(f"期待される偶然有意数(α=0.05): {expected_false_positives:.1f}")
        print(f"実際の有意数(p<0.05): {actual_significant}")
        print(f"多重検定補正後有意数(FDR<{a.fdr_alpha}): {corrected_significant}")
        
        # 可視化
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. p値のヒストグラム
        axes[0,0].hist(p_values, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0,0].axvline(0.05, color='red', linestyle='--', label='α=0.05')
        axes[0,0].axhline(expected_false_positives/50, color='orange', linestyle='--', 
                         label=f'Expected uniform: {expected_false_positives/50:.1f}')
        axes[0,0].set_xlabel('P-value')
        axes[0,0].set_ylabel('Frequency')
        axes[0,0].set_title('Distribution of P-values\n(Should be uniform under null)')
        axes[0,0].legend()
        
        # 2. 補正前後の比較
        significant_mask = np.array(p_values) < 0.05
        corrected_mask = corrected_p_values < a.fdr_alpha
        
        x = ['Uncorrected\n(p<0.05)', 'FDR Corrected\n(q<0.05)']
        y = [actual_significant, corrected_significant]
        colors = ['lightcoral', 'lightgreen']
        
        bars = axes[0,1].bar(x, y, color=colors, alpha=0.7)
        axes[0,1].axhline(expected_false_positives, color='red', linestyle='--', 
                         label=f'Expected false positives: {expected_false_positives:.1f}')
        axes[0,1].set_ylabel('Number of Significant Tests')
        axes[0,1].set_title('Multiple Testing Correction Impact')
        axes[0,1].legend()
        
        # 値をバーに表示
        for bar, value in zip(bars, y):
            axes[0,1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                          str(value), ha='center', va='bottom')
        
        # 3. Q-Q plot
        from scipy import stats
        sorted_p = np.sort(p_values)
        expected_p = np.linspace(0, 1, len(sorted_p))
        axes[1,0].scatter(expected_p, sorted_p, alpha=0.6, s=2)
        axes[1,0].plot([0, 1], [0, 1], 'r--', label='Perfect uniform')
        axes[1,0].set_xlabel('Expected P-value (uniform)')
        axes[1,0].set_ylabel('Observed P-value')
        axes[1,0].set_title('Q-Q Plot: P-values vs Uniform Distribution')
        axes[1,0].legend()
        
        # 4. 効果サイズ vs 有意性
        abs_correlations = [abs(r) for r in correlations]
        colors_sig = ['red' if sig else 'gray' for sig in significant_mask]
        
        axes[1,1].scatter(abs_correlations, [-np.log10(p) for p in p_values], 
                         c=colors_sig, alpha=0.6, s=3)
        axes[1,1].axhline(-np.log10(0.05), color='red', linestyle='--', label='p=0.05')
        axes[1,1].axvline(a.effect_size_threshold, color='orange', linestyle='--', 
                         label=f'Effect size threshold: {a.effect_size_threshold}')
        axes[1,1].set_xlabel('Absolute Correlation (Effect Size)')
        axes[1,1].set_ylabel('-log10(p-value)')
        axes[1,1].set_title('Volcano Plot: Effect Size vs Statistical Significance')
        axes[1,1].legend()
        
        plt.tight_layout()
        plt.savefig(output_dir / 'multiple_testing_evaluation.png', dpi=300, bbox_inches='tight')
        plt.savefig(output_dir / 'multiple_testing_evaluation.pdf', bbox_inches='tight')
        plt.close()
        
        # 結果をDataFrameに保存
        results_df = pd.DataFrame({
            'feature': feature_names,
            'clinical_variable': clinical_names,
            'correlation': correlations,
            'p_value': p_values,
            'corrected_p_value': corrected_p_values,
            'significant_uncorrected': [p < 0.05 for p in p_values],
            'significant_corrected': [p < a.fdr_alpha for p in corrected_p_values]
        })
        results_df.to_csv(output_dir / 'multiple_testing_evaluation.csv', index=False)
        
        print(f"多重検定問題の評価完了！結果は {output_dir}/multiple_testing_evaluation.csv に保存されました")

    def _evaluate_effect_sizes(self, data, feature_cols, clinical_vars, a, output_dir):
        """効果サイズの評価"""
        
        print("\n=== 効果サイズの評価 ===")
        
        # 効果サイズを計算
        effect_sizes = []
        feature_names = []
        clinical_names = []
        
        for feature in feature_cols:
            for clinical_var in clinical_vars:
                subset = data[[feature, clinical_var]].dropna()
                if len(subset) >= 10:
                    r, _ = pearsonr(subset[feature], subset[clinical_var])
                    effect_sizes.append(abs(r))  # 絶対値で効果サイズ
                    feature_names.append(feature)
                    clinical_names.append(clinical_var)
        
        # 効果サイズの統計量
        mean_effect_size = np.mean(effect_sizes)
        std_effect_size = np.std(effect_sizes)
        
        # Cohen's criteria for correlation
        small_effect = sum([es >= 0.1 for es in effect_sizes])
        medium_effect = sum([es >= 0.3 for es in effect_sizes])
        large_effect = sum([es >= 0.5 for es in effect_sizes])
        
        print(f"平均効果サイズ(|r|): {mean_effect_size:.3f}")
        print(f"小効果(|r|≥0.1): {small_effect} ({small_effect/len(effect_sizes)*100:.1f}%)")
        print(f"中効果(|r|≥0.3): {medium_effect} ({medium_effect/len(effect_sizes)*100:.1f}%)")
        print(f"大効果(|r|≥0.5): {large_effect} ({large_effect/len(effect_sizes)*100:.1f}%)")
        
        # 可視化
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. 効果サイズのヒストグラム
        axes[0,0].hist(effect_sizes, bins=30, alpha=0.7, color='lightblue', edgecolor='black')
        axes[0,0].axvline(0.1, color='green', linestyle='--', label='Small effect (0.1)')
        axes[0,0].axvline(0.3, color='orange', linestyle='--', label='Medium effect (0.3)')
        axes[0,0].axvline(0.5, color='red', linestyle='--', label='Large effect (0.5)')
        axes[0,0].axvline(mean_effect_size, color='black', linestyle='-', linewidth=2, 
                         label=f'Mean: {mean_effect_size:.3f}')
        axes[0,0].set_xlabel('Absolute Correlation (Effect Size)')
        axes[0,0].set_ylabel('Frequency')
        axes[0,0].set_title('Distribution of Effect Sizes')
        axes[0,0].legend()
        
        # 2. 効果サイズカテゴリの円グラフ
        negligible = len(effect_sizes) - small_effect
        small = small_effect - medium_effect
        medium = medium_effect - large_effect
        large = large_effect
        
        sizes = [negligible, small, medium, large]
        labels = ['Negligible\n(<0.1)', 'Small\n(0.1-0.3)', 'Medium\n(0.3-0.5)', 'Large\n(≥0.5)']
        colors = ['lightgray', 'lightgreen', 'orange', 'red']
        
        wedges, texts, autotexts = axes[0,1].pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%')
        axes[0,1].set_title('Effect Size Categories (Cohen\'s Criteria)')
        
        # 3. 臨床変数別効果サイズ
        clinical_effect_sizes = {}
        for i, clinical_var in enumerate(clinical_names):
            if clinical_var not in clinical_effect_sizes:
                clinical_effect_sizes[clinical_var] = []
            clinical_effect_sizes[clinical_var].append(effect_sizes[i])
        
        boxplot_data = [clinical_effect_sizes[var] for var in clinical_vars if var in clinical_effect_sizes]
        boxplot_labels = [var for var in clinical_vars if var in clinical_effect_sizes]
        
        axes[1,0].boxplot(boxplot_data, labels=boxplot_labels)
        axes[1,0].set_ylabel('Absolute Correlation')
        axes[1,0].set_title('Effect Sizes by Clinical Variable')
        axes[1,0].tick_params(axis='x', rotation=45)
        
        # 4. 上位効果サイズトップ20
        top_indices = np.argsort(effect_sizes)[-20:]
        top_effects = [effect_sizes[i] for i in top_indices]
        top_labels = [f'{feature_names[i]}-{clinical_names[i]}' for i in top_indices]
        
        y_pos = np.arange(len(top_effects))
        axes[1,1].barh(y_pos, top_effects, alpha=0.7)
        axes[1,1].set_yticks(y_pos)
        axes[1,1].set_yticklabels([label[:15] + '...' if len(label) > 15 else label for label in top_labels], fontsize=8)
        axes[1,1].set_xlabel('Absolute Correlation')
        axes[1,1].set_title('Top 20 Effect Sizes')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'effect_size_evaluation.png', dpi=300, bbox_inches='tight')
        plt.savefig(output_dir / 'effect_size_evaluation.pdf', bbox_inches='tight')
        plt.close()
        
        # 結果をDataFrameに保存
        results_df = pd.DataFrame({
            'feature': feature_names,
            'clinical_variable': clinical_names,
            'effect_size': effect_sizes,
            'effect_category': ['Large' if es >= 0.5 else 'Medium' if es >= 0.3 else 'Small' if es >= 0.1 else 'Negligible' 
                               for es in effect_sizes]
        })
        results_df.to_csv(output_dir / 'effect_size_evaluation.csv', index=False)
        
        print(f"効果サイズの評価完了！結果は {output_dir}/effect_size_evaluation.csv に保存されました")

    def _permutation_test_evaluation(self, data, feature_cols, clinical_vars, a, output_dir):
        """順列検定による偶然性評価"""
        
        print("\n=== 順列検定による偶然性評価 ===")
        
        # 実際の相関を計算
        observed_correlations = []
        feature_names = []
        clinical_names = []
        
        for feature in feature_cols[:20]:  # 計算時間短縮のため上位20特徴量で実行
            for clinical_var in clinical_vars:
                subset = data[[feature, clinical_var]].dropna()
                if len(subset) >= 10:
                    r, _ = pearsonr(subset[feature], subset[clinical_var])
                    observed_correlations.append(abs(r))
                    feature_names.append(feature)
                    clinical_names.append(clinical_var)
        
        print(f"順列検定実行中... ({len(observed_correlations)} 相関 × {a.n_permutations} 順列)")
        
        # 順列検定の実行 - null分布の構築
        max_null_correlations = []
        
        for perm_i in range(a.n_permutations):
            if perm_i % 100 == 0:
                print(f"  順列 {perm_i}/{a.n_permutations} 完了")
            
            # 臨床変数をシャッフル
            permuted_data = data.copy()
            for clinical_var in clinical_vars:
                permuted_data[clinical_var] = np.random.permutation(permuted_data[clinical_var].values)
            
            # この順列での最大相関を記録
            perm_correlations = []
            for i, (feature, clinical_var) in enumerate(zip(feature_names, clinical_names)):
                subset = permuted_data[[feature, clinical_var]].dropna()
                if len(subset) >= 10:
                    r, _ = pearsonr(subset[feature], subset[clinical_var])
                    perm_correlations.append(abs(r))
            
            if perm_correlations:
                max_null_correlations.append(max(perm_correlations))
        
        # 各観測値の順列検定p値を計算
        permutation_p_values = []
        for obs_corr in observed_correlations:
            p_val = (np.sum(np.array(max_null_correlations) >= obs_corr) + 1) / (a.n_permutations + 1)
            permutation_p_values.append(p_val)
        
        # 結果の統計
        max_observed = max(observed_correlations)
        null_95_percentile = np.percentile(max_null_correlations, 95)
        null_99_percentile = np.percentile(max_null_correlations, 99)
        
        print(f"最大観測相関: {max_observed:.3f}")
        print(f"Null分布95%ile: {null_95_percentile:.3f}")
        print(f"Null分布99%ile: {null_99_percentile:.3f}")
        print(f"順列検定で有意(p<0.05): {sum([p < 0.05 for p in permutation_p_values])}")
        
        # 可視化
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. Null分布 vs 観測値
        axes[0,0].hist(max_null_correlations, bins=50, alpha=0.7, color='lightblue', 
                      label='Null distribution', density=True)
        axes[0,0].axvline(max_observed, color='red', linestyle='-', linewidth=2, 
                         label=f'Max observed: {max_observed:.3f}')
        axes[0,0].axvline(null_95_percentile, color='orange', linestyle='--', 
                         label=f'95%ile: {null_95_percentile:.3f}')
        axes[0,0].axvline(null_99_percentile, color='red', linestyle='--', 
                         label=f'99%ile: {null_99_percentile:.3f}')
        axes[0,0].set_xlabel('Maximum Absolute Correlation')
        axes[0,0].set_ylabel('Density')
        axes[0,0].set_title('Null Distribution vs Observed Maximum')
        axes[0,0].legend()
        
        # 2. 順列検定p値の分布
        axes[0,1].hist(permutation_p_values, bins=20, alpha=0.7, color='lightgreen')
        axes[0,1].axvline(0.05, color='red', linestyle='--', label='α=0.05')
        axes[0,1].set_xlabel('Permutation P-value')
        axes[0,1].set_ylabel('Frequency')
        axes[0,1].set_title('Distribution of Permutation P-values')
        axes[0,1].legend()
        
        # 3. 観測相関 vs 順列p値
        axes[1,0].scatter(observed_correlations, permutation_p_values, alpha=0.6)
        axes[1,0].axhline(0.05, color='red', linestyle='--', label='p=0.05')
        axes[1,0].set_xlabel('Observed Absolute Correlation')
        axes[1,0].set_ylabel('Permutation P-value')
        axes[1,0].set_title('Correlation vs Permutation Significance')
        axes[1,0].legend()
        
        # 4. 有意性比較
        parametric_significant = sum([r > null_95_percentile for r in observed_correlations])
        permutation_significant = sum([p < 0.05 for p in permutation_p_values])
        
        comparison_data = [parametric_significant, permutation_significant]
        comparison_labels = ['Parametric\n(>95%ile)', 'Permutation\n(p<0.05)']
        
        bars = axes[1,1].bar(comparison_labels, comparison_data, color=['lightcoral', 'lightgreen'], alpha=0.7)
        axes[1,1].set_ylabel('Number of Significant Correlations')
        axes[1,1].set_title('Parametric vs Permutation Significance')
        
        # 値をバーに表示
        for bar, value in zip(bars, comparison_data):
            axes[1,1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                          str(value), ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'permutation_test_evaluation.png', dpi=300, bbox_inches='tight')
        plt.savefig(output_dir / 'permutation_test_evaluation.pdf', bbox_inches='tight')
        plt.close()
        
        # 結果をDataFrameに保存
        results_df = pd.DataFrame({
            'feature': feature_names,
            'clinical_variable': clinical_names,
            'observed_correlation': observed_correlations,
            'permutation_p_value': permutation_p_values,
            'significant_permutation': [p < 0.05 for p in permutation_p_values]
        })
        results_df.to_csv(output_dir / 'permutation_test_evaluation.csv', index=False)
        
        # Null分布の統計も保存
        null_stats_df = pd.DataFrame({
            'statistic': ['max_observed', 'null_mean', 'null_std', 'null_95_percentile', 'null_99_percentile'],
            'value': [max_observed, np.mean(max_null_correlations), np.std(max_null_correlations), 
                     null_95_percentile, null_99_percentile]
        })
        null_stats_df.to_csv(output_dir / 'null_distribution_stats.csv', index=False)
        
        print(f"順列検定による偶然性評価完了！結果は {output_dir}/permutation_test_evaluation.csv に保存されました")

    def _robustness_evaluation(self, data, feature_cols, clinical_vars, a, output_dir):
        """結果の頑健性評価"""
        
        print("\n=== 結果の頑健性評価 ===")
        
        # Bootstrap法による頑健性評価
        from sklearn.utils import resample
        
        # 元の相関を計算
        original_correlations = {}
        for feature in feature_cols[:20]:  # 計算時間短縮
            for clinical_var in clinical_vars:
                subset = data[[feature, clinical_var]].dropna()
                if len(subset) >= 20:  # Bootstrap用に最低20サンプル
                    r, _ = pearsonr(subset[feature], subset[clinical_var])
                    original_correlations[f'{feature}-{clinical_var}'] = {
                        'original_r': r,
                        'n_samples': len(subset)
                    }
        
        print(f"Bootstrap評価実行中... ({len(original_correlations)} 相関 × 100 bootstrap)")
        
        # Bootstrap サンプリング
        bootstrap_results = {}
        for combo, orig_data in original_correlations.items():
            feature, clinical_var = combo.split('-')
            subset = data[[feature, clinical_var]].dropna()
            
            bootstrap_correlations = []
            for _ in range(100):  # 100回のbootstrap
                boot_data = resample(subset, random_state=None)
                try:
                    r, _ = pearsonr(boot_data[feature], boot_data[clinical_var])
                    bootstrap_correlations.append(r)
                except:
                    pass
            
            if bootstrap_correlations:
                bootstrap_results[combo] = {
                    'original_r': orig_data['original_r'],
                    'bootstrap_mean': np.mean(bootstrap_correlations),
                    'bootstrap_std': np.std(bootstrap_correlations),
                    'bootstrap_ci_low': np.percentile(bootstrap_correlations, 2.5),
                    'bootstrap_ci_high': np.percentile(bootstrap_correlations, 97.5),
                    'n_samples': orig_data['n_samples']
                }
        
        # 頑健性メトリクス計算
        robustness_scores = []
        bias_scores = []
        ci_widths = []
        
        for combo, results in bootstrap_results.items():
            # バイアス (bootstrap平均 - 元の値)
            bias = abs(results['bootstrap_mean'] - results['original_r'])
            bias_scores.append(bias)
            
            # 信頼区間の幅
            ci_width = results['bootstrap_ci_high'] - results['bootstrap_ci_low']
            ci_widths.append(ci_width)
            
            # 頑健性スコア (バイアスと信頼区間幅の逆数)
            robustness = 1 / (bias + ci_width + 0.001)  # 0除算防止
            robustness_scores.append(robustness)
        
        print(f"平均バイアス: {np.mean(bias_scores):.4f}")
        print(f"平均CI幅: {np.mean(ci_widths):.3f}")
        print(f"高頑健性(bias<0.1 & CI<0.5): {sum([b<0.1 and c<0.5 for b,c in zip(bias_scores, ci_widths)])}")
        
        # 可視化
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. 元の相関 vs Bootstrap平均
        original_vals = [results['original_r'] for results in bootstrap_results.values()]
        bootstrap_vals = [results['bootstrap_mean'] for results in bootstrap_results.values()]
        
        axes[0,0].scatter(original_vals, bootstrap_vals, alpha=0.6)
        axes[0,0].plot([-1, 1], [-1, 1], 'r--', label='Perfect agreement')
        axes[0,0].set_xlabel('Original Correlation')
        axes[0,0].set_ylabel('Bootstrap Mean Correlation')
        axes[0,0].set_title('Bootstrap Reliability')
        axes[0,0].legend()
        
        # 2. バイアス分布
        axes[0,1].hist(bias_scores, bins=20, alpha=0.7, color='lightcoral')
        axes[0,1].axvline(np.mean(bias_scores), color='red', linestyle='-', 
                         label=f'Mean bias: {np.mean(bias_scores):.4f}')
        axes[0,1].axvline(0.1, color='orange', linestyle='--', label='Bias threshold: 0.1')
        axes[0,1].set_xlabel('Absolute Bias')
        axes[0,1].set_ylabel('Frequency')
        axes[0,1].set_title('Bootstrap Bias Distribution')
        axes[0,1].legend()
        
        # 3. 信頼区間幅分布
        axes[1,0].hist(ci_widths, bins=20, alpha=0.7, color='lightblue')
        axes[1,0].axvline(np.mean(ci_widths), color='blue', linestyle='-', 
                         label=f'Mean CI width: {np.mean(ci_widths):.3f}')
        axes[1,0].axvline(0.5, color='orange', linestyle='--', label='CI threshold: 0.5')
        axes[1,0].set_xlabel('95% CI Width')
        axes[1,0].set_ylabel('Frequency')
        axes[1,0].set_title('Confidence Interval Width Distribution')
        axes[1,0].legend()
        
        # 4. バイアス vs CI幅
        colors = ['green' if b<0.1 and c<0.5 else 'red' for b,c in zip(bias_scores, ci_widths)]
        axes[1,1].scatter(bias_scores, ci_widths, c=colors, alpha=0.6)
        axes[1,1].axvline(0.1, color='orange', linestyle='--', label='Bias threshold')
        axes[1,1].axhline(0.5, color='orange', linestyle='--', label='CI threshold')
        axes[1,1].set_xlabel('Absolute Bias')
        axes[1,1].set_ylabel('95% CI Width')
        axes[1,1].set_title('Robustness Map\n(Green=Robust, Red=Not robust)')
        axes[1,1].legend()
        
        plt.tight_layout()
        plt.savefig(output_dir / 'robustness_evaluation.png', dpi=300, bbox_inches='tight')
        plt.savefig(output_dir / 'robustness_evaluation.pdf', bbox_inches='tight')
        plt.close()
        
        # 結果をDataFrameに保存
        results_list = []
        for combo, results in bootstrap_results.items():
            feature, clinical_var = combo.split('-')
            results_list.append({
                'feature': feature,
                'clinical_variable': clinical_var,
                'original_correlation': results['original_r'],
                'bootstrap_mean': results['bootstrap_mean'],
                'bootstrap_std': results['bootstrap_std'],
                'bootstrap_ci_low': results['bootstrap_ci_low'],
                'bootstrap_ci_high': results['bootstrap_ci_high'],
                'bias': abs(results['bootstrap_mean'] - results['original_r']),
                'ci_width': results['bootstrap_ci_high'] - results['bootstrap_ci_low'],
                'robust': abs(results['bootstrap_mean'] - results['original_r']) < 0.1 and (results['bootstrap_ci_high'] - results['bootstrap_ci_low']) < 0.5,
                'n_samples': results['n_samples']
            })
        
        results_df = pd.DataFrame(results_list)
        results_df.to_csv(output_dir / 'robustness_evaluation.csv', index=False)
        
        print(f"結果の頑健性評価完了！結果は {output_dir}/robustness_evaluation.csv に保存されました")

    def _create_validation_report(self, data, feature_cols, clinical_vars, a, output_dir):
        """包括的評価レポート作成"""
        
        print("\n=== 包括的評価レポート作成 ===")
        
        # これまでの評価結果を読み込み
        try:
            multiple_testing_df = pd.read_csv(output_dir / 'multiple_testing_evaluation.csv')
            effect_size_df = pd.read_csv(output_dir / 'effect_size_evaluation.csv')
            permutation_df = pd.read_csv(output_dir / 'permutation_test_evaluation.csv')
            robustness_df = pd.read_csv(output_dir / 'robustness_evaluation.csv')
            null_stats_df = pd.read_csv(output_dir / 'null_distribution_stats.csv')
        except:
            print("警告: 一部の評価結果ファイルが見つかりません")
            return
        
        # 統計サマリー計算
        total_tests = len(multiple_testing_df)
        expected_false_positives = total_tests * 0.05
        actual_significant_uncorrected = multiple_testing_df['significant_uncorrected'].sum()
        actual_significant_corrected = multiple_testing_df['significant_corrected'].sum()
        
        large_effects = effect_size_df[effect_size_df['effect_category'] == 'Large'].shape[0]
        medium_effects = effect_size_df[effect_size_df['effect_category'] == 'Medium'].shape[0]
        
        permutation_significant = permutation_df['significant_permutation'].sum()
        robust_correlations = robustness_df[robustness_df['robust'] == True].shape[0]
        
        # 包括的評価ダッシュボード作成
        fig = plt.figure(figsize=(20, 16))
        gs = fig.add_gridspec(4, 3, hspace=0.3, wspace=0.3)
        
        # 1. 多重検定の影響
        ax1 = fig.add_subplot(gs[0, 0])
        categories = ['Expected\nFalse Positives', 'Uncorrected\nSignificant', 'FDR Corrected\nSignificant']
        values = [expected_false_positives, actual_significant_uncorrected, actual_significant_corrected]
        colors = ['red', 'orange', 'green']
        
        bars = ax1.bar(categories, values, color=colors, alpha=0.7)
        ax1.set_ylabel('Number of Tests')
        ax1.set_title('Multiple Testing Impact')
        
        for bar, value in zip(bars, values):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{int(value)}', ha='center', va='bottom')
        
        # 2. 効果サイズ分布
        ax2 = fig.add_subplot(gs[0, 1])
        effect_categories = effect_size_df['effect_category'].value_counts()
        colors_effect = {'Negligible': 'lightgray', 'Small': 'lightgreen', 
                        'Medium': 'orange', 'Large': 'red'}
        
        wedges, texts, autotexts = ax2.pie(effect_categories.values, 
                                          labels=effect_categories.index,
                                          colors=[colors_effect[cat] for cat in effect_categories.index],
                                          autopct='%1.1f%%')
        ax2.set_title('Effect Size Distribution')
        
        # 3. 順列検定結果
        ax3 = fig.add_subplot(gs[0, 2])
        perm_categories = ['Parametric\nSignificant', 'Permutation\nSignificant']
        perm_values = [actual_significant_uncorrected, permutation_significant]
        
        bars = ax3.bar(perm_categories, perm_values, color=['lightcoral', 'lightgreen'], alpha=0.7)
        ax3.set_ylabel('Number of Significant Tests')
        ax3.set_title('Parametric vs Permutation Test')
        
        for bar, value in zip(bars, perm_values):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{int(value)}', ha='center', va='bottom')
        
        # 4. 頑健性評価
        ax4 = fig.add_subplot(gs[1, 0])
        robust_categories = ['Robust', 'Not Robust']
        robust_values = [robust_correlations, len(robustness_df) - robust_correlations]
        
        bars = ax4.bar(robust_categories, robust_values, color=['green', 'red'], alpha=0.7)
        ax4.set_ylabel('Number of Correlations')
        ax4.set_title('Bootstrap Robustness')
        
        for bar, value in zip(bars, robust_values):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{int(value)}', ha='center', va='bottom')
        
        # 5. P値分布
        ax5 = fig.add_subplot(gs[1, 1])
        ax5.hist(multiple_testing_df['p_value'], bins=30, alpha=0.7, color='skyblue', density=True)
        ax5.axhline(1.0, color='red', linestyle='--', label='Uniform expectation')
        ax5.axvline(0.05, color='orange', linestyle='--', label='α=0.05')
        ax5.set_xlabel('P-value')
        ax5.set_ylabel('Density')
        ax5.set_title('P-value Distribution')
        ax5.legend()
        
        # 6. 効果サイズ vs P値 (Volcano plot)
        ax6 = fig.add_subplot(gs[1, 2])
        effect_merge = multiple_testing_df.merge(effect_size_df, on=['feature', 'clinical_variable'])
        
        significant_mask = effect_merge['significant_uncorrected']
        colors_volcano = ['red' if sig else 'gray' for sig in significant_mask]
        
        ax6.scatter(effect_merge['effect_size'], 
                   [-np.log10(p) for p in effect_merge['p_value']], 
                   c=colors_volcano, alpha=0.6, s=10)
        ax6.axhline(-np.log10(0.05), color='red', linestyle='--', label='p=0.05')
        ax6.axvline(a.effect_size_threshold, color='orange', linestyle='--', 
                   label=f'Effect threshold: {a.effect_size_threshold}')
        ax6.set_xlabel('Effect Size (|correlation|)')
        ax6.set_ylabel('-log10(p-value)')
        ax6.set_title('Volcano Plot: Effect vs Significance')
        ax6.legend()
        
        # 7-9. 臨床変数別解析
        clinical_summaries = []
        for clinical_var in clinical_vars:
            var_data = multiple_testing_df[multiple_testing_df['clinical_variable'] == clinical_var]
            if len(var_data) > 0:
                clinical_summaries.append({
                    'variable': clinical_var,
                    'n_tests': len(var_data),
                    'significant_uncorrected': var_data['significant_uncorrected'].sum(),
                    'significant_corrected': var_data['significant_corrected'].sum(),
                    'max_effect_size': effect_size_df[effect_size_df['clinical_variable'] == clinical_var]['effect_size'].max() if clinical_var in effect_size_df['clinical_variable'].values else 0
                })
        
        clinical_summary_df = pd.DataFrame(clinical_summaries)
        
        # 7. 臨床変数別有意数
        ax7 = fig.add_subplot(gs[2, :])
        x_pos = np.arange(len(clinical_summary_df))
        width = 0.35
        
        ax7.bar(x_pos - width/2, clinical_summary_df['significant_uncorrected'], 
               width, label='Uncorrected', alpha=0.8, color='lightcoral')
        ax7.bar(x_pos + width/2, clinical_summary_df['significant_corrected'], 
               width, label='FDR Corrected', alpha=0.8, color='lightgreen')
        
        ax7.set_xticks(x_pos)
        ax7.set_xticklabels(clinical_summary_df['variable'], rotation=45)
        ax7.set_ylabel('Number of Significant Correlations')
        ax7.set_title('Significant Correlations by Clinical Variable')
        ax7.legend()
        
        # 統計的解釈とレポート
        ax_text = fig.add_subplot(gs[3, :])
        ax_text.axis('off')
        
        # 解釈の自動生成
        interpretation = self._generate_interpretation(
            total_tests, expected_false_positives, actual_significant_uncorrected, 
            actual_significant_corrected, large_effects, medium_effects,
            permutation_significant, robust_correlations, len(robustness_df)
        )
        
        ax_text.text(0.05, 0.95, interpretation, transform=ax_text.transAxes, 
                    fontsize=11, verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.7))
        
        plt.suptitle('DLBCL Feature Analysis: Comprehensive Validation Report', fontsize=16, fontweight='bold')
        plt.savefig(output_dir / 'comprehensive_validation_report.png', dpi=300, bbox_inches='tight')
        plt.savefig(output_dir / 'comprehensive_validation_report.pdf', bbox_inches='tight')
        plt.close()
        
        # 数値サマリーも保存
        summary_stats = {
            'total_tests': total_tests,
            'expected_false_positives': expected_false_positives,
            'actual_significant_uncorrected': actual_significant_uncorrected,
            'actual_significant_corrected': actual_significant_corrected,
            'inflation_factor': actual_significant_uncorrected / expected_false_positives,
            'large_effect_correlations': large_effects,
            'medium_effect_correlations': medium_effects,
            'permutation_significant': permutation_significant,
            'robust_correlations': robust_correlations,
            'robustness_rate': robust_correlations / len(robustness_df) if len(robustness_df) > 0 else 0
        }
        
        summary_df = pd.DataFrame([summary_stats])
        summary_df.to_csv(output_dir / 'validation_summary_stats.csv', index=False)
        
        print(f"包括的評価レポート作成完了！結果は {output_dir}/comprehensive_validation_report.png に保存されました")

    def _generate_interpretation(self, total_tests, expected_fp, actual_uncorr, actual_corr, 
                               large_effects, medium_effects, perm_sig, robust_corr, total_robust):
        """統計的解釈の自動生成"""
        
        interpretation = "DLBCL Pathological Image Feature Analysis - Statistical Validation\n"
        interpretation += "=" * 70 + "\n\n"
        
        # 多重検定問題の評価
        inflation_factor = actual_uncorr / expected_fp if expected_fp > 0 else 0
        interpretation += f"MULTIPLE TESTING ASSESSMENT:\n"
        interpretation += f"• Total tests performed: {total_tests}\n"
        interpretation += f"• Expected false positives (α=0.05): {expected_fp:.1f}\n"
        interpretation += f"• Actual significant (uncorrected): {actual_uncorr}\n"
        interpretation += f"• Inflation factor: {inflation_factor:.1f}x\n"
        interpretation += f"• Significant after FDR correction: {actual_corr}\n\n"
        
        if inflation_factor > 2:
            interpretation += "⚠️  WARNING: Significant inflation of false positives detected!\n"
            interpretation += "   This suggests many 'significant' results may be spurious.\n\n"
        elif inflation_factor > 1.5:
            interpretation += "⚠️  CAUTION: Moderate inflation observed.\n"
            interpretation += "   Multiple testing correction is essential.\n\n"
        else:
            interpretation += "✓ Reasonable false positive rate observed.\n\n"
        
        # 効果サイズの評価
        interpretation += f"EFFECT SIZE ASSESSMENT:\n"
        interpretation += f"• Large effects (|r|≥0.5): {large_effects}\n"
        interpretation += f"• Medium effects (|r|≥0.3): {medium_effects}\n"
        
        if large_effects == 0:
            interpretation += "⚠️  No large effect sizes detected.\n"
            interpretation += "   Clinical relevance may be limited.\n\n"
        elif large_effects < 5:
            interpretation += "⚠️  Few large effect sizes.\n"
            interpretation += "   Focus on most robust findings.\n\n"
        else:
            interpretation += "✓ Multiple large effects detected.\n\n"
        
        # 順列検定の結果
        interpretation += f"PERMUTATION TEST RESULTS:\n"
        interpretation += f"• Permutation-based significant: {perm_sig}\n"
        interpretation += f"• Reduction from parametric: {actual_uncorr - perm_sig}\n"
        
        if perm_sig < actual_uncorr * 0.5:
            interpretation += "⚠️  Major reduction in significance with permutation test.\n"
            interpretation += "   Many parametric results may be artifacts.\n\n"
        else:
            interpretation += "✓ Permutation test confirms most parametric findings.\n\n"
        
        # 頑健性の評価
        robustness_rate = robust_corr / total_robust if total_robust > 0 else 0
        interpretation += f"ROBUSTNESS ASSESSMENT:\n"
        interpretation += f"• Robust correlations: {robust_corr}/{total_robust}\n"
        interpretation += f"• Robustness rate: {robustness_rate:.1%}\n"
        
        if robustness_rate < 0.5:
            interpretation += "⚠️  Low robustness rate.\n"
            interpretation += "   Results may be unstable across samples.\n\n"
        else:
            interpretation += "✓ Good robustness observed.\n\n"
        
        # 総合的な推奨
        interpretation += "RECOMMENDATIONS:\n"
        if inflation_factor > 2 or large_effects == 0 or robustness_rate < 0.5:
            interpretation += "• ⚠️  CAUTION: Multiple concerning signals detected\n"
            interpretation += "• Focus only on FDR-corrected, large-effect, robust findings\n"
            interpretation += "• Consider independent validation dataset\n"
            interpretation += "• Investigate biological plausibility of top features\n"
        else:
            interpretation += "• ✓ Generally acceptable statistical properties\n"
            interpretation += "• Proceed with FDR-corrected results\n"
            interpretation += "• Prioritize large effect sizes for follow-up\n"
        
        interpretation += f"\n📊 With {total_tests} tests on {768} features,\n"
        interpretation += "some significant correlations are expected by chance alone.\n"
        interpretation += "Focus on robust, large-effect findings with biological rationale."
        
        return interpretation

    class PathologyAnalysisArgs(CommonArgs):
        output_dir: str = param('out/pathology_analysis', description="Directory to save pathology analysis results")

    def run_pathology_analysis(self, a: PathologyAnalysisArgs):
        """Analyze Japanese pathology findings and extract IHC results"""
        
        print("Analyzing Japanese pathology findings and extracting IHC results...")
        
        # Define the 10 cases with clinical data and findings
        cases = [
            {
                'case_id': '20-2875',
                'clinical_data': {
                    'MYC IHC': 'missing',
                    'BCL2 IHC': 1,
                    'BCL6 IHC': 1,
                    'CD10 IHC': 0,
                    'MUM1 IHC': 1,
                    'EBV': 1,
                    'HANS': 0
                },
                'findings': '【提出検体】頚部リンパ節生検。【組織所見】淡好酸性から淡明な胞体を持つ大型異型細胞がびまん性に増殖しており周囲脂肪織にも一部浸潤しています。異型細胞は次の形質を示します。陽性：CD20(部分的), PAX5, Bcl-2, Bcl-6, Ki-67(90%以上), EBER-ISH(高いところで30%程度) 陰性：CD3, CD5, CD7 CD20は一部染色性が見られず、染色不良の可能性もあります。c-mycは一部染色不良の影響も考えられ、陽性と陰性の判断困難。転座の有無をご確認ください。Diffuse large B-cell lymphomaと考えます。WHO分類にあるEBV-positive diffuse large B-cell lymphomaでは80%以上の腫瘍細胞がEBV陽性とされています。'
            },
            {
                'case_id': '20-2925',
                'clinical_data': {
                    'MYC IHC': 'missing',
                    'BCL2 IHC': 'missing',
                    'BCL6 IHC': 1,
                    'CD10 IHC': 0,
                    'MUM1 IHC': 'missing',
                    'EBV': 0,
                    'HANS': 'missing'
                },
                'findings': '<Additional report 2020/10/29> 大型異型細胞は、CD20(+), CD30一部陽性、CD15(-)にて、Bcl-2ではT-cellに加えて、中型から大型異型細胞が陽性。上記は、異型細胞としてはB-cell lineageで、diffuse large B-cell lymphomaの所見です。CD10(-), Bcl-6(+), MUM1の染色を追加して、さらにsubtype分類を検討予定。'
            },
            {
                'case_id': '20-2946',
                'clinical_data': {
                    'MYC IHC': 'missing',
                    'BCL2 IHC': 'missing',
                    'BCL6 IHC': 1,
                    'CD10 IHC': 0,
                    'MUM1 IHC': 1,
                    'EBV': 'missing',
                    'HANS': 0
                },
                'findings': '<Additional report 2020/10/23> 免疫染色を追加して、CD5(+), Bcl-2(focal +), CD10(-), Bcl-6(+), MUM1(+)にて、前医報告のIgH/CCND1の融合シグナルなしの所見とも併せ、DLBCL, anaplastic variantと判断されます。なおCD5(+)DLBCLのため、Hans criteriaではnon-GCB typeに該当しますが、参考所見に留まります。'
            },
            {
                'case_id': '20-3355',
                'clinical_data': {
                    'MYC IHC': 1,
                    'BCL2 IHC': 1,
                    'BCL6 IHC': 0,
                    'CD10 IHC': 0,
                    'MUM1 IHC': 1,
                    'EBV': 0,
                    'HANS': 0
                },
                'findings': '【追加報告】 2020/11/30, 杉野 異型細胞は、Bcl-2陽性、Bcl-6陰性、MUM1陽性、c-myc陽性です。Non-germinal center B-cell typeに相当します。EBER-ISH陰性です。CD10陰性。Double expressor lymphoma (DEL)に相当しますが、MYC転座、BCL2転座の有無の確認をお勧めします。'
            },
            {
                'case_id': '20-3545',
                'clinical_data': {
                    'MYC IHC': 1,
                    'BCL2 IHC': 1,
                    'BCL6 IHC': 1,
                    'CD10 IHC': 1,
                    'MUM1 IHC': 'missing',
                    'EBV': 'missing',
                    'HANS': 1
                },
                'findings': '【組織所見】リンパ球様の大型異型細胞がびまん性に増殖しています。濾胞様構築は見られません。異型細胞は、CD3陰性、CD5陰性、CD20陽性、CD10陽性、Bcl-2陽性、Bcl-6陽性、c-myc弱陽性(30-70%程度)、TdT陰性です。Diffuse large B-cell lymphomaの所見で、germinal center B-cell typeに相当します。'
            },
            {
                'case_id': '20-3736',
                'clinical_data': {
                    'MYC IHC': 1,
                    'BCL2 IHC': 1,
                    'BCL6 IHC': 1,
                    'CD10 IHC': 0,
                    'MUM1 IHC': 1,
                    'EBV': 0,
                    'HANS': 0
                },
                'findings': '右頚部リンパ節生検：びまん性大細胞型リンパ腫、activated B-cell subtype を認める。免疫組織化学的に異型リンパ球は、CD20 (+)、BCL6 (+)、MUM-1 (+)、BCL2 (+)、MYC (+)、CD3 (-)、CD5 (-)、CD10 (-)、MIB-1 は 90% 以上に陽性を示す。EBER-ISH (-)。'
            },
            {
                'case_id': '20-3829',
                'clinical_data': {
                    'MYC IHC': 'missing',
                    'BCL2 IHC': 1,
                    'BCL6 IHC': 1,
                    'CD10 IHC': 0,
                    'MUM1 IHC': 1,
                    'EBV': 'missing',
                    'HANS': 0
                },
                'findings': '<Additional report 2021/01/14> 免疫染色を追加して、大型で異型目立つリンパ球のびまん性増殖についてCD20(+), CD5(+), CD10(-), bcl-2(+), bcl-6(+), MUM1(+), Ki67(MIB-1 LI 80%)にて、CD3は少数散在性のbystanderに陽性となるに留まり、DLBCL, non-GCB typeの診断です。'
            },
            {
                'case_id': '21-0053',
                'clinical_data': {
                    'MYC IHC': 'missing',
                    'BCL2 IHC': 'missing',
                    'BCL6 IHC': 1,
                    'CD10 IHC': 0,
                    'MUM1 IHC': 1,
                    'EBV': 'missing',
                    'HANS': 0
                },
                'findings': '<Additional report 2021/01/18 小田/種井> 免疫染色を追加して検討しました。腫瘍細胞はCD10陰性、CD5陰性、Bcl6陽性、MUM-1陽性、Ki－67標識率は80%程度です。 Diffuse large B-cell lymphoma, non GCB typeの診断です。'
            },
            {
                'case_id': '21-0062',
                'clinical_data': {
                    'MYC IHC': 'missing',
                    'BCL2 IHC': 1,
                    'BCL6 IHC': 'missing',
                    'CD10 IHC': 1,
                    'MUM1 IHC': 'missing',
                    'EBV': 'missing',
                    'HANS': 1
                },
                'findings': '<Additional report 2021/01/18> 免疫染色を追加して、CD10が弱いながら陽性、CD21はわずかに陽性、ほかBcl-2(+), Ki67(MIB-1 LI 40%)程度にて、明らかな大型細胞のびまん性増殖ではなく、小型から中型の細胞が主体で、follicular lymphomaからのtransformationの可能性が考えられます。'
            },
            {
                'case_id': '21-0317',
                'clinical_data': {
                    'MYC IHC': 'missing',
                    'BCL2 IHC': 'missing',
                    'BCL6 IHC': 1,
                    'CD10 IHC': 1,
                    'MUM1 IHC': 1,
                    'EBV': 'missing',
                    'HANS': 1
                },
                'findings': '<Additional report 2021/02/17> 免疫染色を追加して、CD10(+), Bcl-6(+), MUM1(+)にてDLBCL, GCB typeに相当します。加えて念のためCD21による染色を追加しましたが、濾胞構造は認められません。'
            }
        ]
        
        # Extract IHC results from pathology findings
        extracted_results = []
        
        for case in cases:
            extracted_ihc = self._extract_ihc_from_findings(case['findings'])
            
            # Compare with clinical data
            comparison = self._compare_ihc_results(case['clinical_data'], extracted_ihc)
            
            extracted_results.append({
                'case_id': case['case_id'],
                'clinical_data': case['clinical_data'],
                'extracted_ihc': extracted_ihc,
                'comparison': comparison,
                'findings': case['findings']
            })
            
            # Print results
            print(f"\n=== Case {case['case_id']} ===")
            print("Clinical Data:")
            for key, value in case['clinical_data'].items():
                print(f"  {key}: {value}")
            
            print("\nExtracted IHC Results:")
            for marker, result in extracted_ihc.items():
                print(f"  {marker}: {result}")
            
            print("\nComparison:")
            for marker, comp in comparison.items():
                if comp['status'] != 'not_compared':
                    print(f"  {marker}: {comp['status']} (Clinical: {comp['clinical']}, Extracted: {comp['extracted']})")
        
        # Create output directory
        os.makedirs(a.output_dir, exist_ok=True)
        
        # Save results to CSV
        self._save_pathology_results(extracted_results, a.output_dir)
        
        # Create summary visualization
        self._create_pathology_visualization(extracted_results, a.output_dir)
        
        print(f"\nPathology analysis completed. Results saved in {a.output_dir}")
        
        return extracted_results

    def _extract_ihc_from_findings(self, findings_text):
        """Extract IHC results from Japanese pathology findings"""
        
        extracted = {}
        
        # Patterns for positive results
        positive_patterns = {
            'CD20': [r'CD20[(\s]*\+', r'CD20.*陽性', r'CD20.*positive'],
            'CD10': [r'CD10[(\s]*\+', r'CD10.*陽性', r'CD10.*positive'],
            'BCL2': [r'[Bb]cl-?2[(\s]*\+', r'[Bb]cl-?2.*陽性', r'BCL2.*positive'],
            'BCL6': [r'[Bb]cl-?6[(\s]*\+', r'[Bb]cl-?6.*陽性', r'BCL6.*positive'],
            'MUM1': [r'MUM-?1[(\s]*\+', r'MUM-?1.*陽性', r'MUM1.*positive'],
            'MYC': [r'c-?[Mm]yc[(\s]*\+', r'c-?[Mm]yc.*陽性', r'MYC.*positive'],
            'EBV': [r'EBER.*陽性', r'EBV.*陽性', r'EBER-ISH[(\s]*\+'],
            'CD3': [r'CD3[(\s]*\+', r'CD3.*陽性'],
            'CD5': [r'CD5[(\s]*\+', r'CD5.*陽性']
        }
        
        # Patterns for negative results
        negative_patterns = {
            'CD20': [r'CD20[(\s]*\-', r'CD20.*陰性', r'CD20.*negative'],
            'CD10': [r'CD10[(\s]*\-', r'CD10.*陰性', r'CD10.*negative'],
            'BCL2': [r'[Bb]cl-?2[(\s]*\-', r'[Bb]cl-?2.*陰性'],
            'BCL6': [r'[Bb]cl-?6[(\s]*\-', r'[Bb]cl-?6.*陰性'],
            'MUM1': [r'MUM-?1[(\s]*\-', r'MUM-?1.*陰性'],
            'MYC': [r'c-?[Mm]yc[(\s]*\-', r'c-?[Mm]yc.*陰性'],
            'EBV': [r'EBER.*陰性', r'EBV.*陰性', r'EBER-ISH[(\s]*\-'],
            'CD3': [r'CD3[(\s]*\-', r'CD3.*陰性'],
            'CD5': [r'CD5[(\s]*\-', r'CD5.*陰性']
        }
        
        # Check for each marker
        for marker in positive_patterns.keys():
            # Check for positive
            for pattern in positive_patterns[marker]:
                if re.search(pattern, findings_text):
                    extracted[marker] = 1
                    break
            
            # If not found positive, check for negative
            if marker not in extracted:
                for pattern in negative_patterns[marker]:
                    if re.search(pattern, findings_text):
                        extracted[marker] = 0
                        break
            
            # If neither found, mark as not mentioned
            if marker not in extracted:
                extracted[marker] = 'not_mentioned'
        
        # Special handling for c-myc ambiguous cases
        if re.search(r'c-myc.*判断困難', findings_text) or re.search(r'c-myc.*弱陽性', findings_text):
            extracted['MYC'] = 'ambiguous'
        
        # Handle focal positive cases
        if re.search(r'[Bb]cl-?2.*focal.*\+', findings_text):
            extracted['BCL2'] = 'focal_positive'
        
        # Handle percentages for EBER
        eber_match = re.search(r'EBER.*?(\d+)%', findings_text)
        if eber_match:
            percentage = int(eber_match.group(1))
            if percentage >= 80:
                extracted['EBV'] = 1
            elif percentage > 0:
                extracted['EBV'] = f'{percentage}%'
            else:
                extracted['EBV'] = 0
        
        return extracted

    def _compare_ihc_results(self, clinical_data, extracted_ihc):
        """Compare clinical data with extracted IHC results"""
        
        comparison = {}
        
        # Map clinical variable names to extracted marker names
        mapping = {
            'MYC IHC': 'MYC',
            'BCL2 IHC': 'BCL2', 
            'BCL6 IHC': 'BCL6',
            'CD10 IHC': 'CD10',
            'MUM1 IHC': 'MUM1',
            'EBV': 'EBV'
        }
        
        for clinical_var, marker in mapping.items():
            if clinical_var in clinical_data and marker in extracted_ihc:
                clinical_val = clinical_data[clinical_var]
                extracted_val = extracted_ihc[marker]
                
                # Skip if clinical data is missing
                if clinical_val == 'missing':
                    comparison[marker] = {
                        'status': 'clinical_missing',
                        'clinical': clinical_val,
                        'extracted': extracted_val
                    }
                    continue
                
                # Skip if extracted data is not mentioned
                if extracted_val == 'not_mentioned':
                    comparison[marker] = {
                        'status': 'not_mentioned_in_findings',
                        'clinical': clinical_val,
                        'extracted': extracted_val
                    }
                    continue
                
                # Compare values
                if str(clinical_val) == str(extracted_val):
                    status = 'match'
                elif extracted_val in ['ambiguous', 'focal_positive'] and clinical_val == 1:
                    status = 'partial_match'
                elif isinstance(extracted_val, str) and '%' in str(extracted_val):
                    status = 'percentage_reported'
                else:
                    status = 'mismatch'
                
                comparison[marker] = {
                    'status': status,
                    'clinical': clinical_val,
                    'extracted': extracted_val
                }
            else:
                # One side missing
                comparison[marker] = {
                    'status': 'not_compared',
                    'clinical': clinical_data.get(clinical_var, 'missing'),
                    'extracted': extracted_ihc.get(marker, 'not_mentioned')
                }
        
        return comparison

    def _save_pathology_results(self, results, output_dir):
        """Save pathology analysis results to CSV files"""
        
        # Flatten results for CSV export
        flattened_data = []
        
        for result in results:
            row = {'case_id': result['case_id']}
            
            # Add clinical data
            for key, value in result['clinical_data'].items():
                row[f'clinical_{key}'] = value
            
            # Add extracted IHC
            for marker, value in result['extracted_ihc'].items():
                row[f'extracted_{marker}'] = value
            
            # Add comparison status
            for marker, comp in result['comparison'].items():
                row[f'comparison_{marker}_status'] = comp['status']
            
            flattened_data.append(row)
        
        # Save main results
        df_results = pd.DataFrame(flattened_data)
        df_results.to_csv(f"{output_dir}/pathology_analysis_results.csv", index=False)
        
        # Create summary statistics
        summary_stats = self._calculate_pathology_summary(results)
        
        with open(f"{output_dir}/pathology_summary.txt", 'w', encoding='utf-8') as f:
            f.write("Japanese Pathology Findings Analysis Summary\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Total cases analyzed: {len(results)}\n\n")
            
            f.write("IHC Marker Detection Rate:\n")
            for marker, stats in summary_stats['detection_rates'].items():
                f.write(f"  {marker}: {stats['detected']}/{stats['total']} ({stats['rate']:.1%})\n")
            
            f.write("\nComparison Results:\n")
            for status, count in summary_stats['comparison_status'].items():
                f.write(f"  {status}: {count}\n")
            
            f.write(f"\nOverall Agreement Rate: {summary_stats['overall_agreement']:.1%}\n")

    def _calculate_pathology_summary(self, results):
        """Calculate summary statistics for pathology analysis"""
        
        detection_rates = {}
        comparison_status = {}
        total_comparisons = 0
        matches = 0
        
        # Calculate detection rates for each marker
        markers = ['CD20', 'CD10', 'BCL2', 'BCL6', 'MUM1', 'MYC', 'EBV', 'CD3', 'CD5']
        
        for marker in markers:
            detected = 0
            total = len(results)
            
            for result in results:
                if marker in result['extracted_ihc'] and result['extracted_ihc'][marker] != 'not_mentioned':
                    detected += 1
            
            detection_rates[marker] = {
                'detected': detected,
                'total': total,
                'rate': detected / total if total > 0 else 0
            }
        
        # Calculate comparison status counts
        for result in results:
            for marker, comp in result['comparison'].items():
                status = comp['status']
                comparison_status[status] = comparison_status.get(status, 0) + 1
                
                if status in ['match', 'partial_match']:
                    matches += 1
                if status in ['match', 'partial_match', 'mismatch']:
                    total_comparisons += 1
        
        overall_agreement = matches / total_comparisons if total_comparisons > 0 else 0
        
        return {
            'detection_rates': detection_rates,
            'comparison_status': comparison_status,
            'overall_agreement': overall_agreement
        }

    def _create_pathology_visualization(self, results, output_dir):
        """Create visualizations for pathology analysis results"""
        
        # Extract comparison data for visualization
        markers = ['CD10', 'BCL2', 'BCL6', 'MUM1', 'MYC', 'EBV']
        comparison_matrix = []
        case_ids = []
        
        for result in results:
            case_ids.append(result['case_id'])
            row = []
            for marker in markers:
                if marker in result['comparison']:
                    status = result['comparison'][marker]['status']
                    if status == 'match':
                        row.append(1)
                    elif status in ['partial_match', 'percentage_reported']:
                        row.append(0.5)
                    elif status == 'mismatch':
                        row.append(0)
                    else:
                        row.append(-0.5)  # Not compared/missing
                else:
                    row.append(-0.5)
            comparison_matrix.append(row)
        
        # Create heatmap
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Comparison heatmap
        comparison_array = np.array(comparison_matrix)
        im1 = axes[0,0].imshow(comparison_array, cmap='RdYlGn', aspect='auto', vmin=-0.5, vmax=1)
        axes[0,0].set_xticks(range(len(markers)))
        axes[0,0].set_xticklabels(markers)
        axes[0,0].set_yticks(range(len(case_ids)))
        axes[0,0].set_yticklabels(case_ids)
        axes[0,0].set_title('IHC Results Comparison\n(Green=Match, Yellow=Partial, Red=Mismatch, Gray=Missing)')
        
        # Add colorbar
        cbar1 = plt.colorbar(im1, ax=axes[0,0])
        cbar1.set_label('Agreement Level')
        
        # 2. Detection rate by marker
        summary_stats = self._calculate_pathology_summary(results)
        markers_detect = list(summary_stats['detection_rates'].keys())
        detection_rates = [summary_stats['detection_rates'][m]['rate'] for m in markers_detect]
        
        axes[0,1].bar(markers_detect, detection_rates, alpha=0.7, color='steelblue')
        axes[0,1].set_ylabel('Detection Rate')
        axes[0,1].set_title('IHC Marker Detection Rate in Pathology Reports')
        axes[0,1].tick_params(axis='x', rotation=45)
        
        # 3. Comparison status distribution
        status_counts = summary_stats['comparison_status']
        status_labels = list(status_counts.keys())
        status_values = list(status_counts.values())
        
        colors = ['green', 'lightgreen', 'red', 'gray', 'lightgray', 'yellow'][:len(status_labels)]
        axes[1,0].pie(status_values, labels=status_labels, colors=colors, autopct='%1.1f%%')
        axes[1,0].set_title('Comparison Status Distribution')
        
        # 4. Case-by-case agreement scores
        agreement_scores = []
        for result in results:
            total_compared = 0
            matches = 0
            
            for marker, comp in result['comparison'].items():
                if comp['status'] in ['match', 'partial_match', 'mismatch']:
                    total_compared += 1
                    if comp['status'] in ['match', 'partial_match']:
                        matches += 1
            
            score = matches / total_compared if total_compared > 0 else 0
            agreement_scores.append(score)
        
        axes[1,1].bar(range(len(case_ids)), agreement_scores, alpha=0.7, color='lightcoral')
        axes[1,1].set_xticks(range(len(case_ids)))
        axes[1,1].set_xticklabels(case_ids, rotation=45)
        axes[1,1].set_ylabel('Agreement Rate')
        axes[1,1].set_title('Agreement Rate by Case')
        axes[1,1].set_ylim(0, 1)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/pathology_analysis_visualization.png", dpi=300, bbox_inches='tight')
        plt.savefig(f"{output_dir}/pathology_analysis_visualization.pdf", bbox_inches='tight')
        plt.close()
        
        print(f"Visualization saved to {output_dir}/pathology_analysis_visualization.png")


if __name__ == '__main__':
    cli = CLI()
    cli.run()

