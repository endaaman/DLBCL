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
                self.clinical_data = pd.read_csv(f'data/DLBCL-{dataset.capitalize()}/clinical_data_extracted_from_findings.csv')
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

    def run_cluster(self, a):
        with h5py.File('./data/DLBCL-Morph/slide_features.h5', 'r') as f:
            features = f['features'][:]
            df = pd.DataFrame({
                'name': [int((v.decode('utf-8'))) for v in f['names'][:]],
                'filename': [v.decode('utf-8') for v in f['filenames'][:]],
                'order': f['orders'][:],
            })

        df_clinical = pd.read_excel('./data/clinical_data_cleaned.xlsx', index_col=0)
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
        reducer = umap.UMAP(
                n_neighbors=10,
                min_dist=0.05,
                n_components=2,
                metric='cosine',
                random_state=a.seed,
                n_jobs=1,
            )
        embedding = reducer.fit_transform(scaled_features)
        print('Loaded features', features.shape)

        if a.target in [
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
        os.makedirs('out/morph', exist_ok=True)
        name = a.target.replace(' ', '_')
        plt.savefig(f'out/morph/umap_{name}.png')
        if not a.noshow:
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



if __name__ == '__main__':
    cli = CLI()
    cli.run()

