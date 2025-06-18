import os
from pathlib import Path
import pandas as pd
import numpy as np
import scanpy as sc
import anndata as ad
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from umap import UMAP
from pydantic_autocli import param

from .utils.cli import ExperimentCLI
from .utils.dataset import load_dataset, merge_dataset


class CLI(ExperimentCLI):
    class LeidenArgs(ExperimentCLI.CommonArgs):
        resolution: float = param(0.5, description="Leiden clustering resolution")
        n_neighbors: int = param(15, description="Number of neighbors for clustering")

    def run_leiden(self, a: LeidenArgs):
        """Leiden クラスタリングを実行"""
        features = self.dataset.features

        # AnnDataオブジェクト作成
        adata = ad.AnnData(features)
        adata.obs['patient_id'] = self.dataset.feature_names

        # 近傍グラフ計算
        sc.pp.neighbors(adata, n_neighbors=a.n_neighbors)

        # Leidenクラスタリング実行
        sc.tl.leiden(adata, resolution=a.resolution, key_added='leiden_cluster')

        # 結果取得
        clusters = adata.obs['leiden_cluster'].astype(int).values
        n_clusters = len(set(clusters))

        print(f"Leiden クラスタリング完了: {n_clusters} clusters, {len(features)} samples")

        # 結果保存
        results_df = pd.DataFrame({
            'patient_id': self.dataset.feature_names,
            'leiden_cluster': clusters
        })
        output_file = f'{self.output_dir}/leiden_clusters.csv'
        results_df.to_csv(output_file, index=False)
        print(f"結果保存: {output_file}")

        return True

    class VisualizeArgs(ExperimentCLI.CommonArgs):
        target: str = param('Age', description="可視化対象（臨床変数名、leiden_cluster等）")
        n_neighbors: int = param(15, description="UMAP近傍数")
        min_dist: float = param(0.1, description="UMAP最小距離")
        noshow: bool = False

    def run_visualize(self, a: VisualizeArgs):
        """UMAP埋め込みによる統一的可視化"""
        features = self.dataset.features
        merged_data = self.dataset.merged_data

        # 特徴量とマージデータのサイズを合わせる
        min_size = min(len(features), len(merged_data))
        features = features[:min_size]
        merged_data = merged_data.iloc[:min_size]

        # leiden_clusterの場合は保存されたファイルから読み込み
        if a.target == 'leiden_cluster':
            cluster_file = f'{self.output_dir}/leiden_clusters.csv'
            cluster_df = pd.read_csv(cluster_file)
            # patient_idでマージ
            merged_data = merged_data.merge(cluster_df, on='patient_id', how='left')
            # マージ後にサイズ再調整
            min_size = min(len(features), len(merged_data))
            features = features[:min_size]
            merged_data = merged_data.iloc[:min_size]

        # ターゲット列の存在確認
        if a.target not in merged_data.columns:
            available = [col for col in merged_data.columns if not col.startswith('feature_')]
            raise ValueError(f"Target '{a.target}' not found. Available: {available}")

        # UMAP実行
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(features)

        reducer = UMAP(n_neighbors=a.n_neighbors, min_dist=a.min_dist,
                       n_components=2, random_state=a.seed)
        embedding = reducer.fit_transform(scaled_features)

        # 可視化モード判定
        target_data = merged_data[a.target]
        unique_values = target_data.dropna().unique()

        if len(unique_values) <= 20 and not np.issubdtype(target_data.dtype, np.floating):
            mode = 'categorical'
        else:
            mode = 'numeric'

        # プロット作成
        plt.figure(figsize=(10, 8))

        if mode == 'categorical':
            # カテゴリカル可視化
            for i, value in enumerate(unique_values):
                if pd.isna(value):
                    continue
                mask = target_data == value
                plt.scatter(embedding[mask, 0], embedding[mask, 1],
                           label=f'{a.target} {value}', alpha=0.7, s=50)

            # NaN値の可視化
            nan_mask = target_data.isna()
            if nan_mask.sum() > 0:
                plt.scatter(embedding[nan_mask, 0], embedding[nan_mask, 1],
                           c='gray', marker='x', label='Missing', alpha=0.7, s=50)
            plt.legend()
        else:
            # 数値可視化
            valid_mask = ~target_data.isna()
            scatter = plt.scatter(embedding[valid_mask, 0], embedding[valid_mask, 1],
                                 c=target_data[valid_mask], cmap='viridis', alpha=0.7, s=50)
            plt.colorbar(scatter, label=a.target)

            # NaN値の可視化
            if (~valid_mask).sum() > 0:
                plt.scatter(embedding[~valid_mask, 0], embedding[~valid_mask, 1],
                           c='gray', marker='x', label='Missing', alpha=0.7, s=50)
                plt.legend()

        plt.title(f'UMAP: {a.target} ({a.target.upper()}, Combat: {a.use_combat})')
        plt.xlabel('UMAP 1')
        plt.ylabel('UMAP 2')

        # 保存
        plot_file = Path(f'{self.output_dir}/umap/{a.target.replace(" ", "_")}.png')
        os.makedirs(plot_file.parent, exist_ok=True)
        plt.savefig(str(plot_file), dpi=300, bbox_inches='tight')
        print(f"可視化保存: {plot_file}")

        if not a.noshow:
            plt.show()

    class CombatComparisonArgs(ExperimentCLI.CommonArgs):
        n_neighbors: int = param(15, description="UMAP近傍数")
        min_dist: float = param(0.1, description="UMAP最小距離")

    def run_combat_comparison(self, a: CombatComparisonArgs):

        # 補正前データ読み込み（Combat補正されていない生データ）
        dataset_morph_raw = load_dataset('morph')
        dataset_patho2_raw = load_dataset('patho2')
        dataset_merged_raw = merge_dataset(dataset_morph_raw, dataset_patho2_raw)

        # 補正後データ（現在のself.dataset_merged）
        dataset_merged_corrected = self.dataset_merged

        # データセットラベル作成
        n_morph = len(dataset_morph_raw.features)
        n_patho2 = len(dataset_patho2_raw.features)
        dataset_labels = ['morph'] * n_morph + ['patho2'] * n_patho2

        # 特徴量標準化
        scaler_raw = StandardScaler()
        scaled_raw = scaler_raw.fit_transform(dataset_merged_raw.features)

        scaler_corrected = StandardScaler()
        scaled_corrected = scaler_corrected.fit_transform(dataset_merged_corrected.features)

        # UMAP実行
        reducer = UMAP(n_neighbors=a.n_neighbors, min_dist=a.min_dist,
                       n_components=2, random_state=a.seed)

        embedding_raw = reducer.fit_transform(scaled_raw)
        embedding_corrected = reducer.fit_transform(scaled_corrected)

        # 並列プロット
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        colors = {'morph': 'blue', 'patho2': 'red'}

        # 補正前
        for dataset in ['morph', 'patho2']:
            mask = np.array(dataset_labels) == dataset
            axes[0].scatter(embedding_raw[mask, 0], embedding_raw[mask, 1],
                           c=colors[dataset], label=dataset, alpha=0.7, s=30)
        axes[0].set_title('Before Combat Correction')
        axes[0].legend()

        # 補正後
        for dataset in ['morph', 'patho2']:
            mask = np.array(dataset_labels) == dataset
            axes[1].scatter(embedding_corrected[mask, 0], embedding_corrected[mask, 1],
                           c=colors[dataset], label=dataset, alpha=0.7, s=30)
        axes[1].set_title('After Combat Correction')
        axes[1].legend()

        plt.tight_layout()

        # 保存
        plot_file = f'{self.output_dir}/combat_comparison.png'
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"Combat比較保存: {plot_file}")

        # シルエット係数計算
        labels_numeric = [0 if label == 'morph' else 1 for label in dataset_labels]
        sil_raw = silhouette_score(embedding_raw, labels_numeric)
        sil_corrected = silhouette_score(embedding_corrected, labels_numeric)

        print(f"シルエット係数 - 補正前: {sil_raw:.3f}, 補正後: {sil_corrected:.3f}")
        print(f"改善度: {sil_raw - sil_corrected:.3f}")

        # 評価結果保存
        eval_df = pd.DataFrame({
            'metric': ['silhouette_before', 'silhouette_after', 'improvement'],
            'value': [sil_raw, sil_corrected, sil_raw - sil_corrected]
        })
        eval_file = f'{self.output_dir}/combat_evaluation.csv'
        eval_df.to_csv(eval_file, index=False)
        print(f"評価結果保存: {eval_file}")

        plt.show()


if __name__ == '__main__':
    cli = CLI()
    cli.run()
