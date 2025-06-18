import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr, spearmanr
from pydantic import Field
from pydantic_autocli import param

from .utils import BaseMLCLI

warnings.filterwarnings('ignore', category=FutureWarning, message='.*force_all_finite.*')


class CLI(BaseMLCLI):
    class CommonArgs(BaseMLCLI.CommonArgs):
        pass

    # def apply_thresholds(self, data, target_col, dataset):
    #     """データセット別の閾値処理を適用"""
    #     if dataset == 'morph':
    #         if target_col == 'MYC IHC':
    #             return (data >= 40).astype(int)
    #         elif target_col == 'BCL2 IHC':
    #             return (data >= 50).astype(int)
    #         elif target_col == 'BCL6 IHC':
    #             return (data >= 30).astype(int)
    #         else:
    #             # その他のカラムは既にバイナリ
    #             unique_vals = data.unique()
    #             if set(unique_vals).issubset({0, 1}):
    #                 return data.astype(int)
    #             else:
    #                 return (data == unique_vals[1]).astype(int) if len(unique_vals) == 2 else data
    #     else:
    #         # Patho2は既にバイナリ
    #         unique_vals = data.unique()
    #         if set(unique_vals).issubset({0, 1}):
    #             return data.astype(int)
    #         else:
    #             return (data == unique_vals[1]).astype(int) if len(unique_vals) == 2 else data

    class WeightAnalysisArgs(CommonArgs):
        """重み解析引数"""
        output_dir: str = param('out/weight_analysis', description="出力ディレクトリ")
        random_state: int = param(42, description="乱数シード")
        max_iter: int = param(1000, description="ロジスティック回帰の最大反復回数")

    def run_weight_analysis(self, a: WeightAnalysisArgs):
        """各染色マーカーでロジスティック回帰を実行し、重み解析を行う"""

        print(f"重み解析開始: dataset={a.dataset}")

        if self.merged_data is None:
            print(f"データの読み込みに失敗しました: {a.dataset}")
            return

        # 出力ディレクトリ作成
        output_dir = Path(a.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"データ形状: {self.merged_data.shape}")
        print(f"特徴量数: {len(self.feature_cols)}")
        print(f"分析対象カラム ({len(self.target_cols)}): {self.target_cols}")

        # 特徴量を標準化
        scaler = StandardScaler()
        X = scaler.fit_transform(self.merged_data[self.feature_cols])

        # 各ターゲットカラムの重みを格納
        all_weights = {}
        target_info = {}

        for target_col in self.target_cols:
            print(f"\n--- {target_col} の重み解析 ---")

            # 欠損値を除去
            mask = ~self.merged_data[target_col].isna()
            X_clean = X[mask]
            y_raw = self.merged_data[target_col][mask].copy()

            # データセット別バイナリ化
            # BaseMLCLIで補正済み
            # y_clean = self.apply_thresholds(y_raw, target_col, a.dataset)

            print(f"  サンプル数: {len(y_clean)}")
            print(f"  正例: {y_clean.sum()}, 負例: {len(y_clean) - y_clean.sum()}")

            # 単一クラスの場合はスキップ
            if y_clean.sum() == 0 or y_clean.sum() == len(y_clean):
                print(f"  スキップ: {target_col} は単一クラスです")
                continue

            # 正例が少なすぎる場合もスキップ
            if y_clean.sum() < 2 or (len(y_clean) - y_clean.sum()) < 2:
                print(f"  スキップ: {target_col} はクラス不均衡が極端です")
                continue

            # ロジスティック回帰
            clf = LogisticRegression(random_state=a.random_state, max_iter=a.max_iter)
            clf.fit(X_clean, y_clean)

            # 重みを取得
            weights = clf.coef_[0]  # 768次元

            # 結果を保存
            all_weights[target_col] = weights
            target_info[target_col] = {
                'n_samples': len(y_clean),
                'n_positive': y_clean.sum(),
                'n_negative': len(y_clean) - y_clean.sum(),
                'dataset': a.dataset
            }

            print(f"  重み統計: mean={weights.mean():.3f}, std={weights.std():.3f}")
            print(f"  重み範囲: [{weights.min():.3f}, {weights.max():.3f}]")

        # 重み行列を作成 (カラム×特徴量)
        if all_weights:
            weight_matrix = pd.DataFrame(all_weights).T
            weight_matrix.columns = [f'feature_{i}' for i in range(len(self.feature_cols))]

            # 重み行列を保存
            weight_matrix.to_csv(output_dir / f'weights_{a.dataset}.csv')

            # ターゲット情報を保存
            target_info_df = pd.DataFrame(target_info).T
            target_info_df.to_csv(output_dir / f'target_info_{a.dataset}.csv')

            print(f"\n=== {a.dataset.upper()} 重み解析結果 ===")
            print(f"重み行列形状: {weight_matrix.shape}")
            print(f"保存されたターゲット数: {len(all_weights)}")

            return {
                'weights': weight_matrix,
                'target_info': target_info_df,
                'scaler': scaler
            }
        else:
            print("有効なターゲットカラムがありませんでした")
            return None

    class CompareWeightsArgs(CommonArgs):
        """重み比較引数"""
        output_dir: str = param('out/weight_comparison', description="出力ディレクトリ")

    def run_compare_weights(self, a: CompareWeightsArgs):
        """MorphとPatho2の重みを比較分析"""
        print("データセット間重み比較分析開始")

        # 出力ディレクトリ作成
        output_dir = Path(a.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # 両データセットの重みファイルを読み込み
        morph_weights_file = Path('out/weight_analysis/weights_morph.csv')
        patho2_weights_file = Path('out/weight_analysis/weights_patho2.csv')

        if not morph_weights_file.exists():
            print(f"Morphの重みファイルが見つかりません: {morph_weights_file}")
            print("先にmorphデータセットで重み解析を実行してください")
            return

        if not patho2_weights_file.exists():
            print(f"Patho2の重みファイルが見つかりません: {patho2_weights_file}")
            print("先にpatho2データセットで重み解析を実行してください")
            return

        # 重み行列読み込み
        morph_weights = pd.read_csv(morph_weights_file, index_col=0)
        patho2_weights = pd.read_csv(patho2_weights_file, index_col=0)

        print(f"Morph重み行列: {morph_weights.shape}")
        print(f"Patho2重み行列: {patho2_weights.shape}")

        # 共通のターゲットカラムを特定
        common_targets = set(morph_weights.index) & set(patho2_weights.index)
        print(f"共通ターゲット ({len(common_targets)}): {sorted(common_targets)}")

        if len(common_targets) == 0:
            print("共通のターゲットカラムがありません")
            return

        # 1. カラム間重み相関（データセット内）
        self._analyze_within_dataset_correlations(morph_weights, patho2_weights, output_dir)

        # 2. データセット間重み相関（同一カラム）
        self._analyze_between_dataset_correlations(morph_weights, patho2_weights, common_targets, output_dir)

        # 3. 重要特徴量の重複分析
        self._analyze_top_feature_overlap(morph_weights, patho2_weights, common_targets, output_dir)

        return {
            'morph_weights': morph_weights,
            'patho2_weights': patho2_weights,
            'common_targets': common_targets
        }

    def _analyze_within_dataset_correlations(self, morph_weights, patho2_weights, output_dir):
        """データセット内でのカラム間重み相関を分析"""
        print("\n=== データセット内カラム間重み相関 ===")

        # Morphデータセット内相関
        morph_corr = morph_weights.T.corr()
        patho2_corr = patho2_weights.T.corr()

        # ヒートマップ作成
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

        sns.heatmap(morph_corr, annot=True, cmap='RdBu_r', center=0,
                   square=True, ax=ax1, cbar_kws={'shrink': 0.8})
        ax1.set_title('Morph Dataset: Weight Correlations Between Markers')

        sns.heatmap(patho2_corr, annot=True, cmap='RdBu_r', center=0,
                   square=True, ax=ax2, cbar_kws={'shrink': 0.8})
        ax2.set_title('Patho2 Dataset: Weight Correlations Between Markers')

        plt.tight_layout()
        plt.savefig(output_dir / 'within_dataset_correlations.png', dpi=300, bbox_inches='tight')
        plt.close()

        # 相関行列を保存
        morph_corr.to_csv(output_dir / 'morph_within_correlations.csv')
        patho2_corr.to_csv(output_dir / 'patho2_within_correlations.csv')

        print(f"Morph内最高相関: {morph_corr.abs().where(~np.eye(len(morph_corr), dtype=bool)).max().max():.3f}")
        print(f"Patho2内最高相関: {patho2_corr.abs().where(~np.eye(len(patho2_corr), dtype=bool)).max().max():.3f}")

    def _analyze_between_dataset_correlations(self, morph_weights, patho2_weights, common_targets, output_dir):
        """データセット間での同一カラム重み相関を分析"""
        print("\n=== データセット間同一カラム重み相関 ===")

        correlations = {}

        for target in common_targets:
            morph_w = morph_weights.loc[target]
            patho2_w = patho2_weights.loc[target]

            # 相関係数計算
            corr_pearson, p_pearson = pearsonr(morph_w, patho2_w)
            corr_spearman, p_spearman = spearmanr(morph_w, patho2_w)

            correlations[target] = {
                'pearson_r': corr_pearson,
                'pearson_p': p_pearson,
                'spearman_r': corr_spearman,
                'spearman_p': p_spearman
            }

            print(f"{target}: Pearson r={corr_pearson:.3f} (p={p_pearson:.3f}), Spearman r={corr_spearman:.3f} (p={p_spearman:.3f})")

        # 結果をDataFrameに変換して保存
        corr_df = pd.DataFrame(correlations).T
        corr_df.to_csv(output_dir / 'between_dataset_correlations.csv')

        # 散布図作成（上位相関のもの）
        top_targets = corr_df.nlargest(4, 'pearson_r').index

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()

        for i, target in enumerate(top_targets):
            if i >= 4:
                break

            morph_w = morph_weights.loc[target]
            patho2_w = patho2_weights.loc[target]

            axes[i].scatter(morph_w, patho2_w, alpha=0.6)
            axes[i].set_xlabel('Morph Weights')
            axes[i].set_ylabel('Patho2 Weights')
            axes[i].set_title(f'{target}\nr={correlations[target]["pearson_r"]:.3f}')
            axes[i].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / 'weight_scatter_plots.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _analyze_top_feature_overlap(self, morph_weights, patho2_weights, common_targets, output_dir):
        """重要特徴量の重複分析"""
        print("\n=== 重要特徴量重複分析 ===")

        top_k_list = [10, 20, 50, 100]
        overlap_results = {}

        for target in common_targets:
            morph_w = morph_weights.loc[target]
            patho2_w = patho2_weights.loc[target]

            overlap_results[target] = {}

            for k in top_k_list:
                # 上位k個の重要特徴量（絶対値）
                morph_top_k = set(morph_w.abs().nlargest(k).index)
                patho2_top_k = set(patho2_w.abs().nlargest(k).index)

                # 重複数と重複率
                overlap_count = len(morph_top_k & patho2_top_k)
                overlap_ratio = overlap_count / k

                overlap_results[target][f'top_{k}_overlap_count'] = overlap_count
                overlap_results[target][f'top_{k}_overlap_ratio'] = overlap_ratio

                print(f"{target} Top-{k}: {overlap_count}/{k} ({overlap_ratio:.3f})")

        # 結果をDataFrameに変換して保存
        overlap_df = pd.DataFrame(overlap_results).T
        overlap_df.to_csv(output_dir / 'top_feature_overlaps.csv')

        # 重複率のヒートマップ
        ratio_cols = [col for col in overlap_df.columns if 'ratio' in col]
        ratio_data = overlap_df[ratio_cols]
        ratio_data.columns = [col.replace('top_', 'Top-').replace('_overlap_ratio', '') for col in ratio_data.columns]

        plt.figure(figsize=(10, 6))
        sns.heatmap(ratio_data, annot=True, cmap='Blues', vmin=0, vmax=1)
        plt.title('Top Feature Overlap Ratios Between Datasets')
        plt.ylabel('Target Markers')
        plt.xlabel('Top-K Features')
        plt.tight_layout()
        plt.savefig(output_dir / 'top_feature_overlap_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()


if __name__ == '__main__':
    cli = CLI()
    cli.run()
