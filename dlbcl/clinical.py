from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr, chi2_contingency, mannwhitneyu, fisher_exact, kruskal
from pydantic_autocli import param
from statsmodels.stats.multitest import multipletests

from .utils.cli import ExperimentCLI


class CLI(ExperimentCLI):

    class ClinicalCorrelationArgs(ExperimentCLI.CommonArgs):
        """臨床データのみの相関解析引数"""
        correlation_method: str = param('pearson', choices=['pearson', 'spearman'], description="相関係数の種類")
        fdr_alpha: float = param(0.05, description="False Discovery Rateの閾値")
        min_samples: int = param(10, description="相関計算に必要な最小サンプル数")
        include_survival: bool = param(True, description="生存データ（OS、PFS）を含めるか")

    def run_clinical_correlation(self, a: ClinicalCorrelationArgs):
        """臨床データのみの相関解析"""

        print(f"臨床データ相関解析開始: dataset={a.dataset}")

        # 基底クラスのoutput_dirに追加パスを付ける
        output_dir = self.output_dir / 'clinical_correlation'
        output_dir.mkdir(parents=True, exist_ok=True)

        # 臨床データのみ使用
        clinical_data = self.dataset.merged_data.copy()
        print(f"臨床データ: {len(clinical_data)} 患者")

        # 解析対象の臨床変数を定義
        numeric_vars = []
        categorical_vars = []

        # 全ての候補変数
        all_candidates = {
            'pathology_markers': ['CD10 IHC', 'MUM1 IHC', 'BCL2 IHC', 'BCL6 IHC', 'MYC IHC', 'HANS', 'EBV',
                                 'BCL2 FISH', 'BCL6 FISH', 'MYC FISH'],
            'clinical_numeric': ['Age', 'OS', 'PFS'],
            'clinical_categorical': ['LDH', 'ECOG PS', 'Stage', 'IPI Risk Group (4 Class)', 'Follow-up Status']
        }

        # データ型に基づいて自動分類
        for category, candidates in all_candidates.items():
            for var in candidates:
                if var not in clinical_data.columns:
                    continue

                # 生存データの除外チェック
                if not a.include_survival and var in ['OS', 'PFS']:
                    continue

                # 有効データの確認
                valid_data = clinical_data[var].dropna()
                if len(valid_data) < a.min_samples:
                    continue

                unique_values = valid_data.unique()

                # データ型による分類
                if len(unique_values) <= 10 or set(unique_values) <= {0, 1, 0.0, 1.0}:
                    categorical_vars.append(var)
                else:
                    numeric_vars.append(var)

        all_vars = numeric_vars + categorical_vars

        print(f"データ型に基づく自動分類結果:")
        print(f"  連続値変数: {numeric_vars}")
        print(f"  カテゴリカル変数: {categorical_vars[:10]}{'...' if len(categorical_vars) > 10 else ''}")
        print(f"解析対象変数: {len(all_vars)} 個")
        print(f"  - 連続値: {len(numeric_vars)} 個 ({numeric_vars[:5]}...)")
        print(f"  - カテゴリカル: {len(categorical_vars)} 個 ({categorical_vars[:5]}...)")

        # 変数の分布概要表示
        print("\n=== 変数の分布概要 ===")
        for var in all_vars:
            valid_values = clinical_data[var].dropna()
            if len(valid_values) == 0:
                print(f"  {var}: データなし")
                continue

            if var in numeric_vars:
                # 連続値
                mean_val = valid_values.mean()
                std_val = valid_values.std()
                print(f"  {var}: 平均={mean_val:.2f}±{std_val:.2f} (n={len(valid_values)})")
            else:
                # カテゴリカル
                value_counts = valid_values.value_counts()
                total = len(valid_values)

                # 二値変数（0/1）は順序を統一、それ以外は頻度順
                if set(value_counts.index) <= {0, 1, 0.0, 1.0}:
                    # 0, 1の順序で統一
                    ordered_values = [0, 1]
                    value_counts = value_counts.reindex([v for v in ordered_values if v in value_counts.index])

                # 上位3つまで表示
                top_values = []
                for val, count in value_counts.head(3).items():
                    pct = count / total * 100
                    # 数値は整数で表示
                    if isinstance(val, (int, float)) and val == int(val):
                        val_str = str(int(val))
                    else:
                        val_str = str(val)
                    top_values.append(f"{val_str}:{pct:.0f}%")

                ratio_str = " ".join(top_values)
                if len(value_counts) > 3:
                    ratio_str += " ..."

                print(f"  {var}: {ratio_str} (n={total})")


        # 解析実行

        # 1. 分布のプロット作成
        self._create_distribution_plots(clinical_data, numeric_vars, categorical_vars, output_dir)

        # 2. 連続値変数間の相関解析
        self._analyze_numeric_correlations(clinical_data, numeric_vars, a, output_dir)

        # 3. カテゴリカル変数間の関連解析
        self._analyze_categorical_associations(clinical_data, categorical_vars, a, output_dir)

        # 4. 連続値 vs カテゴリカル変数の解析
        self._analyze_mixed_associations(clinical_data, numeric_vars, categorical_vars, a, output_dir)

        # 5. 包括的相関マトリックス作成
        self._create_comprehensive_clinical_matrix(clinical_data, all_vars, a, output_dir)

        print(f"\n臨床データ相関解析完了！結果は {output_dir} に保存されました")

    def _analyze_numeric_correlations(self, data, numeric_vars, a, output_dir):
        """連続値変数間の相関解析"""

        print("\n=== 連続値変数間の相関解析 ===")

        if len(numeric_vars) < 2:
            print("連続値変数が2個未満のため、相関解析をスキップします")
            return

        # 相関行列を計算
        correlation_matrix = np.full((len(numeric_vars), len(numeric_vars)), np.nan)
        p_value_matrix = np.full((len(numeric_vars), len(numeric_vars)), np.nan)

        results = []

        for i, var1 in enumerate(numeric_vars):
            for j, var2 in enumerate(numeric_vars):
                if i >= j:  # 対角線と下三角のみ計算
                    continue

                # 有効なデータを取得
                subset = data[[var1, var2]].dropna()

                if len(subset) < a.min_samples:
                    continue

                # 定数チェック
                if subset[var1].std() == 0 or subset[var2].std() == 0:
                    print(f"  警告: {var1} と {var2} の片方が定数のため相関計算をスキップ")
                    continue

                try:
                    if a.correlation_method == 'pearson':
                        corr, p_val = pearsonr(subset[var1], subset[var2])
                    else:
                        corr, p_val = spearmanr(subset[var1], subset[var2])

                    # NaNチェック
                    if np.isnan(corr) or np.isnan(p_val):
                        print(f"  警告: {var1} と {var2} の相関計算でNaNが発生")
                        continue

                    correlation_matrix[i, j] = corr
                    correlation_matrix[j, i] = corr
                    p_value_matrix[i, j] = p_val
                    p_value_matrix[j, i] = p_val

                    results.append({
                        'variable_1': var1,
                        'variable_2': var2,
                        'correlation': corr,
                        'p_value': p_val,
                        'n_samples': len(subset)
                    })
                except Exception as e:
                    print(f"  エラー: {var1} vs {var2} - {str(e)}")

        # 対角線は1に設定
        for i in range(len(numeric_vars)):
            correlation_matrix[i, i] = 1.0
            p_value_matrix[i, i] = 0.0

        if results:
            # 多重検定補正
            p_values = [r['p_value'] for r in results]
            corrected_p = multipletests(p_values, method='fdr_bh')[1]

            for i, result in enumerate(results):
                result['p_corrected'] = corrected_p[i]
                result['significant'] = corrected_p[i] < a.fdr_alpha

            # 結果をDataFrameに保存
            results_df = pd.DataFrame(results)
            results_df = results_df.sort_values('correlation', key=abs, ascending=False)
            results_df.to_csv(output_dir / 'numeric_correlations.csv', index=False)

            # 有意な相関を表示
            significant_results = results_df[results_df['significant']]
            print(f"有意な相関（FDR < {a.fdr_alpha}）: {len(significant_results)} 個")

            if len(significant_results) > 0:
                print("トップ5の有意な相関:")
                for _, row in significant_results.head().iterrows():
                    print(f"  {row['variable_1']} - {row['variable_2']}: r={row['correlation']:.3f}, p={row['p_corrected']:.3e}")

        # ヒートマップ作成
        if len(numeric_vars) >= 2:
            self._create_correlation_heatmap(correlation_matrix, numeric_vars, a, output_dir, 'numeric')

        print(f"連続値変数相関解析完了: {len(results)} ペア解析")

    def _analyze_categorical_associations(self, data, categorical_vars, a, output_dir):
        """カテゴリカル変数間の関連解析（カイ二乗検定）"""

        print("\n=== カテゴリカル変数間の関連解析 ===")

        if len(categorical_vars) < 2:
            print("カテゴリカル変数が2個未満のため、関連解析をスキップします")
            return

        results = []

        for i, var1 in enumerate(categorical_vars):
            for j, var2 in enumerate(categorical_vars):
                if i >= j:  # 対角線と下三角のみ計算
                    continue

                # 有効なデータを取得
                subset = data[[var1, var2]].dropna()

                if len(subset) < a.min_samples:
                    continue

                try:
                    crosstab = pd.crosstab(subset[var1], subset[var2])

                    if crosstab.size <= 1:
                        continue

                    if crosstab.shape == (2, 2) and crosstab.min().min() >= 5:
                        # Fisher's exact test
                        _, p_val = fisher_exact(crosstab)
                        test_method = 'Fisher'
                        chi2 = None
                    else:
                        # Chi-square test
                        chi2, p_val, dof, expected = chi2_contingency(crosstab)
                        test_method = 'Chi-square'

                    # Cramér's V
                    n = crosstab.sum().sum()
                    cramers_v = np.sqrt(chi2 / (n * (min(crosstab.shape) - 1))) if chi2 else np.nan

                    results.append({
                        'variable_1': var1,
                        'variable_2': var2,
                        'test_method': test_method,
                        'effect_size': cramers_v,
                        'p_value': p_val,
                        'n_samples': len(subset)
                    })
                except:
                    pass

        if results:
            # 多重検定補正
            p_values = [r['p_value'] for r in results]
            corrected_p = multipletests(p_values, method='fdr_bh')[1]

            for i, result in enumerate(results):
                result['p_corrected'] = corrected_p[i]
                result['significant'] = corrected_p[i] < a.fdr_alpha

            # 結果をDataFrameに保存
            results_df = pd.DataFrame(results)
            results_df = results_df.sort_values('p_corrected')
            results_df.to_csv(output_dir / 'categorical_associations.csv', index=False)

            # 有意な関連を表示
            significant_results = results_df[results_df['significant']]
            print(f"有意な関連（FDR < {a.fdr_alpha}）: {len(significant_results)} 個")

            if len(significant_results) > 0:
                print("トップ5の有意な関連:")
                for _, row in significant_results.head().iterrows():
                    print(f"  {row['variable_1']} - {row['variable_2']}: {row['test_method']}, V={row['effect_size']:.3f}, p={row['p_corrected']:.3e}")

        print(f"カテゴリカル変数関連解析完了: {len(results)} ペア解析")

    def _analyze_mixed_associations(self, data, numeric_vars, categorical_vars, a, output_dir):
        """連続値 vs カテゴリカル変数の関連解析"""

        print("\n=== 連続値 vs カテゴリカル変数の関連解析 ===")

        if len(numeric_vars) == 0 or len(categorical_vars) == 0:
            print("連続値またはカテゴリカル変数がないため、混合解析をスキップします")
            return

        results = []

        for numeric_var in numeric_vars:
            for categorical_var in categorical_vars:
                # 有効なデータを取得
                subset = data[[numeric_var, categorical_var]].dropna()

                if len(subset) < a.min_samples:
                    continue

                try:
                    # カテゴリ別のグループを作成
                    groups = []
                    for category in subset[categorical_var].unique():
                        group_data = subset[subset[categorical_var] == category][numeric_var]
                        if len(group_data) >= 3:  # 最低3サンプル
                            groups.append(group_data)

                    if len(groups) < 2:
                        continue

                    # 検定実施
                    if len(groups) == 2:
                        stat, p_val = mannwhitneyu(groups[0], groups[1], alternative='two-sided')
                        test_method = 'Mann-Whitney'
                    else:
                        stat, p_val = kruskal(*groups)
                        test_method = 'Kruskal-Wallis'

                    # 効果サイズ計算
                    all_values = subset[numeric_var]
                    overall_std = all_values.std()
                    if overall_std == 0:
                        effect_size = 0
                    else:
                        group_means = [group.mean() for group in groups]
                        effect_size = (max(group_means) - min(group_means)) / overall_std

                    results.append({
                        'numeric_variable': numeric_var,
                        'categorical_variable': categorical_var,
                        'test_method': test_method,
                        'effect_size': effect_size,
                        'p_value': p_val,
                        'n_samples': len(subset),
                        'n_groups': len(groups)
                    })
                except:
                    pass

        if results:
            # 多重検定補正
            p_values = [r['p_value'] for r in results]
            corrected_p = multipletests(p_values, method='fdr_bh')[1]

            for i, result in enumerate(results):
                result['p_corrected'] = corrected_p[i]
                result['significant'] = corrected_p[i] < a.fdr_alpha

            # 結果をDataFrameに保存
            results_df = pd.DataFrame(results)
            results_df = results_df.sort_values('p_corrected')
            results_df.to_csv(output_dir / 'mixed_associations.csv', index=False)

            # 有意な関連を表示
            significant_results = results_df[results_df['significant']]
            print(f"有意な関連（FDR < {a.fdr_alpha}）: {len(significant_results)} 個")

            if len(significant_results) > 0:
                print("トップ5の有意な関連:")
                for _, row in significant_results.head().iterrows():
                    print(f"  {row['numeric_variable']} vs {row['categorical_variable']}: {row['test_method']}, ES={row['effect_size']:.3f}, p={row['p_corrected']:.3e}")

        print(f"混合変数関連解析完了: {len(results)} ペア解析")

    def _create_comprehensive_clinical_matrix(self, data, all_vars, a, output_dir):
        """包括的な臨床変数関連マトリックス作成"""

        print("\n=== 包括的臨床変数関連マトリックス作成 ===")

        # 関連の強さを示すマトリックス作成
        n_vars = len(all_vars)
        association_matrix = np.full((n_vars, n_vars), np.nan)
        p_value_matrix = np.full((n_vars, n_vars), np.nan)

        for i in range(n_vars):
            association_matrix[i, i] = 1.0  # 対角線
            p_value_matrix[i, i] = 0.0

        # 既存の結果ファイルから関連度を読み込み
        files_to_read = [
            ('numeric_correlations.csv', 'correlation'),
            ('categorical_associations.csv', 'effect_size'),
            ('mixed_associations.csv', 'effect_size')
        ]

        for filename, value_col in files_to_read:
            filepath = output_dir / filename
            if filepath.exists():
                df = pd.read_csv(filepath)
                for _, row in df.iterrows():
                    var1 = row.get('variable_1') or row.get('numeric_variable')
                    var2 = row.get('variable_2') or row.get('categorical_variable')

                    if var1 in all_vars and var2 in all_vars:
                        i, j = all_vars.index(var1), all_vars.index(var2)
                        association_value = abs(row[value_col]) if not pd.isna(row[value_col]) else 0
                        p_value = row['p_corrected']

                        association_matrix[i, j] = association_value
                        association_matrix[j, i] = association_value
                        p_value_matrix[i, j] = p_value
                        p_value_matrix[j, i] = p_value

        # ヒートマップ作成
        self._create_comprehensive_heatmap(association_matrix, p_value_matrix, all_vars, a, output_dir)

        # サマリー統計
        summary = {
            'dataset': a.dataset,
            'n_clinical_variables': len(all_vars),
            'correlation_method': a.correlation_method,
            'fdr_threshold': a.fdr_alpha,
            'significant_associations': np.sum(p_value_matrix < a.fdr_alpha) // 2,  # 対称行列なので半分
            'strong_associations': np.sum(association_matrix > 0.5) // 2,
            'mean_association': np.nanmean(association_matrix[association_matrix != 1.0])
        }

        with open(output_dir / 'clinical_summary.txt', 'w') as f:
            for key, value in summary.items():
                f.write(f"{key}: {value}\n")

        print("包括的マトリックス作成完了")

    def _create_correlation_heatmap(self, matrix, variables, a, output_dir, prefix):
        """相関行列のヒートマップ作成"""

        plt.figure(figsize=(10, 8))

        # マスクを作成（NaN値用）
        mask = np.isnan(matrix)

        # ヒートマップ作成
        sns.heatmap(matrix, annot=True, fmt='.3f', cmap='RdBu_r', center=0,
                   xticklabels=variables, yticklabels=variables,
                   mask=mask, square=True, cbar_kws={'label': f'{a.correlation_method.capitalize()} Correlation'})

        plt.title(f'Clinical Variables Correlation Matrix\n({a.dataset.upper()}, {a.correlation_method})')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()

        plt.savefig(output_dir / f'{prefix}_correlation_heatmap.png', dpi=300, bbox_inches='tight')
        plt.savefig(output_dir / f'{prefix}_correlation_heatmap.pdf', bbox_inches='tight')
        plt.close()

    def _create_comprehensive_heatmap(self, association_matrix, p_matrix, variables, a, output_dir):
        """包括的な関連度ヒートマップ作成"""

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

        # 関連度の強さ
        mask = np.isnan(association_matrix)
        sns.heatmap(association_matrix, annot=True, fmt='.3f', cmap='viridis',
                   xticklabels=variables, yticklabels=variables,
                   mask=mask, square=True, ax=ax1,
                   cbar_kws={'label': 'Association Strength'})
        ax1.set_title(f'Clinical Variables Association Strength\n({a.dataset.upper()})')
        ax1.tick_params(axis='x', rotation=45)

        # 有意性
        significance_matrix = -np.log10(p_matrix + 1e-16)  # -log10(p-value)
        mask2 = np.isnan(significance_matrix) | np.isinf(significance_matrix)
        sns.heatmap(significance_matrix, annot=True, fmt='.1f', cmap='Reds',
                   xticklabels=variables, yticklabels=variables,
                   mask=mask2, square=True, ax=ax2,
                   cbar_kws={'label': '-log10(p-value)'})
        ax2.set_title(f'Statistical Significance\n(-log10 corrected p-value)')
        ax2.tick_params(axis='x', rotation=45)

        plt.tight_layout()
        plt.savefig(output_dir / 'comprehensive_clinical_heatmap.png', dpi=300, bbox_inches='tight')
        plt.savefig(output_dir / 'comprehensive_clinical_heatmap.pdf', bbox_inches='tight')
        plt.close()

    def _create_distribution_plots(self, data, numeric_vars, categorical_vars, output_dir):
        """変数の分布をプロット"""

        # 連続値変数のヒストグラム
        if numeric_vars:
            n_numeric = len(numeric_vars)
            fig, axes = plt.subplots(1, n_numeric, figsize=(5*n_numeric, 4))
            if n_numeric == 1:
                axes = [axes]

            for i, var in enumerate(numeric_vars):
                valid_data = data[var].dropna()
                axes[i].hist(valid_data, bins=20, alpha=0.7, edgecolor='black')
                mean_val = valid_data.mean()
                std_val = valid_data.std()
                axes[i].axvline(mean_val, color='red', linestyle='--', label=f'Mean={mean_val:.1f}')
                axes[i].set_title(f'{var}\n(n={len(valid_data)})')
                axes[i].set_xlabel(var)
                axes[i].set_ylabel('Frequency')
                axes[i].legend()

            plt.tight_layout()
            plt.savefig(output_dir / 'distributions_numeric.png', dpi=300, bbox_inches='tight')
            plt.close()

        # カテゴリカル変数の棒グラフ
        if categorical_vars:
            # 変数数に応じてレイアウト調整
            n_vars = len(categorical_vars)
            n_cols = min(3, n_vars)
            n_rows = (n_vars + n_cols - 1) // n_cols

            fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
            if n_vars == 1:
                axes = [axes]
            elif n_rows == 1:
                axes = axes.flatten() if n_cols > 1 else [axes]
            else:
                axes = axes.flatten()

            for i, var in enumerate(categorical_vars):
                valid_data = data[var].dropna()
                value_counts = valid_data.value_counts()

                # 値でソート
                sorted_indices = sorted(value_counts.index)
                value_counts = value_counts.reindex(sorted_indices)

                # 値を整数表示に変換
                labels = []
                for val in value_counts.index:
                    if isinstance(val, (int, float)) and val == int(val):
                        labels.append(str(int(val)))
                    else:
                        labels.append(str(val))

                bars = axes[i].bar(labels, value_counts.values, alpha=0.7, edgecolor='black')

                # パーセンテージをバーの上に表示
                total = len(valid_data)
                for j, (bar, count) in enumerate(zip(bars, value_counts.values)):
                    pct = count / total * 100
                    axes[i].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                               f'{pct:.0f}%', ha='center', va='bottom', fontsize=9)

                axes[i].set_title(f'{var}\n(n={total})')
                axes[i].set_xlabel(var)
                axes[i].set_ylabel('Frequency')

                # x軸ラベルの回転
                if len(labels) > 5:
                    axes[i].tick_params(axis='x', rotation=45)

            # 余ったサブプロットを非表示
            for i in range(len(categorical_vars), len(axes)):
                axes[i].set_visible(False)

            plt.tight_layout()
            plt.savefig(output_dir / 'distributions_categorical.png', dpi=300, bbox_inches='tight')
            plt.close()

        print(f"分布プロットを保存: {output_dir}")


if __name__ == '__main__':
    cli = CLI()
    cli.run()
