import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr, chi2_contingency, mannwhitneyu, fisher_exact
from scipy.cluster.hierarchy import dendrogram, linkage
from pydantic import Field
from pydantic_autocli import param
from statsmodels.stats.multitest import multipletests

from .utils import ExperimentCLI
from .utils.data_loader import load_common_data

warnings.filterwarnings('ignore', category=FutureWarning, message='.*force_all_finite.*')


class CLI(ExperimentCLI):
    class CommonArgs(BaseMLCLI.CommonArgs):
        pass

    class ClinicalCorrelationArgs(CommonArgs):
        """臨床データのみの相関解析引数"""
        output_dir: str = param('', description="出力ディレクトリ (デフォルト: out/{dataset}/clinical_correlation)")
        correlation_method: str = param('pearson', choices=['pearson', 'spearman'], description="相関係数の種類")
        fdr_alpha: float = param(0.05, description="False Discovery Rateの閾値")
        min_samples: int = param(10, description="相関計算に必要な最小サンプル数")
        include_survival: bool = param(True, description="生存データ（OS、PFS）を含めるか")

    def run_clinical_correlation(self, a: ClinicalCorrelationArgs):
        """臨床データのみの相関解析"""

        print(f"臨床データ相関解析開始: dataset={a.dataset}")

        if not a.output_dir:
            a.output_dir = f'out/{a.dataset}/clinical_correlation'

        # 出力ディレクトリ作成
        output_dir = Path(a.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # 臨床データのみ使用
        clinical_data = self.clinical_data.copy()
        print(f"臨床データ: {len(clinical_data)} 患者")
        print(f"利用可能な列: {list(clinical_data.columns)}")

        # 解析対象の臨床変数を定義
        numeric_vars = []
        categorical_vars = []

        # 免疫染色マーカー（連続値）- 両データセットで同じ列名
        ihc_markers = ['CD10 IHC', 'MUM1 IHC', 'BCL2 IHC', 'BCL6 IHC', 'MYC IHC']

        print(f"IHCマーカー候補: {ihc_markers}")
        available_ihc = [var for var in ihc_markers if var in clinical_data.columns]
        print(f"利用可能なIHCマーカー: {available_ihc}")
        numeric_vars.extend(available_ihc)

        # 年齢・その他連続値
        other_numeric = ['Age']
        if a.include_survival:
            other_numeric.extend(['OS', 'PFS'])
        numeric_vars.extend([var for var in other_numeric if var in clinical_data.columns])

        # カテゴリカル変数 - 共通変数を優先
        categorical_candidates = ['HANS', 'LDH', 'ECOG PS', 'Stage', 'IPI Risk Group (4 Class)',
                                'Follow-up Status', 'BCL2 FISH', 'BCL6 FISH', 'MYC FISH', 'EBV']

        print(f"カテゴリカル変数候補: {categorical_candidates}")
        available_categorical = [var for var in categorical_candidates if var in clinical_data.columns]
        print(f"利用可能なカテゴリカル変数: {available_categorical}")
        categorical_vars.extend(available_categorical)

        all_vars = numeric_vars + categorical_vars
        print(f"解析対象変数: {len(all_vars)} 個")
        print(f"  - 連続値: {len(numeric_vars)} 個 ({numeric_vars[:5]}...)")
        print(f"  - カテゴリカル: {len(categorical_vars)} 個 ({categorical_vars[:5]}...)")

        # 1. 連続値変数間の相関解析
        self._analyze_numeric_correlations(clinical_data, numeric_vars, a, output_dir)

        # 2. カテゴリカル変数間の関連解析
        self._analyze_categorical_associations(clinical_data, categorical_vars, a, output_dir)

        # 3. 連続値 vs カテゴリカル変数の解析
        self._analyze_mixed_associations(clinical_data, numeric_vars, categorical_vars, a, output_dir)

        # 4. 包括的相関マトリックス作成
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

                if len(subset) >= a.min_samples:
                    try:
                        if a.correlation_method == 'pearson':
                            corr, p_val = pearsonr(subset[var1], subset[var2])
                        else:
                            corr, p_val = spearmanr(subset[var1], subset[var2])

                        correlation_matrix[i, j] = corr
                        correlation_matrix[j, i] = corr  # 対称行列
                        p_value_matrix[i, j] = p_val
                        p_value_matrix[j, i] = p_val

                        results.append({
                            'variable_1': var1,
                            'variable_2': var2,
                            'correlation': corr,
                            'p_value': p_val,
                            'n_samples': len(subset)
                        })
                    except:
                        pass

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

            # 相関行列のヒートマップ作成
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

                if len(subset) >= a.min_samples:
                    try:
                        # クロス集計表作成
                        crosstab = pd.crosstab(subset[var1], subset[var2])

                        # 期待度数が5未満のセルが80%未満の場合のみカイ二乗検定実行
                        if crosstab.size > 1:
                            if crosstab.shape == (2, 2) and crosstab.min().min() >= 5:
                                # 2×2表でFisher's exact test
                                _, p_val = fisher_exact(crosstab)
                                test_method = 'Fisher'
                            else:
                                # カイ二乗検定
                                chi2, p_val, dof, expected = chi2_contingency(crosstab)
                                test_method = 'Chi-square'

                            # Cramér's V を効果サイズとして計算
                            n = crosstab.sum().sum()
                            cramers_v = np.sqrt(chi2 / (n * (min(crosstab.shape) - 1))) if test_method == 'Chi-square' else np.nan

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

                if len(subset) >= a.min_samples:
                    try:
                        # カテゴリ別の値を取得
                        groups = []
                        for category in subset[categorical_var].unique():
                            group_data = subset[subset[categorical_var] == category][numeric_var]
                            if len(group_data) >= 3:  # 最低3サンプル
                                groups.append(group_data)

                        if len(groups) >= 2:
                            # Mann-Whitney U test (2群) または Kruskal-Wallis test (3群以上)
                            if len(groups) == 2:
                                stat, p_val = mannwhitneyu(groups[0], groups[1], alternative='two-sided')
                                test_method = 'Mann-Whitney'
                            else:
                                from scipy.stats import kruskal
                                stat, p_val = kruskal(*groups)
                                test_method = 'Kruskal-Wallis'

                            # 効果サイズ（群間の標準化された差）
                            all_values = subset[numeric_var]
                            overall_std = all_values.std()
                            group_means = [group.mean() for group in groups]
                            effect_size = (max(group_means) - min(group_means)) / overall_std if overall_std > 0 else 0

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


if __name__ == '__main__':
    cli = CLI()
    cli.run()
