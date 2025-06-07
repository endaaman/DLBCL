
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.patches as patches
from scipy.stats import pearsonr, spearmanr
from scipy import stats
from pydantic import Field
from pydantic_autocli import param
from sklearn.utils import resample
from statsmodels.stats.multitest import multipletests

from .utils import BaseMLCLI, BaseMLArgs
from .utils.data_loader import load_common_data

warnings.filterwarnings('ignore', category=FutureWarning, message='.*force_all_finite.*')


class CLI(BaseMLCLI):
    class CommonArgs(BaseMLCLI.CommonArgs):
        pass

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
        clinical_vars = [
            'PFS', 'OS', 'LDH', 'Age', 'Stage', 'CD10_binary', 'HANS',
            'MYC IHC', 'BCL2 IHC', 'BCL6 IHC', 'CD10 IHC', 'MUM1 IHC',
            'BCL2 FISH', 'BCL6 FISH', 'MYC FISH',
            'ECOG PS', 'EN', 'IPI Score', 'Follow-up Status'
        ]
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



if __name__ == '__main__':
    cli = CLI()
    cli.run()

