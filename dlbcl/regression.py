import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib_venn import venn2
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import StandardScaler
from pathlib import Path
from pydantic_autocli import param

from .utils import BaseMLCLI
from .utils.correlation import load_both_datasets


class CLI(BaseMLCLI):
    class CommonArgs(BaseMLCLI.CommonArgs):
        pass

    def apply_thresholds(self, data, target_col, dataset):
        """データセット別の閾値処理を適用"""
        if dataset == 'morph':
            if target_col == 'MYC IHC':
                return (data >= 40).astype(int)
            elif target_col == 'BCL2 IHC':
                return (data >= 50).astype(int)
            elif target_col == 'BCL6 IHC':
                return (data >= 30).astype(int)

        # その他は既存値をそのまま使用
        unique_vals = data.unique()
        if set(unique_vals).issubset({0, 1}):
            return data.astype(int)
        elif len(unique_vals) == 2:
            return (data == unique_vals[1]).astype(int)
        else:
            return data

    class RocAnalysisArgs(CommonArgs):
        """ROC分析引数"""
        output_dir: str = param('', description="出力ディレクトリ (デフォルト: out/{dataset}/regression)")
        max_iter: int = param(1000, description="ロジスティック回帰の最大反復回数")
        use_train_test_split: bool = param(False, description="Train-test分割を使用するか（Falseで全データ学習）")
        test_size: float = param(0.3, description="テストデータの割合（use_train_test_split=Trueの場合）")

    def run_roc_analysis(self, a: RocAnalysisArgs):
        """全データでROC分析を実行"""

        print(f"ROC分析開始: dataset={a.dataset}")

        if self.merged_data is None:
            print(f"データの読み込みに失敗しました: {a.dataset}")
            return

        if not a.output_dir:
            a.output_dir = f'out/{a.dataset}/regression'

        output_dir = Path(a.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"データ形状: {self.merged_data.shape}")
        print(f"特徴量数: {len(self.feature_cols)}")
        print(f"分析対象カラム ({len(self.target_cols)}): {self.target_cols}")

        # 特徴量を標準化
        scaler = StandardScaler()
        X = scaler.fit_transform(self.merged_data[self.feature_cols])

        results = {}

        # ROC曲線プロット準備
        n_cols = min(4, len(self.target_cols))
        n_rows = (len(self.target_cols) + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
        if n_rows == 1 and n_cols == 1:
            axes = [axes]
        elif n_rows == 1 or n_cols == 1:
            axes = axes.flatten()
        else:
            axes = axes.flatten()

        plot_idx = 0

        for target_col in self.target_cols:
            print(f"\n--- {target_col} の分析 ---")

            # 欠損値を除去
            mask = ~self.merged_data[target_col].isna()
            X_clean = X[mask]
            y_raw = self.merged_data[target_col][mask].copy()

            # データセット別バイナリ化
            y_clean = self.apply_thresholds(y_raw, target_col, a.dataset)

            print(f"  サンプル数: {len(y_clean)}")
            print(f"  正例: {y_clean.sum()}, 負例: {len(y_clean) - y_clean.sum()}")

            # 単一クラスまたは極端な不均衡の場合はスキップ
            if y_clean.sum() <= 1 or (len(y_clean) - y_clean.sum()) <= 1:
                print(f"  スキップ: {target_col} は解析不可能なクラス分布です")
                results[target_col] = {
                    'auc': np.nan,
                    'n_samples': len(y_clean),
                    'n_positive': y_clean.sum(),
                    'n_negative': len(y_clean) - y_clean.sum(),
                    'status': 'insufficient_class'
                }
                continue

            if a.use_train_test_split:
                # Train-test分割
                from sklearn.model_selection import train_test_split
                X_train, X_test, y_train, y_test = train_test_split(
                    X_clean, y_clean, test_size=a.test_size,
                    random_state=42, stratify=y_clean
                )

                # 学習
                clf = LogisticRegression(random_state=42, max_iter=a.max_iter)
                clf.fit(X_train, y_train)

                # テストデータで予測
                y_prob = clf.predict_proba(X_test)[:, 1]

                # ROC曲線
                fpr, tpr, _ = roc_curve(y_test, y_prob)
                roc_auc = auc(fpr, tpr)

                eval_data_size = len(y_test)
                eval_pos = y_test.sum()
                eval_neg = len(y_test) - y_test.sum()
            else:
                # 全データで学習・評価
                clf = LogisticRegression(random_state=42, max_iter=a.max_iter)
                clf.fit(X_clean, y_clean)

                # 全データで予測確率
                y_prob = clf.predict_proba(X_clean)[:, 1]

                # ROC曲線
                fpr, tpr, _ = roc_curve(y_clean, y_prob)
                roc_auc = auc(fpr, tpr)

                eval_data_size = len(y_clean)
                eval_pos = y_clean.sum()
                eval_neg = len(y_clean) - y_clean.sum()

            # 結果保存
            results[target_col] = {
                'auc': roc_auc,
                'n_samples': len(y_clean),
                'n_positive': y_clean.sum(),
                'n_negative': len(y_clean) - y_clean.sum(),
                'eval_samples': eval_data_size,
                'eval_positive': eval_pos,
                'eval_negative': eval_neg,
                'train_test_split': a.use_train_test_split,
                'status': 'success'
            }

            # プロット
            if plot_idx < len(axes):
                split_info = f"(Train-Test)" if a.use_train_test_split else "(Full Data)"
                axes[plot_idx].plot(fpr, tpr, color='darkorange', lw=2,
                                  label=f'ROC curve (AUC = {roc_auc:.3f})')
                axes[plot_idx].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                axes[plot_idx].set_xlim([0.0, 1.0])
                axes[plot_idx].set_ylim([0.0, 1.05])
                axes[plot_idx].set_xlabel('False Positive Rate')
                axes[plot_idx].set_ylabel('True Positive Rate')
                axes[plot_idx].set_title(f'{target_col} {split_info}\n(eval: n={eval_data_size}, pos={eval_pos})')
                axes[plot_idx].legend(loc="lower right")
                axes[plot_idx].grid(True, alpha=0.3)

            plot_idx += 1
            print(f"  AUC: {roc_auc:.3f}")

        # 未使用のsubplotを非表示
        for j in range(plot_idx, len(axes)):
            axes[j].set_visible(False)

        plt.tight_layout()

        # ファイル名に分割方法を含める
        if a.use_train_test_split:
            suffix = "regression_train_test"
        else:
            suffix = "regression_full_data"

        plt.savefig(output_dir / f'roc_curves_{suffix}.png', dpi=300, bbox_inches='tight')
        plt.close()

        # 結果をCSVで保存
        results_df = pd.DataFrame(results).T
        results_df.to_csv(output_dir / f'regression_results_{suffix}.csv')

        print(f"\n=== {a.dataset.upper()} ROC分析結果 ===")
        print(results_df.round(3))

        return results

    class LassoAnalysisArgs(CommonArgs):
        """Lasso正則化ROC分析引数"""
        output_dir: str = param('', description="出力ディレクトリ (デフォルト: out/{dataset}/lasso)")
        max_iter: int = param(2000, description="ロジスティック回帰の最大反復回数")
        use_train_test_split: bool = param(False, description="Train-test分割を使用するか（Falseで全データ学習）")
        test_size: float = param(0.3, description="テストデータの割合（use_train_test_split=Trueの場合）")
        cv_folds: int = param(5, description="交差検証のfold数（Lasso正則化パラメータ選択用）")

    def run_lasso_analysis(self, a: LassoAnalysisArgs):
        """L1正則化（Lasso）でROC分析を実行"""

        print(f"Lasso ROC分析開始: dataset={a.dataset}")

        if self.merged_data is None:
            print(f"データの読み込みに失敗しました: {a.dataset}")
            return

        if not a.output_dir:
            a.output_dir = f'out/{a.dataset}/lasso'

        output_dir = Path(a.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"データ形状: {self.merged_data.shape}")
        print(f"特徴量数: {len(self.feature_cols)}")
        print(f"分析対象カラム ({len(self.target_cols)}): {self.target_cols}")

        # 特徴量を標準化
        scaler = StandardScaler()
        X = scaler.fit_transform(self.merged_data[self.feature_cols])

        results = {}

        # ROC曲線プロット準備
        n_cols = min(4, len(self.target_cols))
        n_rows = (len(self.target_cols) + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
        if n_rows == 1 and n_cols == 1:
            axes = [axes]
        elif n_rows == 1 or n_cols == 1:
            axes = axes.flatten()
        else:
            axes = axes.flatten()

        plot_idx = 0

        for target_col in self.target_cols:
            print(f"\n--- {target_col} のLasso分析 ---")

            # 欠損値を除去
            mask = ~self.merged_data[target_col].isna()
            X_clean = X[mask]
            y_raw = self.merged_data[target_col][mask].copy()

            # データセット別バイナリ化
            y_clean = self.apply_thresholds(y_raw, target_col, a.dataset)

            print(f"  サンプル数: {len(y_clean)}")
            print(f"  正例: {y_clean.sum()}, 負例: {len(y_clean) - y_clean.sum()}")

            # 単一クラスまたは極端な不均衡の場合はスキップ
            if y_clean.sum() <= 1 or (len(y_clean) - y_clean.sum()) <= 1:
                print(f"  スキップ: {target_col} は解析不可能なクラス分布です")
                results[target_col] = {
                    'auc': np.nan,
                    'n_samples': len(y_clean),
                    'n_positive': y_clean.sum(),
                    'n_negative': len(y_clean) - y_clean.sum(),
                    'status': 'insufficient_class',
                    'best_C': np.nan,
                    'n_features_selected': np.nan
                }
                continue

            if a.use_train_test_split:
                # Train-test分割
                from sklearn.model_selection import train_test_split
                X_train, X_test, y_train, y_test = train_test_split(
                    X_clean, y_clean, test_size=a.test_size,
                    random_state=42, stratify=y_clean
                )

                # Lasso（L1正則化）で学習 - 交差検証でC選択
                clf = LogisticRegressionCV(
                    penalty='l1', solver='liblinear', cv=a.cv_folds,
                    random_state=42, max_iter=a.max_iter
                )
                clf.fit(X_train, y_train)

                # テストデータで予測
                y_prob = clf.predict_proba(X_test)[:, 1]

                # ROC曲線
                fpr, tpr, _ = roc_curve(y_test, y_prob)
                roc_auc = auc(fpr, tpr)

                eval_data_size = len(y_test)
                eval_pos = y_test.sum()
                eval_neg = len(y_test) - y_test.sum()
            else:
                # 全データで学習・評価
                clf = LogisticRegressionCV(
                    penalty='l1', solver='liblinear', cv=a.cv_folds,
                    random_state=42, max_iter=a.max_iter
                )
                clf.fit(X_clean, y_clean)

                # 全データで予測確率
                y_prob = clf.predict_proba(X_clean)[:, 1]

                # ROC曲線
                fpr, tpr, _ = roc_curve(y_clean, y_prob)
                roc_auc = auc(fpr, tpr)

                eval_data_size = len(y_clean)
                eval_pos = y_clean.sum()
                eval_neg = len(y_clean) - y_clean.sum()

            # 特徴量選択の効果を確認
            n_features_selected = np.sum(np.abs(clf.coef_[0]) > 1e-6)
            best_C = clf.C_[0] if hasattr(clf, 'C_') else np.nan

            # 結果保存
            results[target_col] = {
                'auc': roc_auc,
                'n_samples': len(y_clean),
                'n_positive': y_clean.sum(),
                'n_negative': len(y_clean) - y_clean.sum(),
                'eval_samples': eval_data_size,
                'eval_positive': eval_pos,
                'eval_negative': eval_neg,
                'train_test_split': a.use_train_test_split,
                'status': 'success',
                'best_C': best_C,
                'n_features_selected': n_features_selected,
                'total_features': len(self.feature_cols)
            }

            # プロット
            if plot_idx < len(axes):
                split_info = f"(Train-Test)" if a.use_train_test_split else "(Full Data)"
                axes[plot_idx].plot(fpr, tpr, color='darkorange', lw=2,
                                  label=f'Lasso ROC (AUC = {roc_auc:.3f})')
                axes[plot_idx].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                axes[plot_idx].set_xlim([0.0, 1.0])
                axes[plot_idx].set_ylim([0.0, 1.05])
                axes[plot_idx].set_xlabel('False Positive Rate')
                axes[plot_idx].set_ylabel('True Positive Rate')
                axes[plot_idx].set_title(f'{target_col} Lasso {split_info}\n(eval: n={eval_data_size}, features={n_features_selected}/{len(self.feature_cols)})')
                axes[plot_idx].legend(loc="lower right")
                axes[plot_idx].grid(True, alpha=0.3)

            plot_idx += 1
            print(f"  AUC: {roc_auc:.3f}, 選択特徴量数: {n_features_selected}/{len(self.feature_cols)}, C: {best_C:.6f}")

        # 未使用のsubplotを非表示
        for j in range(plot_idx, len(axes)):
            axes[j].set_visible(False)

        plt.tight_layout()

        # ファイル名に分割方法を含める
        if a.use_train_test_split:
            suffix = "lasso_train_test"
        else:
            suffix = "lasso_full_data"

        plt.savefig(output_dir / f'roc_curves_{suffix}.png', dpi=300, bbox_inches='tight')
        plt.close()

        # 結果をCSVで保存
        results_df = pd.DataFrame(results).T
        results_df.to_csv(output_dir / f'lasso_results_{suffix}.csv')

        print(f"\n=== {a.dataset.upper()} Lasso ROC分析結果 ===")
        print(results_df.round(3))

        return results

    class FeatureOverlapArgs(CommonArgs):
        """特徴量重複分析引数"""
        target_column: str = param('', description="分析対象カラム（例: MUM1 IHC）")
        output_dir: str = param('out/lasso', description="出力ディレクトリ")
        max_iter: int = param(2000, description="ロジスティック回帰の最大反復回数")
        cv_folds: int = param(5, description="交差検証のfold数")
        morph_dataset: str = param('morph', description="Morphデータセット名")
        patho2_dataset: str = param('patho2', description="Patho2データセット名")

    def run_feature_overlap_analysis(self, a: FeatureOverlapArgs):
        """指定されたカラムでMorphとPatho2の特徴量選択重複を分析"""
        
        print(f"特徴量重複分析開始: target_column={a.target_column}")
        
        # 出力ディレクトリ作成
        output_dir = Path(a.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 両データセットを読み込み
        try:
            morph_data, patho2_data = load_both_datasets()
        except RuntimeError as e:
            print(f"データセット読み込みエラー: {e}")
            return
            
        # 両データセットでLasso分析を実行し特徴量選択結果を取得
        morph_features, patho2_features = self._get_selected_features_for_datasets(
            a.target_column, morph_data, patho2_data, a.max_iter, a.cv_folds
        )
        
        if morph_features is None or patho2_features is None:
            print("特徴量選択の取得に失敗しました")
            return
            
        # ベン図作成
        self._create_venn_diagram(
            morph_features, patho2_features, a.target_column, output_dir
        )
        
        # 重複分析結果をCSVで保存
        self._save_overlap_analysis(
            morph_features, patho2_features, a.target_column, output_dir
        )
        
    def _get_selected_features_for_datasets(self, target_column, morph_data, patho2_data, max_iter, cv_folds):
        """両データセットで指定カラムのLasso特徴量選択を実行"""
        
        results = {}
        datasets = [('morph', morph_data), ('patho2', patho2_data)]
        
        for dataset_name, data in datasets:
            print(f"\n--- {dataset_name.upper()}データセットでLasso特徴量選択 ---")
            
            # 特徴量カラムを取得（feature_で始まる列のみ）
            feature_cols = [col for col in data.columns if col.startswith('feature_')]
            
            if target_column not in data.columns:
                print(f"{target_column}が{dataset_name}データに存在しません")
                return None, None
                
            # 欠損値を除去
            mask = ~data[target_column].isna()
            X_clean = data[feature_cols][mask]
            y_raw = data[target_column][mask].copy()
            
            # 特徴量にNaNがある場合は完全に除去
            feature_mask = ~X_clean.isna().any(axis=1)
            X_clean = X_clean[feature_mask]
            y_raw = y_raw[feature_mask]
            
            # データセット別バイナリ化
            y_clean = self.apply_thresholds(y_raw, target_column, dataset_name)
            
            print(f"  サンプル数: {len(y_clean)}")
            print(f"  正例: {y_clean.sum()}, 負例: {len(y_clean) - y_clean.sum()}")
            
            # 解析不可能な場合はスキップ
            if y_clean.sum() <= 1 or (len(y_clean) - y_clean.sum()) <= 1:
                print(f"  スキップ: {target_column}は解析不可能なクラス分布です")
                return None, None
                
            # 特徴量標準化
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_clean)
            
            # Lasso回帰で特徴量選択
            clf = LogisticRegressionCV(
                penalty='l1', solver='liblinear', cv=cv_folds,
                random_state=42, max_iter=max_iter
            )
            clf.fit(X_scaled, y_clean)
            
            # 選択された特徴量のインデックスを取得
            selected_indices = np.where(np.abs(clf.coef_[0]) > 1e-6)[0]
            selected_features = [feature_cols[i] for i in selected_indices]
            
            print(f"  選択特徴量数: {len(selected_features)}/{len(feature_cols)}")
            print(f"  最適C: {clf.C_[0]:.6f}")
            
            results[dataset_name] = {
                'selected_features': selected_features,
                'selected_indices': selected_indices,
                'coefficients': clf.coef_[0][selected_indices],
                'feature_cols': feature_cols,
                'best_C': clf.C_[0],
                'n_selected': len(selected_features),
                'n_total': len(feature_cols)
            }
            
        return results['morph'], results['patho2']
        
    def _create_venn_diagram(self, morph_features, patho2_features, target_column, output_dir):
        """ベン図を作成して特徴量の重複を可視化"""
        
        morph_set = set(morph_features['selected_features'])
        patho2_set = set(patho2_features['selected_features'])
        
        # 重複情報
        overlap = morph_set & patho2_set
        morph_only = morph_set - patho2_set
        patho2_only = patho2_set - morph_set
        
        print(f"\n--- {target_column} 特徴量重複分析結果 ---")
        print(f"Morph専用特徴量: {len(morph_only)}")
        print(f"Patho2専用特徴量: {len(patho2_only)}")
        print(f"共通特徴量: {len(overlap)}")
        print(f"総特徴量: {len(morph_set | patho2_set)}")
        
        if len(overlap) > 0:
            print(f"共通特徴量: {sorted(list(overlap))}")
        
        # ベン図作成
        plt.figure(figsize=(10, 8))
        
        venn = venn2(
            [morph_set, patho2_set],
            set_labels=[f'Morph\n({len(morph_set)} features)', f'Patho2\n({len(patho2_set)} features)']
        )
        
        # 色設定
        venn.get_patch_by_id('10').set_color('lightblue')
        venn.get_patch_by_id('01').set_color('lightcoral')
        if venn.get_patch_by_id('11'):
            venn.get_patch_by_id('11').set_color('lightgreen')
        
        plt.title(f'{target_column}: Feature Selection Overlap\n'
                 f'Morph: {morph_features["n_selected"]} features, '
                 f'Patho2: {patho2_features["n_selected"]} features, '
                 f'Overlap: {len(overlap)} features',
                 fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        filename = f'{target_column.replace(" ", "_").lower()}_feature_overlap_venn.png'
        plt.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"ベン図を保存しました: {output_dir / filename}")
        
    def _save_overlap_analysis(self, morph_features, patho2_features, target_column, output_dir):
        """重複分析結果をCSVファイルに保存"""
        
        morph_set = set(morph_features['selected_features'])
        patho2_set = set(patho2_features['selected_features'])
        
        # 全特徴量リスト作成
        all_features = sorted(list(morph_set | patho2_set))
        
        # 結果DataFrame作成
        results_data = []
        
        for feature in all_features:
            in_morph = feature in morph_set
            in_patho2 = feature in patho2_set
            
            # 係数情報
            morph_coef = 0.0
            patho2_coef = 0.0
            
            if in_morph:
                idx = morph_features['selected_features'].index(feature)
                morph_coef = morph_features['coefficients'][idx]
                
            if in_patho2:
                idx = patho2_features['selected_features'].index(feature)
                patho2_coef = patho2_features['coefficients'][idx]
            
            results_data.append({
                'feature': feature,
                'in_morph': in_morph,
                'in_patho2': in_patho2,
                'in_both': in_morph and in_patho2,
                'morph_coefficient': morph_coef,
                'patho2_coefficient': patho2_coef,
                'coefficient_diff': abs(morph_coef - patho2_coef) if (in_morph and in_patho2) else np.nan
            })
        
        results_df = pd.DataFrame(results_data)
        
        # CSV保存
        csv_filename = f'{target_column.replace(" ", "_").lower()}_feature_overlap_analysis.csv'
        csv_path = output_dir / csv_filename
        results_df.to_csv(csv_path, index=False)
        
        # サマリー統計を追加保存
        summary_data = {
            'target_column': target_column,
            'morph_total_features': morph_features['n_total'],
            'morph_selected_features': morph_features['n_selected'],
            'morph_best_C': morph_features['best_C'],
            'patho2_total_features': patho2_features['n_total'],
            'patho2_selected_features': patho2_features['n_selected'],
            'patho2_best_C': patho2_features['best_C'],
            'overlap_features': len(morph_set & patho2_set),
            'morph_only_features': len(morph_set - patho2_set),
            'patho2_only_features': len(patho2_set - morph_set),
            'jaccard_similarity': len(morph_set & patho2_set) / len(morph_set | patho2_set) if len(morph_set | patho2_set) > 0 else 0
        }
        
        summary_df = pd.DataFrame([summary_data])
        summary_filename = f'{target_column.replace(" ", "_").lower()}_overlap_summary.csv'
        summary_path = output_dir / summary_filename
        summary_df.to_csv(summary_path, index=False)
        
        print(f"重複分析結果を保存しました: {csv_path}")
        print(f"サマリー統計を保存しました: {summary_path}")
        
        # 結果表示
        print(f"\n=== {target_column} 特徴量重複分析サマリー ===")
        print(f"Jaccard類似度: {summary_data['jaccard_similarity']:.3f}")
        print(f"Morph選択率: {morph_features['n_selected']}/{morph_features['n_total']} ({morph_features['n_selected']/morph_features['n_total']:.3f})")
        print(f"Patho2選択率: {patho2_features['n_selected']}/{patho2_features['n_total']} ({patho2_features['n_selected']/patho2_features['n_total']:.3f})")


if __name__ == '__main__':
    cli = CLI()
    cli.run()
