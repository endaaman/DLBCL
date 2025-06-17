import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import StandardScaler
from pathlib import Path
from pydantic_autocli import param

from .utils import BaseMLCLI


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


if __name__ == '__main__':
    cli = CLI()
    cli.run()