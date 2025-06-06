# DLBCL 病理画像解析プロジェクト

## 研究背景

SSL（Self-Supervised Learning）で訓練された病理組織基盤モデル（Prov-GigaPath）によって、病理画像から高品質な埋め込み情報を得ることができるようになった。本研究では、DLBCL（Diffuse Large B-cell Lymphoma：びまん性大細胞型B細胞リンパ腫）の病理組織画像から抽出された特徴量と臨床データとの関連を統計的手法で解析し、新たな形態病理学的知見を得ることを目指す。

## データセット

### DLBCL-Morph（メインデータセット）
- **症例数**: 149例、**WSI数**: 202枚
- **臨床データ**: 充実（免疫染色、FISH、予後情報）
- **状況**: 解析完了 ✅

### DLBCL-Patho2（検証用データセット）
- **症例数**: 64例、**WSI数**: 96枚  
- **臨床データ**: 限定的（主に免疫染色）
- **状況**: 解析完了 ✅

## アーキテクチャ

### モジュール構成
本プロジェクトは以下のモジュールに分離されています：

- **`dlbcl.main`**: 基本的な解析機能（クラスタリング、UMAP、統計解析、ヒートマップ）
- **`dlbcl.feature`**: 特徴量解析と統計的検証機能
- **`dlbcl.utils`**: 共通処理（データ読み込み、CLI基盤、シード管理）

### 共通基盤
- **BaseMLCLI**: 共通のCLI基盤クラス
- **共通データローダー**: 臨床データと特徴量データの統一的読み込み
- **共通パラメータ**: `--dataset`, `--seed`, `--device`などの統一パラメータ

## 実装済み機能

### 1. クラスタリング解析
**HDBSCAN**を用いた教師なしクラスタリング：
- **Morph**: 2クラスター（シルエット係数: 0.178）
- **Patho2**: 2クラスター（シルエット係数: -0.021）
- **統合解析**: 両データセット結合でのクラスタリング（297サンプル）

```bash
# 個別データセットのクラスタリング
uv run python -m dlbcl.main cluster --dataset morph --target cluster
uv run python -m dlbcl.main cluster --dataset patho2 --target cluster

# 統合クラスタリング
uv run python -m dlbcl.main integrated-cluster
```

### 2. UMAP可視化
**768次元特徴量**の2次元可視化：
- 臨床変数別の分布パターン可視化
- クラスター結果の可視化
- データセット間比較

```bash
# 臨床変数別可視化（例）
uv run python -m dlbcl.main cluster --dataset morph --target HANS
uv run python -m dlbcl.main cluster --dataset morph --target "Age"
uv run python -m dlbcl.main cluster --dataset morph --target "IPI Risk Group (4 Class)"
```

### 3. 統計解析
**相関解析・群間比較**による特徴量-臨床関連の検出：
- 多重検定補正（FDR）適用
- 有意な相関関係: 32個（Morph）
- 有意な群間差: 17個（Morph）

```bash
# 統計解析実行
uv run python -m dlbcl.main statistical-analysis --dataset morph
uv run python -m dlbcl.main statistical-analysis --dataset patho2
```

### 4. 生存解析 🆕
**Kaplan-Meier曲線**とログランク検定：
- 全体生存率（OS）・無増悪生存率（PFS）
- 臨床変数別生存解析
- クラスター別生存解析

```bash
# 生存解析実行
uv run python -m dlbcl.main survival-analysis --dataset morph
uv run python -m dlbcl.main survival-analysis --dataset morph --cluster-based
```

### 5. 特徴量解析と統計的検証 🆕
**`dlbcl.feature`**モジュールによる高度な統計的検証：
- 多重検定問題の評価（FDR補正）
- 効果サイズの詳細評価（Cohen's基準）
- 順列検定による偶然性検証
- Bootstrap法による頑健性評価
- 包括的な統計的妥当性レポート

```bash
# 解析妥当性評価（推奨: 計算集約的）
uv run python -m dlbcl.feature analysis-validation --dataset morph --n-permutations 1000

# 短縮版テスト実行
uv run python -m dlbcl.feature analysis-validation --dataset morph --n-permutations 10
```

## 主要解析結果

### クラスタリング結果
- **Morph**: ノイズポイント55/202サンプル、明確な2クラスター形成
- **統合解析**: データセット間で明確な分布差を確認
- **クラスター品質**: Morphで良好、Patho2では要改善

### 生存解析結果（Morph）
- **年齢**: 高齢群で有意に予後不良（p<0.0001）
- **LDH**: 高値群で予後良好傾向（p=0.0016, 0.0008）
- **HANS分類**: 有意差なし（p=0.96, 0.76）
- **フォローアップ**: 中央値7.54ヶ月、イベント数74/202

### 統計解析結果（既存）
- **feature_740**: 最強の予後予測因子（PFS: r=-0.35, p=0.0015）
- **BCL6 IHC**: 複数特徴量と中程度相関（r=0.29-0.31）
- **CD10 IHC**: 負の相関パターン確認

### 統計的検証結果（新規） 🆕
**`dlbcl.feature analysis-validation`**による妥当性評価：
- **総検定数**: 5,376回（768特徴量 × 7臨床変数）
- **期待偽陽性**: 268.8個 vs **実際有意**: 742個（2.8倍のインフレーション）
- **FDR補正後有意**: 19個（大幅減少）
- **大効果サイズ**: 0個、**中効果サイズ**: 4個のみ
- **順列検定**: パラメトリック結果の多くが人工的
- **頑健性**: 良好（Bootstrap CI幅平均: 0.262）

## 出力ファイル構造

```
out/
├── morph/
│   ├── umap_*.png                     # UMAP可視化（15+ファイル）
│   ├── hdbscan_clustering_results.csv # クラスタリング結果
│   └── survival/
│       ├── OS_overall.png             # 全体生存曲線
│       ├── PFS_overall.png            # 無増悪生存曲線
│       ├── OS_Age.png                 # 年齢別生存解析
│       └── PFS_HANS.png               # HANS別生存解析
├── patho2/
│   ├── umap_*.png                     # Patho2 UMAP可視化
│   └── hdbscan_clustering_results.csv
├── integrated/
│   ├── integrated_clustering.png      # 統合クラスタリング結果
│   └── integrated_clustering_results.csv
├── statistics/                        # 統計解析結果（既存）
├── comprehensive_heatmap/              # ヒートマップ解析
└── analysis_validation/                # 統計的検証結果 🆕
    ├── comprehensive_validation_report.png  # 包括的評価レポート
    ├── multiple_testing_evaluation.csv      # 多重検定評価
    ├── effect_size_evaluation.csv           # 効果サイズ評価  
    ├── permutation_test_evaluation.csv      # 順列検定結果
    ├── robustness_evaluation.csv            # 頑健性評価
    └── validation_summary_stats.csv         # サマリー統計
```

## 技術スタック

- **基盤モデル**: Prov-GigaPath（768次元slide_feature）
- **クラスタリング**: HDBSCAN
- **次元削減**: UMAP
- **生存解析**: lifelines（Kaplan-Meier, log-rank test）
- **統計解析**: scipy, statsmodels
- **開発フレームワーク**: pydantic-autocli

## 利用可能なコマンド

### dlbcl.main（基本解析）
```bash
# データ準備
uv run python -m dlbcl.main gather-slide-features --dataset [morph|patho2]

# 基本解析
uv run python -m dlbcl.main cluster --dataset [morph|patho2] --target [cluster|HANS|Age|...]
uv run python -m dlbcl.main umap --dataset [morph|patho2] --keys ["DBSCAN cluster"]
uv run python -m dlbcl.main statistical-analysis --dataset [morph|patho2]
uv run python -m dlbcl.main comprehensive-heatmap --dataset [morph|patho2]

# 生存解析（Morphのみ）
uv run python -m dlbcl.main survival-analysis --dataset morph [--cluster-based]
```

### dlbcl.feature（統計的検証）
```bash
# 包括的統計的妥当性評価
uv run python -m dlbcl.feature analysis-validation \
    --dataset [morph|patho2] \
    --n-permutations 1000 \
    --effect-size-threshold 0.3 \
    --fdr-alpha 0.05 \
    --output-dir out/analysis_validation
```

### 共通パラメータ
- `--dataset [morph|patho2]`: 使用データセット
- `--seed 42`: 再現性のためのシード値  
- `--device cuda`: GPU使用設定

## 今後の展開

1. **統計的妥当性の向上**: FDR補正後の有意な特徴量の生物学的検証
2. **効果サイズに基づく解析**: 中効果以上の特徴量に焦点を当てた詳細解析
3. **パッチレベル解析**: WSI代表性の検証
4. **多変量解析**: Cox回帰等の高度統計手法
5. **独立検証**: 外部データセットでの妥当性確認

## 重要な注意事項

⚠️ **統計的検証結果に基づく推奨事項**:
- **多重検定の影響**: 5,376回の検定で742個の「有意」結果は、期待値（269個）の2.8倍
- **FDR補正**: 補正後は19個まで減少、これらに焦点を当てるべき
- **効果サイズ**: 大効果サイズ(|r|≥0.5)は0個、解釈には慎重さが必要
- **順列検定**: パラメトリック検定結果の多くが人工的パターンを示唆