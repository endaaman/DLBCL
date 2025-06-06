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
└── statistics/                        # 統計解析結果（既存）
```

## 技術スタック

- **基盤モデル**: Prov-GigaPath（768次元slide_feature）
- **クラスタリング**: HDBSCAN
- **次元削減**: UMAP
- **生存解析**: lifelines（Kaplan-Meier, log-rank test）
- **統計解析**: scipy, statsmodels
- **開発フレームワーク**: pydantic-autocli

## 利用可能なコマンド

```bash
# データ準備
uv run python -m dlbcl.main gather-slide-features --dataset [morph|patho2]

# 基本解析
uv run python -m dlbcl.main cluster --dataset [morph|patho2] --target [cluster|HANS|Age|...]
uv run python -m dlbcl.main integrated-cluster
uv run python -m dlbcl.main statistical-analysis --dataset [morph|patho2]

# 生存解析（Morphのみ）
uv run python -m dlbcl.main survival-analysis --dataset morph [--cluster-based]

# その他
uv run python -m dlbcl.main comprehensive-heatmap --dataset [morph|patho2]
uv run python -m dlbcl.main pathology-analysis --dataset [morph|patho2]
```

## 今後の展開

1. **特徴量解析の詳細化**: 個別特徴量の詳細解析
2. **パッチレベル解析**: WSI代表性の検証
3. **多変量解析**: Cox回帰等の高度統計手法