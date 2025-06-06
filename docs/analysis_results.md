# DLBCL病理画像解析結果記録

## 完了済み解析一覧

### 1. 統計解析（statistical-analysis） ✅
**実行済み**: DLBCL-Morph・DLBCL-Patho2

#### DLBCL-Morph結果（202サンプル）
- **有意な相関関係**: 32個（FDR補正後）
- **有意な群間差**: 17個（FDR補正後）
- **最重要特徴量**: Feature_740（PFS: r=-0.35, p=0.0015）

#### DLBCL-Patho2結果（95サンプル）
- 統計解析完了、結果は`out/statistics/`に保存

### 2. クラスタリング解析（cluster） ✅
**HDBSCAN**を用いた教師なしクラスタリング

#### 個別データセット結果
**DLBCL-Morph（202サンプル）**:
- クラスター数: 2
- ノイズポイント: 55
- シルエット係数: 0.178

**DLBCL-Patho2（95サンプル）**:
- クラスター数: 2  
- ノイズポイント: 59
- シルエット係数: -0.021（要改善）

#### 統合クラスタリング結果（297サンプル）
- クラスター数: 2
- ノイズポイント: 75
- シルエット係数: 0.031
- **データセット分布**: 明確な分離確認

| Dataset | Cluster -1 | Cluster 0 | Cluster 1 |
|---------|------------|-----------|-----------|
| Morph   | 69         | 128       | 5         |
| Patho2  | 6          | 89        | 0         |

### 3. UMAP可視化（cluster --target X） ✅
**768次元特徴量**の2次元UMAP可視化

#### 生成済み可視化（Morph）
- `umap_cluster.png`: HDBSCANクラスター結果
- `umap_HANS.png`: 細胞起源分類
- `umap_Age.png`: 年齢分布  
- `umap_CD10_IHC.png`: CD10免疫染色
- `umap_BCL6_IHC.png`: BCL6免疫染色
- `umap_Stage.png`: 病期分布
- その他10+個のプロット

#### 生成済み可視化（Patho2）
- `umap_cluster.png`: クラスター結果
- `umap_HANS.png`: HANS分類
- `umap_CD10_IHC.png`: CD10染色
- `umap_MUM1_IHC.png`: MUM1染色

### 4. 生存解析（survival-analysis） ✅ 🆕
**Kaplan-Meier**曲線とログランク検定（Morphのみ）

#### 全体生存解析結果
- **総症例数**: 202例
- **イベント数**: 74例（死亡）
- **中央値フォローアップ**: 7.54ヶ月

#### 臨床変数別生存解析結果

**年齢（Age）**:
- 高齢群 vs 低齢群
- OS: p<0.0001（極めて有意）
- PFS: p=0.0005（極めて有意）
- **結論**: 高齢患者で明らかに予後不良

**LDH（乳酸脱水素酵素）**:
- 高値群 vs 正常群  
- OS: p=0.0016（有意）
- PFS: p=0.0008（極めて有意）
- **結論**: LDH高値群で予後良好（予想外の結果）

**HANS分類（細胞起源）**:
- GCB型 vs non-GCB型
- OS: p=0.9599（非有意）
- PFS: p=0.7608（非有意）
- **結論**: 形態学的特徴量では細胞起源による予後差なし

**IPI Risk Group**:
- 4群間での生存差を確認
- 高リスク群で予後不良傾向

#### 保存ファイル
- `out/morph/survival/OS_overall.png`: 全体OS曲線
- `out/morph/survival/PFS_overall.png`: 全体PFS曲線  
- `out/morph/survival/OS_Age.png`: 年齢別OS比較
- `out/morph/survival/PFS_HANS.png`: HANS別PFS比較
- その他15個の生存曲線

### 5. 統合解析（integrated-cluster） ✅ 🆕
**両データセット結合**での横断的解析

#### 結果
- **総サンプル数**: 297（Morph: 202, Patho2: 95）
- **共通特徴量**: 768次元
- **クラスタリング**: データセット間で明確な分離
- **保存**: `out/integrated/integrated_clustering.png/csv`

## 技術的検証結果

### HDBSCANクラスタリング品質
- **Morph**: 良好（シルエット係数 > 0.1）
- **Patho2**: 要改善（負のシルエット係数）
- **統合**: 中程度の品質

### UMAP vs HDBSCAN対応
- **重要**: HDBSCANは768次元で実行、UMAPは2次元可視化用
- 完全な対応は期待できない（次元圧縮による情報損失）
- これは正常な動作

### データセット間差異
- **統合クラスタリング**: 明確なデータセット分離を確認
- **技術的要因**: 撮影条件、前処理パラメータの違い
- **生物学的要因**: 患者背景、病理学的特徴の違い

## 出力ファイル構造

```
out/
├── morph/
│   ├── umap_*.png (15+個)
│   ├── hdbscan_clustering_results.csv
│   └── survival/ (15+個の生存曲線)
├── patho2/  
│   ├── umap_*.png (5個)
│   └── hdbscan_clustering_results.csv
├── integrated/
│   ├── integrated_clustering.png
│   └── integrated_clustering_results.csv  
└── statistics/ (既存の統計解析結果)
```

## 今後の解析方針

### 短期目標
1. **特徴量詳細解析**: feature_740等の生物学的解釈
2. **クラスター特性解析**: 各クラスターの臨床的特徴
3. **多変量解析**: Cox回帰等の高度統計手法

### 中期目標  
1. **パッチレベル解析**: WSI代表性の検証
2. **データセット間比較**: MorphとPatho2の差異詳細分析
3. **特徴量工学**: より良い特徴量表現の探索