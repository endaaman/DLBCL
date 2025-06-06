# DLBCL病理画像解析結果記録

## 実行済み解析一覧

### 1. 統計解析（statistical-analysis）
- **実行日**: 2024年12月
- **データセット**: DLBCL-Morph（149症例、202WSI）
- **概要**: slide_feature（768次元）と臨床データの包括的統計解析

#### 主要結果
- **有意な相関関係**: 7個（FDR補正後 p < 0.05）
- **有意な群間差**: 20個（FDR補正後 p < 0.05）
- **最重要特徴量**: Feature_740（PFS: r=-0.353, p=0.0023）

#### 保存ファイル
- `out/statistics_debug/correlation_analysis_corrected.csv`
- `out/statistics_debug/group_comparison_analysis_corrected.csv`
- `out/statistics_debug/summary_statistics.csv`

### 2. 包括的ヒートマップ解析（comprehensive-heatmap）
- **実行日**: 2024年12月
- **概要**: 768特徴量×20臨床変数の相関ヒートマップ（デンドログラム付き）

#### 主要結果
- **相関範囲**: -0.35 ～ +0.32
- **平均絶対相関**: 0.071
- **階層クラスタリング**: Ward法による特徴量グループ化

#### 保存ファイル
- `out/comprehensive_heatmap/comprehensive_heatmap.png/pdf`
- `out/comprehensive_heatmap/significant_correlations_heatmap.png`
- `out/comprehensive_heatmap/correlation_matrix.csv`

### 3. 注目特徴量解析（feature-clinical-analysis）
- **実行日**: 2024年12月
- **概要**: 重要特徴量と臨床データの詳細可視化・解析

#### 対象特徴量
- Feature_740：最重要予後関連特徴量
- Feature_371：CD10関連特徴量
- Feature_221：HANS関連特徴量
- Feature_630、745：OS関連特徴量

#### 主要結果
**Feature_740詳細解析**
- 総症例数：202
- PFS相関：r=-0.353, p=2.49×10⁻⁷（極めて強い負の相関）
- OS相関：r=-0.300, p=1.43×10⁻⁵（強い負の相関）
- 分布：平均=-0.277, 標準偏差=0.152

**予後リスクスコア開発**
- Feature_740（重み3）、Feature_630（重み1）、Feature_745（重み1）で構成
- 標準化後の重み付きスコアとして実装

#### 保存ファイル
- `out/feature_clinical_analysis/feature_740_analysis.png/pdf`：Feature_740の包括的解析
- `out/feature_clinical_analysis/cell_origin_analysis.png/pdf`：細胞起源関連特徴量の解析
- `out/feature_clinical_analysis/prognosis_features_correlation.png/pdf`：予後関連特徴量の相関解析
- `out/feature_clinical_analysis/prognosis_risk_score.png/pdf`：予後リスクスコアの可視化
- `out/feature_clinical_analysis/correlation_network.png/pdf`：相関ネットワーク図
- `out/feature_clinical_analysis/analysis_dashboard.png/pdf`：統合ダッシュボード
- `out/feature_clinical_analysis/feature_740_summary.txt`：Feature_740の数値サマリー

## 重要な発見と仮説

### キー特徴量：Feature_740
**統計的証拠:**
- PFS（無増悪生存期間）：r = -0.353, p = 0.0023
- OS（全生存期間）：r = -0.300, p = 0.0403
- LDH（腫瘍負荷）：r = 0.324, p = 0.0149

**病理学的解釈:**
Feature_740高値症例は予後不良型パターンを示し、より攻撃的な腫瘍形態を反映している可能性。

### 細胞起源関連特徴量
**CD10 IHC vs Feature_371:**
- 相関係数：r = -0.304, p = 0.0403
- 解釈：GCB型とnon-GCB型の形態学的差異を反映

**HANS分類 vs Feature_221:**
- 相関係数：r = 0.307, p = 0.0403
- 解釈：細胞起源分類と組織学的パターンの関連

### 腫瘍負荷関連パターン
**LDH関連特徴量群:**
6個の特徴量（feature_740, 502, 382, 398, 721, 654）がLDHと有意に関連

**節外病変関連特徴量群:**
4個の特徴量（feature_703, 90, 643, 137）が節外病変数（EN）と関連

## 利用可能なコマンド

### 実行可能なautocliコマンド

```bash
# 統計解析
uv run python -m dlbcl.main statistical-analysis --dataset morph --alpha 0.05

# 包括的ヒートマップ
uv run python -m dlbcl.main comprehensive-heatmap --dataset morph

# 注目特徴量解析
uv run python -m dlbcl.main feature-clinical-analysis --dataset morph

# 特定特徴量に絞った解析
uv run python -m dlbcl.main feature-clinical-analysis --dataset morph \
  --target-features feature_740 feature_371 \
  --target-clinical PFS OS LDH
```

### パラメータ詳細

#### statistical-analysis
- `--dataset`: 'morph' または 'patho2'
- `--alpha`: 有意水準（デフォルト：0.05）

#### comprehensive-heatmap  
- `--dataset`: 'morph' または 'patho2'

#### feature-clinical-analysis
- `--dataset`: 'morph' または 'patho2'
- `--target-features`: 解析対象特徴量リスト（オプション）
- `--target-clinical`: 解析対象臨床変数リスト（オプション）

## 今後の解析計画

### Phase 2: 多変量解析
- Cox回帰によるFeature_740の独立予後価値検証
- ロジスティック回帰による細胞起源予測モデル

### Phase 3: 外部検証
- DLBCL-Patho2データセットでの再現性確認
- 他施設データでの検証

### Phase 4: 機能的検証
- Feature_740高値/低値群の組織学的比較
- Attention mapによる注目領域の病理学的同定

## 技術的メモ

### データ構造
- slide_features.h5: 'features', 'names', 'filenames', 'orders'キー
- patient_ID抽出: `name.split('_')[0]`
- 特徴量次元: 768次元（slide_feature）

### 統計手法
- 相関分析: Pearson, Spearman
- 群間比較: t-test, ANOVA, Kruskal-Wallis, Mann-Whitney U
- 多重検定補正: FDR（False Discovery Rate）

### 可視化手法
- ヒートマップ: seaborn, matplotlib
- 階層クラスタリング: scipy.cluster.hierarchy（Ward法）
- 散布図、ボックスプロット、分布図

## ファイル管理

### 出力ディレクトリ構造
```
out/
├── statistics_debug/           # 統計解析結果
├── comprehensive_heatmap/      # ヒートマップ解析結果  
└── feature_clinical_analysis/  # 注目特徴量解析結果
```

### 推奨ワークフロー
1. データ確認: `statistical-analysis`で基礎統計
2. 全体把握: `comprehensive-heatmap`で相関パターン確認
3. 詳細解析: `feature-clinical-analysis`で重要特徴量の深掘り 