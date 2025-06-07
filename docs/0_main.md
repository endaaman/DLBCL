# DLBCL 病理画像解析プロジェクト

## 研究概要

DLBCL病理画像から抽出された特徴量（GigaPath）と臨床データの関連を解析し、新たな形態病理学的知見を得る。

## データセット
- **DLBCL-Morph**: 149例、202WSI（充実した臨床データ）
- **DLBCL-Patho2**: 64例、96WSI（検証用）

## 主要発見

### 最重要特徴量
- **Feature_740**: PFS最強予測因子（r=-0.35, p=0.0015）
- **BCL6関連**: 複数特徴量と中程度相関
- **CD10関連**: 負の相関パターン

### 生存解析結果
- **年齢**: 高齢群で有意に予後不良（p<0.0001）
- **LDH**: 高値群で予後良好傾向（p=0.0016）
- **MUM1 IHC**: ABC型マーカーとして予後不良傾向（p=0.0506）

### 臨床相関
- **CD10-BCL6**: r=0.366（GCBマーカー間）
- **BCL2-MYC**: r=0.302（Double Expressors）
- **LDH-IPI**: V=0.615（統合指標妥当性）

## モジュール構成

- **`dlbcl.main`**: 画像特徴量解析、UMAP可視化、生存解析
- **`dlbcl.clinical`**: 臨床データ専用相関解析
- **`dlbcl.feature`**: 特徴量解析と統計的検証

## 基本コマンド

```bash
# クラスタリング・可視化
uv run python -m dlbcl.main cluster --dataset morph --target HANS

# 統計解析
uv run python -m dlbcl.main statistical-analysis --dataset morph

# 生存解析
uv run python -m dlbcl.main survival-analysis --dataset morph

# 臨床データ相関解析
uv run python -m dlbcl.clinical clinical-correlation --dataset morph

# 統計的検証
uv run python -m dlbcl.feature analysis-validation --dataset morph
```

## 共通パラメータ
- `--dataset [morph|patho2]`: データセット選択
- `--seed 42`: 再現性確保
- `--device cuda`: GPU使用

## 技術的成果

### 新機能実装
- 適応的二群分割アルゴリズム（免疫染色マーカー対応）
- 臨床データ専用解析モジュール
- 統計的妥当性自動検証

### 統計的検証結果
- **総検定数**: 5,376回、**実際有意**: 742個
- **FDR補正後**: 19個（大幅減少）
- **効果サイズ**: 大効果0個、中効果4個のみ
- **頑健性**: 良好（Bootstrap CI幅平均: 0.262）

## 臨床的意義

1. **予後層別化**: Feature_740による新しい予後予測
2. **個別化医療**: Leidenクラスタによる精密層別化
3. **診断支援**: 免疫染色マーカーの定量的評価

## 今後の展開

1. **外部検証**: 独立データセットでの妥当性確認
2. **多変量解析**: Cox回帰による予後因子同定
3. **臨床実装**: 病理診断支援システム開発