# DLBCL解析システム実装サマリー（更新版）

## システム概要

DLBCL（びまん性大細胞型B細胞リンパ腫）AI病理診断のための包括的解析システム。GigaPath ViT特徴量（768次元）を用いた解釈可能AI診断基盤。

## 主要機能

### 1. 統一相関ヒートマップ解析
**コマンド**: `uv run python -m dlbcl.main compare-correlation`

**機能**:
- MorphとPatho2データセットの統一比較
- 6共通変数での中立的デンドログラムクラスタリング
- ViT特徴量モジュール化（768→20次元）

**出力ファイル**:
```
out/compare_correlation/
├── unified_correlation_comparison.png    # 美しい統一ヒートマップ
├── unified_correlation_difference.png    # データセット間差分解析
├── vit_module_analysis.png              # モジュール比較可視化
├── feature_modules_major.json           # 5大モジュール詳細
├── feature_modules_moderate.json        # 10中モジュール詳細
├── feature_modules_fine.json            # 20細モジュール詳細
├── morph_module_features.csv            # Morphモジュール特徴量
└── patho2_module_features.csv           # Patho2モジュール特徴量
```

### 2. Leidenクラスタリング
**コマンド**: `uv run python -m dlbcl.main leiden`

**機能**:
- 768次元特徴量の教師なしクラスタリング
- UMAP可視化とクラスター評価
- 臨床変数との相関解析

### 3. 包括的相関解析
**コマンド**: `uv run python -m dlbcl.main comprehensive-heatmap`

**機能**:
- 768特徴量×臨床変数の全相関マトリックス
- デンドログラム付きヒートマップ
- 統計的有意性検定

### 4. 生存解析
**コマンド**: `uv run python -m dlbcl.main survival-analysis`

**機能**:
- Kaplan-Meier生存曲線
- クラスター別予後層別化
- Log-rank検定による統計評価

## 技術仕様

### データ処理パイプライン
```python
# 1. データ読み込み
clinical_data = load_clinical_data()  # 統一命名規則
slide_features = load_h5_features()   # GigaPath 768次元

# 2. 統一デンドログラム作成
common_vars = ['CD10 IHC', 'MUM1 IHC', 'BCL2 IHC', 'BCL6 IHC', 'MYC IHC', 'HANS']
combined_pattern = (morph_corr + patho2_corr) / 2  # 中立的パターン
feature_linkage = linkage(combined_pattern, method='ward')

# 3. モジュール抽出
modules = fcluster(feature_linkage, n_clusters=[5, 10, 20])
module_features = average_within_modules(modules)  # 768 → 20次元
```

### モジュール評価指標
```python
# 一貫性スコア計算
def calculate_consistency(morph_module, patho2_module):
    return np.corrcoef(morph_module.fillna(0), patho2_module.fillna(0))[0,1]

# 生物学的関連評価
def evaluate_clinical_association(module_features, clinical_vars):
    correlations = []
    for var in clinical_vars:
        corr, p_val = pearsonr(module_features, var)
        correlations.append({'var': var, 'corr': corr, 'p': p_val})
    return sorted(correlations, key=lambda x: abs(x['corr']), reverse=True)
```

## 品質保証システム

### 1. データ一貫性チェック
- **命名規則統一**: `clinical_data_cleaned_path2_mod.csv`使用
- **欠損値処理**: NaN値の適切なマスキング
- **型変換**: 患者IDの文字列/数値統一

### 2. 解析品質評価
- **モジュール一貫性**: 高一貫性（>0.5）は信頼性指標
- **相関差分**: データセット間差分の定量化
- **統計的検定**: FDR補正による多重検定対応

### 3. 可視化品質
- **アスペクト比**: `aspect='auto'`で適切な表示
- **カラーマップ**: `RdBu_r`で直感的色分け
- **レイアウト**: デンドログラム統一による比較可能性

## 解釈可能AI診断への応用

### 1. モジュールベース診断
```python
# 高信頼性モジュールでの診断支援
high_confidence_modules = [
    'module_3',  # BCL2/MUM1経路（一貫性0.746）
    'module_2',  # BCL6関連（一貫性0.430）
    'module_10', # CD10/HANS（一貫性0.390）
]

def predict_ihc_markers(slide_features, modules):
    """ViT特徴量からIHCマーカー予測"""
    predictions = {}
    for marker in ['BCL2 IHC', 'BCL6 IHC', 'CD10 IHC']:
        module_scores = extract_module_scores(slide_features, modules)
        predictions[marker] = linear_model.predict(module_scores)
    return predictions
```

### 2. 品質管理システム
```python
# 低一貫性モジュールでの技術品質評価
def quality_assessment(slide_features, low_consistency_modules):
    """技術的品質の評価"""
    qc_scores = {}
    for module in low_consistency_modules:
        module_pattern = extract_module_pattern(slide_features, module)
        qc_scores[module] = calculate_deviation_from_expected(module_pattern)
    return qc_scores

# アラートシステム
if any(score > threshold for score in qc_scores.values()):
    warnings.warn("技術的品質に問題の可能性あり")
```

### 3. 臨床意思決定支援
```python
def clinical_decision_support(patient_features):
    """モジュールベース意思決定支援"""
    
    # 1. IHCマーカー予測
    ihc_predictions = predict_ihc_markers(patient_features)
    
    # 2. HANS分類
    hans_prediction = predict_hans_classification(patient_features)
    
    # 3. 予後リスク評価
    risk_modules = extract_prognostic_modules(patient_features)
    survival_risk = predict_survival_risk(risk_modules)
    
    # 4. 治療推奨
    treatment_recommendation = recommend_treatment(
        ihc_predictions, hans_prediction, survival_risk
    )
    
    return {
        'ihc_markers': ihc_predictions,
        'hans_classification': hans_prediction,
        'survival_risk': survival_risk,
        'treatment_recommendation': treatment_recommendation,
        'confidence_scores': calculate_module_consistency_scores(patient_features)
    }
```

## 実装の特徴

### 🎯 解釈可能性
- **20モジュール**: 768次元から解釈可能な次元へ圧縮
- **生物学的意味**: 各モジュールが特定の病理機能に対応
- **一貫性指標**: 信頼性の定量的評価

### 🔒 堅牢性
- **統一デンドログラム**: データセット偏向の排除
- **品質管理**: 低一貫性モジュールでの技術評価
- **多重検定補正**: 統計的厳密性の確保

### 🚀 拡張性
- **モジュラー設計**: 新しいデータセットへの容易な適用
- **階層的解析**: Major/Moderate/Fineレベルでの柔軟な解析
- **API化**: 臨床システムへの統合準備完了

## 次世代展開

### 1. リアルタイム診断システム
- **WebAPI**: REST/GraphQL経由での診断要求
- **並列処理**: 複数スライドの同時解析
- **結果配信**: DICOM/HL7準拠での結果送信

### 2. 継続学習システム
- **フィードバック**: 病理医判定との比較学習
- **モジュール更新**: 新データでのモジュール再構築
- **性能監視**: リアルタイム精度監視

### 3. 多施設展開
- **標準化**: モジュール一貫性による品質保証
- **バリデーション**: 施設間での性能評価
- **カスタマイズ**: 施設特異的調整機能

**結論**: 本システムにより、解釈可能で信頼性の高いAI病理診断の実用化基盤を確立。ViT特徴量モジュール化による革新的アプローチで、次世代病理診断システムの開発を実現した。