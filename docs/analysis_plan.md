# DLBCL研究 解析方針・実験設計

## 研究背景・現状
- 基盤モデル（Prov-GigaPath）から得られた768次元スライドレベル特徴量を活用
- DLBCLの形態学的特徴と臨床データ・予後の関連を探索
- 既存のUMAP可視化では直接的な結果が得られていない
- 基礎的なところから系統的にアプローチする

## 解析戦略：段階的アプローチ

### Phase 1: 基礎データ理解
**目的**: データの品質確認と基本的な性質の把握

#### 1.1 臨床データ探索 (`clinical_stats`)
- **狙い**: データセットの特徴と制約を理解
- **内容**: 記述統計、分布確認、欠損値パターン、変数間の基本的関係
- **判断点**: データ品質、サンプルサイズの妥当性、解析可能な変数の特定

#### 1.2 特徴量品質評価 (`feature_quality`) 
- **狙い**: 768次元特徴量の性質と問題点の把握
- **内容**: 基本統計、相関構造、次元数の妥当性、外れ値の存在
- **判断点**: 特徴量の冗長性、適切な前処理方法、次元削減の必要性

### Phase 2: 単変量関係探索
**目的**: 特徴量と臨床変数の個別関係の発見

#### 2.1 特徴量-臨床変数相関 (`correlation_analysis`)
- **狙い**: 有意な関連のある特徴量の特定
- **内容**: 連続変数との相関、カテゴリカル変数との関連、多重検定補正
- **判断点**: 信頼できる関連の特定、後続解析の方向性決定

#### 2.2 予後との関連 (`survival_analysis`)
- **狙い**: 予後予測に有用な特徴量の発見
- **内容**: 生存曲線、Log-rank検定、単変量Cox回帰
- **判断点**: 予後予測の可能性、重要な特徴量の特定

### Phase 3: 多変量モデリング
**目的**: 実用的な予測・分類モデルの構築

#### 3.1 予後予測モデル (`prognosis_model`)
- **狙い**: 基盤モデル特徴量の臨床的有用性の検証
- **内容**: 特徴選択、Cox回帰、性能評価、既存指標との比較
- **判断点**: 既存のIPI等を超える予測性能の達成

#### 3.2 分類モデル (`classification_model`)
- **狙い**: 既存分類（HANS等）の予測と新規分類の可能性
- **内容**: 各種分類タスク、モデル比較、解釈可能性分析
- **判断点**: 実用レベルの分類性能、新規知見の発見

### Phase 4: パターン発見・探索的解析
**目的**: 新たな形態学的サブタイプや関係性の発見

#### 4.1 教師なしクラスタリング (`clustering_analysis`)
- **狙い**: データ駆動による新規サブタイプの発見
- **内容**: 複数手法比較、最適化、クラスタ特徴づけ
- **判断点**: 臨床的に意味のあるクラスタの存在

#### 4.2 次元削減最適化 (`dimension_reduction`)
- **狙い**: UMAPの改良と可視化による洞察獲得
- **内容**: パラメータ最適化、複数手法比較、解釈
- **判断点**: 明確な構造の可視化、臨床変数との対応

## 実験設計の考え方

### AutoCLI実験フレームワーク
各解析を独立した実験として設計：
```bash
# 基礎探索
uv run python -m dlbcl.main clinical_stats [--options]
uv run python -m dlbcl.main feature_quality [--options]

# 関係探索  
uv run python -m dlbcl.main correlation_analysis [--options]
uv run python -m dlbcl.main survival_analysis [--options]

# モデリング
uv run python -m dlbcl.main prognosis_model [--options]
uv run python -m dlbcl.main classification_model [--options]

# パターン発見
uv run python -m dlbcl.main clustering_analysis [--options]
uv run python -m dlbcl.main dimension_reduction [--options]
```

### 判断基準・閾値設定
- **統計的有意性**: p < 0.05 (多重検定補正後)
- **効果量**: Cohen's d > 0.5 (中程度以上)
- **予測性能**: C-index > 0.7, AUC > 0.8
- **クラスタ品質**: Silhouette score > 0.5

### データの取り扱い
- **欠損値**: 20%以上欠損の変数は除外を検討
- **外れ値**: Z-score > 3 で要検討
- **標準化**: 手法に応じて適切に選択
- **交差検証**: 5-fold CV を基本とする

### アウトプット設計
各実験で生成すべき成果物：
- **数値結果**: CSV/Excel形式での統計結果
- **可視化**: 高解像度の図表（論文品質）
- **解釈**: テキストレポートでの要約と解釈
- **再現性**: パラメータと結果の完全な記録

## 研究成果への道筋

### 段階的な意思決定プロセス
1. **Phase 1完了** → データ品質とフィージビリティの確認
2. **Phase 2完了** → 有望な方向性の特定
3. **Phase 3完了** → 実用性の検証
4. **Phase 4完了** → 新規知見の発見

### 論文化戦略
- **Table 1**: Patient characteristics (Phase 1)
- **Table 2**: Feature-clinical associations (Phase 2)  
- **Table 3**: Prognostic model performance (Phase 3)
- **Table 4**: Classification results (Phase 3)
- **Figure 1**: Study workflow and data overview
- **Figure 2**: Feature analysis and correlations
- **Figure 3**: Survival analysis results
- **Figure 4**: UMAP visualization with annotations
- **Figure 5**: Clustering and novel subtype characterization

### 失敗時の代替戦略
- **予後予測が困難** → 短期的エンドポイントに変更
- **分類性能不足** → 特徴量エンジニアリングや他手法の検討
- **クラスタが不明確** → パッチレベル解析への展開
- **全体的に関連薄い** → 他の基盤モデル（UNI等）との比較

## 実装時の注意点

### データファーストアプローチ
1. **まず現実のデータを確認**してから具体的実装
2. **データの性質に応じて**柔軟に解析計画を調整
3. **段階的に進めて**各ステップの結果を次に活かす

### 品質管理
- 各実験の結果を必ず検証・解釈してから次へ進む
- 統計的仮定の確認（正規性、等分散性等）
- 生物学的・臨床的解釈可能性の常時考慮

### 効率的な探索
- パラメータ設定は粗い範囲から始めて段階的に精密化
- 計算コストの高い解析は小さなサブセットで予備検証
- 有望でない方向は早期に見切りをつける 