# DLBCL プロジェクト アーキテクチャ

## 概要

本プロジェクトは、DLBCL（びまん性大細胞型B細胞リンパ腫）の病理画像特徴量解析のためのPython CLIツールです。統計的解析の妥当性を重視し、モジュール化された構成で保守性と拡張性を確保しています。

## モジュール構成

### 1. `dlbcl.main` - 基本解析モジュール
**目的**: 基本的な探索的データ解析とクラスタリング

**提供機能**:
- `gather-slide-features`: スライド特徴量の収集・統合
- `cluster`: HDBSCAN/UMAPによるクラスタリング・可視化
- `umap`: 次元削減による可視化
- `statistical-analysis`: 基本的な統計解析（相関・群間比較）
- `comprehensive-heatmap`: 包括的相関ヒートマップ

**主要クラス**:
```python
class CLI(BaseMLCLI):
    class UMAPArgs(CommonArgs): ...
    class ClusterArgs(CommonArgs): ...
    class StatisticalAnalysisArgs(CommonArgs): ...
    class ComprehensiveHeatmapArgs(CommonArgs): ...
```

### 2. `dlbcl.feature` - 統計的検証モジュール
**目的**: 高度な統計的妥当性検証

**提供機能**:
- `analysis-validation`: 包括的統計的妥当性評価
  - 多重検定問題の評価
  - 効果サイズの詳細分析
  - 順列検定による偶然性検証
  - Bootstrap法による頑健性評価

**主要クラス**:
```python
class CLI(BaseMLCLI):
    class AnalysisValidationArgs(CommonArgs):
        output_dir: str = param('out/analysis_validation')
        n_permutations: int = param(1000)
        effect_size_threshold: float = param(0.3)
        fdr_alpha: float = param(0.05)
```

### 3. `dlbcl.utils` - 共通基盤モジュール
**目的**: 共通処理とCLI基盤の提供

**構成要素**:
- `cli.py`: BaseMLCLI基盤クラス
- `data_loader.py`: 共通データ読み込み処理
- `seed.py`: 再現性確保のためのシード管理
- `__init__.py`: ユーティリティ関数

## 共通基盤アーキテクチャ

### BaseMLCLI クラス
すべてのCLIモジュールの基盤となるクラス：

```python
class BaseMLArgs(BaseModel):
    seed: int = get_global_seed()
    dataset: str = param('morph', choices=['morph', 'patho2'])

class BaseMLCLI(AutoCLI):
    class CommonArgs(BaseMLArgs):
        device: str = 'cuda'

    def _pre_common(self, a: BaseMLArgs):
        fix_global_seed(a.seed)
        
        # データセットディレクトリ設定
        if a.dataset == 'morph':
            self.dataset_dir = Path('./data/DLBCL-Morph/')
        elif a.dataset == 'patho2':
            self.dataset_dir = Path('./data/DLBCL-Patho2/')
        
        # 共通データ読み込み
        data_dict = load_common_data(a.dataset)
        if data_dict:
            self.clinical_data = data_dict['clinical_data']
            self.features = data_dict['features']
            # ... その他の共通属性
```

### 共通データローダー
`utils/data_loader.py`で提供される統一データ読み込み：

```python
def load_common_data(dataset: str):
    """共通データの読み込み"""
    # 臨床データ読み込み
    if dataset == 'patho2':
        clinical_data = pd.read_csv(f'data/DLBCL-{dataset.capitalize()}/clinical_data_extracted_from_findings.csv')
    else:
        clinical_data = pd.read_csv(f'data/DLBCL-{dataset.capitalize()}/clinical_data_cleaned.csv')
    
    # 特徴量データ読み込み
    with h5py.File(f'data/DLBCL-{dataset.capitalize()}/slide_features.h5', 'r') as f:
        features = f['features'][:]
        feature_names = f['names'][:]
    
    # データマージと前処理
    # ...
    
    return {
        'clinical_data': clinical_data,
        'features': features,
        'merged_data': merged_data,
        # ...
    }
```

## データフロー

```
Raw Data (WSI + Clinical)
    ↓ (gather-slide-features)
Feature Extraction (slide_features.h5)
    ↓ (load_common_data)
Merged Dataset (clinical + features)
    ↓
┌─────────────────┬─────────────────┐
│   dlbcl.main    │  dlbcl.feature  │
│                 │                 │
│ • clustering    │ • validation    │
│ • umap         │ • effect size   │
│ • basic stats  │ • permutation   │
│ • heatmap      │ • robustness    │
└─────────────────┴─────────────────┘
    ↓
Results & Visualizations
```

## 設計原則

### 1. 関心の分離
- **dlbcl.main**: 探索的データ解析
- **dlbcl.feature**: 統計的妥当性検証
- **dlbcl.utils**: 共通基盤処理

### 2. DRY (Don't Repeat Yourself)
- 共通データ読み込み処理の統一
- BaseMLCLIによる共通パラメータ管理
- 統一されたCLIインターフェース

### 3. 拡張性
- 新しい解析モジュールの追加が容易
- 共通基盤の恩恵を自動的に受ける
- 統一されたパラメータとデータ管理

### 4. 再現性
- 全モジュールで統一されたシード管理
- 決定論的な処理順序
- バージョン管理された依存関係

## 利用方法

### 基本的な解析ワークフロー
```bash
# 1. 基本解析
uv run python -m dlbcl.main cluster --dataset morph --target HANS
uv run python -m dlbcl.main statistical-analysis --dataset morph

# 2. 統計的検証
uv run python -m dlbcl.feature analysis-validation --dataset morph --n-permutations 1000

# 3. 詳細可視化
uv run python -m dlbcl.main comprehensive-heatmap --dataset morph
```

### 共通パラメータ
全てのコマンドで利用可能：
- `--dataset [morph|patho2]`: データセット選択
- `--seed 42`: 再現性のためのシード値
- `--device cuda`: 計算デバイス指定

## 出力構造

```
out/
├── [dataset]/                          # データセット別基本解析結果
│   ├── umap_*.png                     # UMAP可視化
│   └── clustering_results.csv        # クラスタリング結果
├── statistics/                        # 基本統計解析
├── comprehensive_heatmap/              # 相関ヒートマップ
└── analysis_validation/                # 統計的検証結果
    ├── comprehensive_validation_report.png
    ├── multiple_testing_evaluation.csv
    ├── effect_size_evaluation.csv
    ├── permutation_test_evaluation.csv
    ├── robustness_evaluation.csv
    └── validation_summary_stats.csv
```

## 今後の拡張ポイント

### 新モジュール追加の容易さ
```python
# 新しい解析モジュール例
class NewAnalysisCLI(BaseMLCLI):
    class NewAnalysisArgs(CommonArgs):
        # 独自パラメータのみ定義
        specific_param: float = 0.5
    
    def run_new_analysis(self, a: NewAnalysisArgs):
        # self.merged_data などが自動的に利用可能
        # 共通データローダーの恩恵を受ける
        pass
```

### 統計的妥当性の継続的向上
- 新しい検証手法の追加
- より厳密な効果サイズ評価
- 独立検証データセットでの確認

この設計により、コードの保守性、拡張性、再現性を同時に確保し、統計的に妥当な解析結果の提供を可能にしています。