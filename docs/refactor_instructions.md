# DLBCL プロジェクト リファクタリング指針

## 新仕様への移行完了: dlbcl.cluster

### 主要な変更点

#### 1. 基底クラス設計の変更
- **旧**: `BaseMLCLI` → **新**: `ExperimentCLI`
- **Combat補正**: `--use-combat`フラグで制御
- **出力ディレクトリ**: `out/{raw|combat}/{dataset}/`の自動管理

#### 2. データ構造の統一
- **Datasetクラス**: dataclass化でシンプルに
- **関数分離**: `load_dataset()`, `merge_dataset()` → `utils/dataset.py`
- **基底クラス保証**: `self.output_dir`の自動作成

#### 3. 実装の大幅簡素化
- **行数削減**: 400行 → 200行未満
- **try文禁止**: エラーハンドリングの簡略化
- **機能分割**: 各メソッド50行未満の小さな実装

### 実装された機能

#### dlbcl.cluster モジュール
1. **`run_leiden`** (42行)
   - Leidenクラスタリング
   - 結果CSV保存

2. **`run_visualize`** (84行)  
   - UMAP統一可視化
   - カテゴリカル/数値自動判定
   - `--noshow`オプション対応
   - leiden_cluster読み込み対応

3. **`run_combat_comparison`** (67行)
   - Combat補正前後比較
   - 並列UMAP可視化
   - シルエット係数による定量評価

### 新仕様での使用例

```bash
# 基本的な可視化
uv run python -m dlbcl.cluster visualize --dataset patho2 --target HANS

# Leidenクラスタリング
uv run python -m dlbcl.cluster leiden --dataset morph --resolution 0.5

# Combat補正比較
uv run python -m dlbcl.cluster combat-comparison --dataset merged --use-combat

# バッチ処理（noshow）
uv run python -m dlbcl.cluster visualize --dataset morph --noshow --target "BCL6 FISH"
```

### 他モジュールへの適用指針

#### 必須変更項目
1. **基底クラス**: `ExperimentCLI`を継承
2. **引数構造**: `class XxxArgs(ExperimentCLI.CommonArgs)`
3. **データアクセス**: `self.dataset.features`, `self.dataset.merged_data`
4. **出力管理**: `self.output_dir`を使用（makedirs不要）
5. **Combat対応**: 補正前データが必要な場合は`load_dataset()`で別途読み込み

#### 推奨パターン
- 各機能50行未満
- `--noshow`オプション追加
- エラーメッセージの簡潔化
- インポートの最小化

### パフォーマンス改善
- 無駄な再読み込み削除
- 基底クラスのデータ活用
- サイズ不整合の自動修正

### 成果
- **開発効率**: 大幅なコード削減
- **保守性**: シンプルで理解しやすい構造  
- **機能性**: 全ての既存機能を維持
- **拡張性**: 新機能追加が容易
