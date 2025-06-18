# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## プロジェクト概要

GigaPath基盤モデルを使用してDLBCL（びまん性大細胞型B細胞リンパ腫）を解析する医療・病理深層学習プロジェクトです。全スライド画像（WSI）から特徴量を抽出し、画像特徴量と臨床データの相関について様々な統計・機械学習解析を実行します。

## 重要な開発ルール

### Python実行 - 絶対厳守
- **Pythonコマンドを直接実行してはならない**
- **必ず`uv run`を使用してPythonスクリプト/コマンドを実行する**
- 例:
  - ✅ `uv run python script.py`
  - ✅ `uv run python -m dlbcl.clinical clinical-correlation`
  - ❌ `python script.py` 
  - ❌ `python -m dlbcl.clinical clinical-correlation`

### 言語・コミュニケーション
- 全てのコミュニケーション、コメント、docstringは日本語で記述
- 変数名・関数名は英語でも可

## コマンド

### 環境セットアップ
```bash
# Pythonバージョン固定（3.11が必要）
uv python pin 3.11

# 依存関係のインストール
uv pip install -e .

# 開発用依存関係のインストール
uv pip install -e ".[dev]"
```

### 解析コマンドの実行

各解析モジュールは独立して実行されます：

```bash
# 臨床相関解析
uv run python -m dlbcl.clinical clinical-correlation --dataset morph

# クラスタリングと可視化
uv run python -m dlbcl.cluster visualize --dataset morph --target HANS

# 特徴量解析と検証
uv run python -m dlbcl.feature analysis-validation --dataset morph

# 生存解析
uv run python -m dlbcl.survival survival-analysis --dataset morph

# 相関解析
uv run python -m dlbcl.correlation compare-correlation --dataset morph

# 回帰解析（ROC、Lasso）
uv run python -m dlbcl.regression roc-analysis --dataset morph

# デンドログラムからのモジュール解析
uv run python -m dlbcl.module module-analysis --dataset morph

# 染色マーカーの重み解析
uv run python -m dlbcl.weight_analysis weight-analysis --dataset morph
```

### 共通パラメータ
- `--dataset [morph|patho2]`: データセット選択
- `--seed`: 再現性のための乱数シード
- `--use-combat`: Combatバッチ補正を適用
- `--device [cuda|cpu]`: デバイス選択
- `--help`: 任意のコマンドのヘルプを表示

### テスト
```bash
# 臨床変数テストの実行
uv run python test_clinical_vars.py
```

## アーキテクチャ

### CLIフレームワーク（pydantic-autocli）
- 基底クラス: `dlbcl/utils/cli.py`の`BaseMLCLI`
- 各解析タイプは`BaseMLCLI`を継承した独自のCLIモジュールを持つ
- `run_*`メソッドがCLIコマンドになる（`run_`プレフィックスは実行時不要）
- 引数は`param()`関数を使用したPydanticモデルで定義

### AutoCLIの使用法
- `def run_foo_bar(self, args):` → `python script.py foo-bar`として実行
- `def prepare(self, args):` → 共有初期化処理
- `class FooBarArgs(AutoCLI.CommonArgs):` → コマンド引数の定義
- 戻り値: `True`/`None`（成功）、`False`（失敗）、`int`（終了コード）

### データ構造
- **data/DLBCL-Morph/**: 149症例、SVS形式スライド
- **data/DLBCL-Patho2/**: 64症例、NDPI形式スライド
- HDF5ファイルに保存された特徴量:
  - `gigapath/slide_feature`: 768次元のスライドレベル特徴量
  - `gigapath/features`: N×1536次元のパッチレベル特徴量
- 生存情報を含むCSV形式の臨床データ

### 出力構造
- 結果は`out/{dataset}/{analysis_type}/`に保存
- Combat補正済み結果は`out/combat/`サブディレクトリに保存

## 開発ガイドライン

### 実験的アプローチ
- 小さなステップから始めて慎重に検証
- シード管理による再現性の確保
- 多重検定補正を含む適切な統計手法の適用
- 統計的有意性と生物学的意義の区別
- 堅牢性を検証するための複数手法の比較
- 既存手法との比較を含む段階的な新手法の実装

### コードスタイル
- 全ての関数パラメータに型ヒントが必要
- 関数とクラスの日本語docstring
- コードベース内の既存パターンに従う
- 既存のライブラリを使用（新規追加前にインポートを確認）

### 重要な注意事項
- 特徴量抽出にGigaPath基盤モデルを使用
- バッチ効果の処理にCombatバッチ補正が利用可能
- 全ての解析はmorphとpatho2の両データセットをサポート
- 結果にはFDR補正を含む包括的な統計解析が含まれる