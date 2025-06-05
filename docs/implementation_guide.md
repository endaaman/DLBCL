# 実装ガイド

## 基本方針
- **データファースト**: まず実際のデータを確認してから実装
- **段階的実行**: Phase 1 → Phase 2 → Phase 3 → Phase 4 の順で進行
- **柔軟な調整**: データの性質に応じて計画を調整

## AutoCLI実験フレームワーク

### 既存の実装パターン
```python
class ExperimentArgs(CommonArgs):
    output_dir: str = param('results/phase1', help='Output directory')
    # その他のパラメータ

def run_experiment(self, a: ExperimentArgs):
    """実験の説明"""
    # 実装内容
    pass
```

### 実行方法
```bash
uv run python -m dlbcl.main experiment_name [--options]
```

## Phase 1実装の基本構造

### データ読み込みパターン
```python
# 臨床データ
if a.dataset == 'morph':
    clinical_file = './data/DLBCL-Morph/clinical_data_cleaned.xlsx'
else:
    clinical_file = './data/DLBCL-Patho2/clinical_data_cleaned.xlsx'
df_clinical = pd.read_excel(clinical_file, index_col=0)

# 特徴量データ  
slide_features_file = str(self.dataset_dir / 'slide_features.h5')
with h5py.File(slide_features_file, 'r') as f:
    features = f['features'][:]
    names = f['names'][:]
```

### 結果保存パターン
```python
# ディレクトリ作成
os.makedirs(a.output_dir, exist_ok=True)

# CSV保存
results_df.to_csv(f'{a.output_dir}/results.csv')

# 図の保存
plt.savefig(f'{a.output_dir}/plot.png', dpi=300, bbox_inches='tight')

# レポート保存
with open(f'{a.output_dir}/summary.txt', 'w') as f:
    f.write('Results summary...')
```

## 実装時のチェックリスト

### Phase 1: 基礎データ理解
- [ ] データ形状・型の確認
- [ ] 欠損値パターンの把握
- [ ] 基本統計量の算出
- [ ] 可視化による直感的理解

### Phase 2: 関係探索
- [ ] 適切な統計手法の選択
- [ ] 多重検定補正の実装
- [ ] 効果量の計算
- [ ] 結果の解釈・意味づけ

### Phase 3: モデリング
- [ ] 交差検証の実装
- [ ] 性能指標の計算
- [ ] ベースラインとの比較
- [ ] モデルの解釈可能性

### Phase 4: パターン発見
- [ ] パラメータ最適化
- [ ] 結果の安定性確認
- [ ] 生物学的妥当性の検証

## トラブルシューティング

### よくある問題
- **データパスエラー**: `self.dataset_dir`の設定確認
- **メモリ不足**: バッチ処理やサンプリングを検討
- **収束しない**: パラメータ範囲やアルゴリズム変更
- **結果が不安定**: 乱数シード固定、交差検証の実装

### デバッグのコツ
- 小さなサブセットで先行検証
- 中間結果の可視化・確認
- エラーメッセージの詳細確認
- 統計的仮定の検証

## 次回実装時の手順

1. **環境確認**
   - データファイルの存在確認
   - 必要ライブラリのインストール

2. **Phase 1から開始**
   - `clinical_stats`実験の実装
   - 結果の確認・解釈

3. **段階的進行**
   - 各Phase完了後に結果を評価
   - 次のPhaseの実装要否を判断

4. **進捗記録**
   - `docs/progress.md`の更新
   - 結果の記録・解釈の蓄積 