# DLBCL解析結果

## 主要結果

### 統計解析
- **Feature_740**: PFS最強予測因子（r=-0.35, p=0.0015）
- **有意相関**: 32個（Morph）、17個群間差
- **BCL6-特徴量**: 複数中程度相関確認

### クラスタリング
**HDBSCAN結果**:
- Morph: 2クラスター、シルエット0.178
- Patho2: 2クラスター、シルエット-0.021  
- 統合: データセット間明確分離

### 生存解析（Morph）
**有意な予後因子**:
- **年齢**: OS p<0.0001、高齢群で予後不良
- **LDH**: OS p=0.0016、PFS p=0.0008
- **MUM1 IHC**: OS p=0.0506（境界域）
- HANS分類: 非有意

### 免疫染色マーカー生存解析
**適応的分割アルゴリズム導入**:
- CD10 IHC: Negative(113) vs Positive(69)
- MUM1 IHC: Negative(86) vs Positive(98) 
- LDH: Normal(126) vs High(72)

**結果**:
- **LDH**: 最強予後因子（OS/PFS両方でp<0.001）
- **MUM1**: ABC型マーカーとして予後不良傾向

### 臨床データ相関解析
**有意な関連**:
- **CD10-BCL6**: r=0.366（GCBマーカー間）
- **BCL2-MYC**: r=0.302（Double Expressors）
- **CD10-MUM1**: r=-0.227（GCB vs ABC対立）
- **LDH-IPI**: V=0.615（統合指標妥当性）

## 技術実装

### 新機能
- `dlbcl.clinical`: 臨床データ専用解析モジュール
- 適応的二群分割: データ分布に応じた最適分割
- 品質管理: 最小サンプルサイズ自動検証

### 出力構造
```
out/
├── morph/survival/     # 生存曲線15+個
├── patho2/cluster/     # UMAP可視化
├── clinical_correlation/ # 臨床相関結果
└── statistics/         # 既存統計解析
```