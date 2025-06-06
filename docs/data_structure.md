# DLBCL データ構造詳細

## データセット概要

### DLBCL-Morph
- **症例数**: 149例
- **WSI数**: 203枚
- **データパス**: `data/DLBCL-Morph/`
- **臨床データ**: `clinical_data_cleaned.csv`（充実した臨床情報）

### DLBCL-Patho2  
- **症例数**: 63例
- **WSI数**: 96枚
- **データパス**: `data/DLBCL-Patho2/`
- **臨床データ**: `clinical_data_cleaned_path2.csv`（限定的な臨床情報）

## H5ファイル構造

各WSIは以下の構造でH5ファイルに保存されている：

```
WSI_ID.h5
├── patches/                     # パッチ画像データ [N_patches, 256, 256, 3]
├── coordinates/                 # パッチ座標 [N_patches, 2]
├── metadata/                    # メタデータ
│   ├── original_mpp
│   ├── original_width
│   ├── original_height
│   ├── image_level
│   ├── mpp
│   ├── scale
│   ├── patch_size
│   ├── patch_count
│   ├── cols
│   └── rows
└── gigapath/                    # GigaPath特徴量
    ├── features                 # パッチレベル特徴量 [N_patches, 1536]
    ├── slide_feature           # スライドレベル特徴量 [768]
    └── clusters                # パッチクラスタ番号 [N_patches]
```

### 実例
- **DLBCL-Morph**: `26787_0.h5` → 11,506パッチ
- **DLBCL-Patho2**: `20-0494_1_1.h5` → 11,665パッチ

## 臨床データ項目

### 免疫組織化学染色
- `MYC IHC`: c-Myc染色陽性率 (%)
- `BCL2 IHC`: BCL2染色陽性率 (%)
- `BCL6 IHC`: BCL6染色陽性率 (%)
- `CD10 IHC`: CD10染色陽性率 (%)
- `MUM1 IHC`: MUM1染色陽性率 (%)

### FISH解析
- `BCL2 FISH`: BCL2 break-apart FISH
- `BCL6 FISH`: BCL6 break-apart FISH  
- `MYC FISH`: MYC break-apart FISH

### 分類・予後因子
- `HANS`: Hans algorithmによる細胞起源分類
- `Age`: 治療開始時年齢
- `ECOG PS`: ECOG Performance Status
- `LDH`: 乳酸脱水素酵素高値 (0/1)
- `EN`: 節外病変数
- `Stage`: Modified Ann Arbor staging
- `IPI Score`: International Prognostic Index score
- `IPI Risk Group`: IPIリスク群 (4分類)
- `RIPI Risk Group`: Revised IPI リスク群

### 予後情報（DLBCL-Morphのみ）
- `OS`: Overall Survival (年)
- `PFS`: Progression Free Survival (年)
- `Follow-up Status`: 最終フォローアップ時の生存状況 (0: 生存, 1: 死亡)

## 既知の問題
1. **slide_feature と臨床データの関連が弱い**
   - UMAPによる可視化では直接的な関連性が見られない
   
2. **slide_feature の代表性に疑問**
   - パッチレベル特徴量の単純集約では不十分な可能性
   
3. **データの欠損値**
   - 臨床データに多くの欠損値が存在
   - DLBCL-Patho2では予後情報が未整備

## 分析戦略
- Phase 1: 統計的分析の強化
- Phase 2: パッチレベル特徴量の活用 
- Phase 3: 新しい集約手法の検討 