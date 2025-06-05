# DLBCL analysis

## Intro
SSLで訓練された病理組織基盤モデル（Prov-GigaPath, UNIなど）によって病理画像から良質な埋め込み情報を得ることができるようになった。これらは様々なデータセットを通じて下流のタスクに適用された結果、既存のアプローチを大きく超える性能を示した。


DLBCLでは、腫瘍は症例ごとに様々な形態を呈し、形態的にはすでにいくつかの亜型分類は知られているが、それらは予後や免疫組織学的特徴との関係性は十分に明らかになっていない。

本研究では、基盤モデルから得られる埋め込みと、それに予後や免疫形質の関連を探索することで、新たな形態病理学知見を得ることを目指す。

## Methods

### Dataset

方法２つのデータセットを用意した。
- DLBCL-Morph: 公開データセット。149例、203WSIある。
  関連データのカラムは column_description.csv を参照。
  臨床データは clinical_data_cleaned.csv
- DLBCL-Patho2: in-houseデータセット。63例、96WSIある。
  関連データは同様のカラムのものを準備中。


### Preprocessing by Foundation model

各WSIデータはパッチ分割して


### Analysis





