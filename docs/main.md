# DLBCL analysis

## Intro
SSLで訓練された病理組織基盤モデル（Prov-GigaPath, UNIなど）によって病理画像から良質な埋め込み情報を得ることができるようになった。これらは様々なデータセットを通じて下流のタスクに適用された結果、既存のアプローチを大きく超える性能を示した。


DLBCLでは、腫瘍は症例ごとに様々な形態を呈し、形態的にはすでにいくつかの亜型分類は知られているが、それらは予後や免疫組織学的特徴との関係性は十分に明らかになっていない。

本研究では、基盤モデルから得られる埋め込みと、それに予後や免疫形質の関連を探索することで、新たな形態病理学知見を得ることを目指す。

## Methods

### Dataset

方法２つのデータセットを用意した。
- DLBCL-Morph: 公開データセット。149例、203WSIある。
  臨床データのカラムは column_description.csv を参照。
  臨床データは clinical_data_cleaned.csv
- DLBCL-Patho2: in-houseデータセット。63例、96WSIある。
  関連データは同様のカラムのものを準備中。


### Preprocessing by Foundation model

各WSIデータは256x256pcのパッチ分割して、それぞれについてのProv-GigaPathの埋め込み（1536次元）を得た。
これらを hdf5にまとめて保存している。

```
'patches'                 : パッチのデータ dim:[<patch_count>, <patch_size[0]>, <patch_size[1]>, 3]
                            ex: [3237, 256, 256, 3] のようなテンソル
'coordinates'             : 各パッチのピクセル単位の座標 dim:[<patch_count>, 2]

'metadata/original_mpp'   : もともとのmpp
'metadata/original_width' : もともとの画像の幅（level=0）
'metadata/original_height': もともとの画像の幅（level=0）
'metadata/image_level'    : 使ったレベル（基本的にはlevel=0になる）
'metadata/mpp'            : 出力されたパッチのmpp
'metadata/scale'          : 出力時のscale
'metadata/patch_size'     : パッチの解像度
'metadata/patch_count'    : パッチの総数
'metadata/cols'           : パッチを並べたときの横方向の数
'metadata/rows'           : パッチを並べたときの縦方向の数

'gigapath/features'       : GigaPathで抽出した特等量 dim:[<patch_count>, 1536]
'gigapath/slide_feature'  : Slide level encoderで抽出した特徴量 dim: [768]
'gigapath/clusters'       : 上記からPCA+leidenで取得したクラスタ番号 dim: [<patch_count>]
```


### Analysis

'gigapath/slide_feature' の 768次元の埋め込みを使って各臨床データとの関連を検討を行う。


### 現状

MorhpデータセットについてUMAPのscatterを出して、各免疫染色や、臨床情報との関連を探したがあまり直接的な結果は得られなかった。

