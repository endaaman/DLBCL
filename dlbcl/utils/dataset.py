import os
import re
import numpy as np
import pandas as pd
import h5py
from dataclasses import dataclass



@dataclass
class Dataset:
    name: str
    features: np.ndarray
    feature_names: np.ndarray
    clinical_data: pd.DataFrame
    target_cols: list
    feature_df: pd.DataFrame = None
    merged_data: pd.DataFrame = None

    def __post_init__(self):
        """データフレーム作成とマージ処理"""
        # 特徴量データフレーム作成
        self.feature_df = pd.DataFrame(self.features, columns=[f'feature_{i}' for i in range(self.features.shape[1])])
        self.feature_df['patient_id'] = self.feature_names

        # 臨床データのコピーを作成し、patient_id を文字列に変換
        clinical_data_copy = self.clinical_data.copy()
        clinical_data_copy['patient_id'] = clinical_data_copy['patient_id'].astype(str)

        # データマージ
        self.merged_data = self.feature_df.merge(clinical_data_copy, on='patient_id', how='inner')
        print(f"Dataset created: {len(self.merged_data)} samples")


def load_dataset(dataset: str) -> Dataset:
    """指定されたデータセット（morph/patho2）を読み込み"""
    if dataset == 'morph':
        base_dir = 'data/DLBCL-Morph'
        clinical_data = pd.read_csv(f'{base_dir}/clinical_data_cleaned.csv')
        target_cols = ['MYC IHC', 'BCL2 IHC', 'BCL6 IHC', 'CD10 IHC', 'MUM1 IHC', 'HANS',
                       'BCL6 FISH', 'MYC FISH', 'BCL2 FISH',
                       'Age', 'LDH', 'ECOG PS', 'Stage', 'IPI Risk Group']
        # IHCマーカーの二値化
        clinical_data['MYC IHC'] = (clinical_data['MYC IHC'] >= 40).astype(int)
        clinical_data['BCL2 IHC'] = (clinical_data['BCL2 IHC'] >= 50).astype(int)
        clinical_data['BCL6 IHC'] = (clinical_data['BCL6 IHC'] >= 30).astype(int)
    elif dataset == 'patho2':
        base_dir = 'data/DLBCL-Patho2'
        clinical_data = pd.read_csv(f'{base_dir}/clinical_data_cleaned_path2_mod.csv')
        # NOTE: patho2はEBVをエクストラで持つが、臨床情報がない
        target_cols = ['MYC IHC', 'BCL2 IHC', 'BCL6 IHC', 'CD10 IHC', 'MUM1 IHC', 'HANS', 'EBV',
                       'Age']
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    print(f"Clinical data loaded: {len(clinical_data)} patients")

    # 特徴量データ読み込み
    with h5py.File(f'{base_dir}/slide_features.h5', 'r') as f:
        features = f['features'][:]
        feature_names = f['names'][:].astype(str)

    return Dataset(
        name=dataset,
        features=features,
        feature_names=feature_names,
        clinical_data=clinical_data,
        target_cols=target_cols
    )


def merge_dataset(dataset_morph: Dataset, dataset_patho2: Dataset) -> Dataset:
    """両データセットを結合してmergedデータセットを作成"""
    print("mergedデータセットを作成中...")

    # 特徴量データを結合
    combined_features = np.vstack([dataset_morph.features, dataset_patho2.features])

    # 特徴量名を結合
    combined_feature_names = np.concatenate([dataset_morph.feature_names, dataset_patho2.feature_names])

    # 臨床データを結合（共通カラムのみ）
    morph_clinical = dataset_morph.clinical_data.copy()
    patho2_clinical = dataset_patho2.clinical_data.copy()

    # データセット識別子を追加
    morph_clinical['dataset'] = 'morph'
    patho2_clinical['dataset'] = 'patho2'

    # 共通カラムで結合
    common_cols = list(set(morph_clinical.columns) & set(patho2_clinical.columns))
    combined_clinical = pd.concat([
        morph_clinical[common_cols],
        patho2_clinical[common_cols]
    ], ignore_index=True)

    # 共通ターゲットカラムを設定
    target_cols = ['Age']

    return Dataset(
        name='merged',
        features=combined_features,
        feature_names=combined_feature_names,
        clinical_data=combined_clinical,
        target_cols=target_cols
    )

