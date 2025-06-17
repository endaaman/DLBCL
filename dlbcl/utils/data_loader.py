import pandas as pd
import numpy as np
import h5py
from pathlib import Path


def load_common_data(dataset: str):
    """共通データの読み込み"""
    try:
        # 臨床データ読み込み
        if dataset == 'patho2':
            clinical_data = pd.read_csv(f'data/DLBCL-{dataset.capitalize()}/clinical_data_cleaned_path2_mod.csv')
        else:
            clinical_data = pd.read_csv(f'data/DLBCL-{dataset.capitalize()}/clinical_data_cleaned.csv')
        print(f"Clinical data loaded: {len(clinical_data)} patients")

        # 特徴量データ読み込み
        with h5py.File(f'data/DLBCL-{dataset.capitalize()}/slide_features.h5', 'r') as f:
            features = f['features'][:]
            feature_names = f['names'][:]

        # patient_IDの抽出
        patient_ids = [name.decode().split('_')[0] for name in feature_names]

        # 特徴量データフレーム作成
        feature_df = pd.DataFrame(features, columns=[f'feature_{i}' for i in range(features.shape[1])])
        feature_df['patient_id'] = patient_ids

        # 臨床データの patient_id を文字列に変換
        clinical_data['patient_id'] = clinical_data['patient_id'].astype(str)

        # データマージ
        merged_data = feature_df.merge(clinical_data, on='patient_id', how='inner')
        print(f"Merged data: {len(merged_data)} samples")

        return {
            'clinical_data': clinical_data,
            'features': features,
            'feature_names': feature_names,
            'patient_ids': patient_ids,
            'feature_df': feature_df,
            'merged_data': merged_data
        }

    except Exception as e:
        print(f"Warning: Failed to load common data: {e}")
        # 一部のコマンド（既存のクラスタリングなど）では共通データが不要な場合があるため、
        # エラーは警告に留めて処理を続行
        return None
