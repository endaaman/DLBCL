import os
import sys
import re
from string import capwords
import inspect
import asyncio
from typing import Callable, Type
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import h5py

from pydantic import BaseModel, Field
from pydantic_autocli import AutoCLI, param
from combat.pycombat import pycombat

from .seed import fix_global_seed, get_global_seed


class Dataset:
    def __init__(self, dataset: str):
        if dataset == 'morph':
            base_dir = 'data/DLBCL-Morph'
            self.clinical_data = pd.read_csv(f'{base_dir}/clinical_data_cleaned.csv')
            self.target_cols = ['MYC IHC', 'BCL2 IHC', 'BCL6 IHC', 'CD10 IHC', 'MUM1 IHC', 'HANS',
                                'BCL6 FISH', 'MYC FISH', 'BCL2 FISH',
                                'Age', 'LDH', 'ECOG PS', 'Stage', 'IPI Risk Group']
            self.clinical_data['MYC IHC'] = (self.clinical_data['MYC IHC'] > 40).astype(int)
            self.clinical_data['BCL2 IHC'] = (self.clinical_data['BCL2 IHC'] > 50).astype(int)
            self.clinical_data['BCL6 IHC'] = (self.clinical_data['BCL6 IHC'] > 30).astype(int)
        else:
            base_dir = 'data/DLBCL-Patho2'
            self.clinical_data = pd.read_csv(f'{base_dir}/clinical_data_cleaned_path2_mod.csv')
            self.target_cols = ['MYC IHC', 'BCL2 IHC', 'BCL6 IHC', 'CD10 IHC', 'MUM1 IHC', 'HANS',
                                'Age']

        print(f"Clinical data loaded: {len(clinical_data)} patients")

        # 特徴量データ読み込み
        with h5py.File(f'{base_dir}/slide_features.h5', 'r') as f:
            self.features = f['features'][:]
            self.feature_names = f['names'][:]

        # 特徴量データフレーム作成
        self.feature_df = pd.DataFrame(self.features, columns=[f'feature_{i}' for i in range(self.features.shape[1])])
        self.feature_df['patient_id'] = self.feature_names

        # 臨床データの patient_id を文字列に変換
        self.clinical_data['patient_id'] = self.clinical_data['patient_id'].astype(str)

        # データマージ
        self.merged_data = self.feature_df.merge(self.clinical_data, on='patient_id', how='inner')
        print(f"Merged data: {len(self.merged_data)} samples")


class BaseMLCLI(AutoCLI):
    class CommonArgs(BaseModel):
        device: str = 'cuda'
        use_combat: bool = param(False, description="Combat補正を使用するか")
        seed: int = get_global_seed()
        target: str = param('morph', choices=['morph', 'patho2', 'merged'])

    def prepare(self, a: CommonArgs):
        fix_global_seed(a.seed)

        self.dataset_morph = Dataset('morph')
        self.dataset_patho2 = Dataset('patho2')

        # 特徴量カラムを取得
        self.feature_cols = [col for col in self.merged_data.columns if col.startswith('feature_')]

        if a.use_combat:
            self._apply_combat_correction()


    def _apply_combat_correction(self):
        """Combat補正を適用"""
        print("Combat補正を適用中...")

        # 特徴量データを結合
        morph_features = self.dataset_morph.features
        patho2_features = self.dataset_patho2.features
        combined_features = np.vstack([morph_features, patho2_features])

        # バッチ変数作成（データセット識別子）
        n_morph = len(morph_dict['patient_ids'])
        n_patho2 = len(patho2_dict['patient_ids'])
        batch = np.array([0] * n_morph + [1] * n_patho2)

        # Combat補正実行
        # NaNを含む特徴量を除外
        valid_features = ~np.isnan(combined_features).any(axis=0)
        if not valid_features.any():
            print("警告: すべての特徴量にNaNが含まれています")
            return combined_features

        features_clean = combined_features[:, valid_features]

        # pycombatはDataFrameを期待するため変換
        features_df = pd.DataFrame(features_clean.T)
        corrected_df = pycombat(features_df, batch)
        corrected_clean = corrected_df.values.T

        # 元の形状に戻す
        corrected_features = combined_features.copy()
        corrected_features[:, valid_features] = corrected_clean

        print(f"Combat補正成功: {corrected_features.shape[1]}特徴量中{valid_features.sum()}特徴量を補正")

        self.dataset_morph.features
        self.dataset_patho2.features
        # に補正後のものをそれぞれ書き込む

