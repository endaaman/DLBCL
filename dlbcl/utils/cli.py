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


class BaseMLArgs(BaseModel):
    seed: int = get_global_seed()
    dataset: str = param('morph', choices=['morph', 'patho2'])

class BaseMLCLI(AutoCLI):
    class CommonArgs(BaseMLArgs):
        device: str = 'cuda'
        use_combat: bool = param(False, description="Combat補正を使用するか")

    def _pre_common(self, a: BaseMLArgs):
        fix_global_seed(a.seed)

        # Combat補正フラグの保存
        self.use_combat = getattr(a, 'use_combat', False)

        # 出力パス設定（Combat補正の有無で切り替え）
        if self.use_combat:
            self.base_output_path = Path('out/combat')
        else:
            self.base_output_path = Path('out')

        # データセットディレクトリ設定
        if a.dataset == 'morph':
            self.dataset_dir = Path('./data/DLBCL-Morph/')
        elif a.dataset == 'patho2':
            self.dataset_dir = Path('./data/DLBCL-Patho2/')
        else:
            raise ValueError('Invalid dataset', a.dataset)

        # 統計解析・特徴量解析で使用する共通データの読み込み
        if self.use_combat:
            # Combat補正版データ読み込み
            self._load_combat_corrected_data(a.dataset)
        else:
            # 通常データ読み込み
            data_dict = load_common_data(a.dataset)
            if data_dict:
                self.clinical_data = data_dict['clinical_data']
                self.features = data_dict['features']
                self.feature_names = data_dict['feature_names']
                self.patient_ids = data_dict['patient_ids']
                self.feature_df = data_dict['feature_df']
                self.merged_data = data_dict['merged_data']

                # 特徴量カラムを取得
                self.feature_cols = [col for col in self.merged_data.columns if col.startswith('feature_')]

                # 対象カラム設定
                self._setup_target_columns(a.dataset)
            else:
                # データ読み込み失敗時
                self._initialize_empty_data()

        super()._pre_common(a)

    def _load_combat_corrected_data(self, dataset):
        """Combat補正されたデータを読み込み"""
        print("Combat補正データを読み込み中...")

        # 両データセットを読み込み
        morph_dict = load_common_data('morph')
        patho2_dict = load_common_data('patho2')

        if morph_dict is None or patho2_dict is None:
            print("データセット読み込みに失敗しました")
            self._initialize_empty_data()
            return

        # 特徴量データを結合してCombat補正
        corrected_features = self._apply_combat_correction(morph_dict, patho2_dict)

        # 指定されたデータセットの補正後データを復元
        if dataset == 'morph':
            n_morph = len(morph_dict['patient_ids'])
            self.clinical_data = morph_dict['clinical_data']
            self.features = corrected_features[:n_morph]
            self.feature_names = morph_dict['feature_names']
            self.patient_ids = morph_dict['patient_ids']
            # 元のmerged_dataを使用（既に正しくマージされている）
            self.merged_data = morph_dict['merged_data'].copy()
            # 特徴量部分だけを補正後のものに置き換え
            for i in range(self.features.shape[1]):
                self.merged_data[f'feature_{i}'] = self.features[:, i]
        else:  # patho2
            n_morph = len(morph_dict['patient_ids'])
            self.clinical_data = patho2_dict['clinical_data']
            self.features = corrected_features[n_morph:]
            self.feature_names = patho2_dict['feature_names']
            self.patient_ids = patho2_dict['patient_ids']
            # 元のmerged_dataを使用（既に正しくマージされている）
            self.merged_data = patho2_dict['merged_data'].copy()
            # 特徴量部分だけを補正後のものに置き換え
            for i in range(self.features.shape[1]):
                self.merged_data[f'feature_{i}'] = self.features[:, i]

        # 特徴量カラムを取得
        self.feature_cols = [col for col in self.merged_data.columns if col.startswith('feature_')]

        # 対象カラム設定
        self._setup_target_columns(dataset)

        print(f"Combat補正完了: {dataset}データセット {self.features.shape}")

    def _apply_combat_correction(self, morph_dict, patho2_dict):
        """Combat補正を適用"""
        print("Combat補正を適用中...")

        # 特徴量データを結合
        morph_features = morph_dict['features']
        patho2_features = patho2_dict['features']
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
        return corrected_features

    def _setup_target_columns(self, dataset):
        """対象カラムの設定"""
        if dataset == 'morph':
            self.target_cols = ['MYC IHC', 'BCL2 IHC', 'BCL6 IHC', 'CD10 IHC',
                              'MUM1 IHC', 'HANS', 'BCL6 FISH', 'MYC FISH', 'BCL2 FISH']
        else:
            # Patho2用の自動検出
            target_cols = []
            exclude_cols = ['patient_id', 'CD10_binary']  # CD10_binaryを除外
            for col in self.merged_data.columns:
                if col.startswith('feature_') or col in exclude_cols:
                    continue
                clean_values = self.merged_data[col].dropna()
                if len(clean_values) == 0:
                    continue
                unique_vals = clean_values.unique()
                if len(unique_vals) == 2 and set(unique_vals).issubset({0, 1, True, False}):
                    target_cols.append(col)
                elif len(unique_vals) == 2:
                    target_cols.append(col)
            self.target_cols = target_cols

    def _initialize_empty_data(self):
        """データ読み込み失敗時の初期化"""
        self.clinical_data = None
        self.features = None
        self.feature_names = None
        self.patient_ids = None
        self.feature_df = None
        self.merged_data = None
        self.feature_cols = []
        self.target_cols = []

