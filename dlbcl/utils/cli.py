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
from dataclasses import dataclass

from pydantic import BaseModel, Field
from pydantic_autocli import AutoCLI, param
from combat.pycombat import pycombat

from .seed import fix_global_seed, get_global_seed


@dataclass
class Dataset:
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
        self.feature_df['patient_id'] = self.feature_names.astype(str)
        
        # 臨床データの patient_id を文字列に変換
        self.clinical_data['patient_id'] = self.clinical_data['patient_id'].astype(str)
        
        # データマージ
        self.merged_data = self.feature_df.merge(self.clinical_data, on='patient_id', how='inner')
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
        clinical_data['MYC IHC'] = (clinical_data['MYC IHC'] > 40).astype(int)
        clinical_data['BCL2 IHC'] = (clinical_data['BCL2 IHC'] > 50).astype(int)
        clinical_data['BCL6 IHC'] = (clinical_data['BCL6 IHC'] > 30).astype(int)
    elif dataset == 'patho2':
        base_dir = 'data/DLBCL-Patho2'
        clinical_data = pd.read_csv(f'{base_dir}/clinical_data_cleaned_path2_mod.csv')
        target_cols = ['MYC IHC', 'BCL2 IHC', 'BCL6 IHC', 'CD10 IHC', 'MUM1 IHC', 'HANS', 'Age']
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    
    print(f"Clinical data loaded: {len(clinical_data)} patients")
    
    # 特徴量データ読み込み
    with h5py.File(f'{base_dir}/slide_features.h5', 'r') as f:
        features = f['features'][:]
        feature_names = f['names'][:]
    
    return Dataset(
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
        features=combined_features,
        feature_names=combined_feature_names,
        clinical_data=combined_clinical,
        target_cols=target_cols
    )


class BaseMLCLI(AutoCLI):
    class CommonArgs(BaseModel):
        device: str = 'cuda'
        use_combat: bool = param(False, description="Combat補正を使用するか")
        seed: int = get_global_seed()
        target: str = param('morph', choices=['morph', 'patho2', 'merged'])

    def prepare(self, a: CommonArgs):
        fix_global_seed(a.seed)

        # 両方のデータセットを読み込み
        self.dataset_morph = load_dataset('morph')
        self.dataset_patho2 = load_dataset('patho2')
        
        # mergedデータセットを作成
        self.dataset_merged = merge_dataset(self.dataset_morph, self.dataset_patho2)
        
        # Combat補正を適用
        if a.use_combat:
            self._apply_combat_correction()
            
        # 解析対象データセットを設定
        self._set_target_dataset(a.target)


    def _apply_combat_correction(self):
        """Combat補正を両データセット間に適用"""
        print("Combat補正を適用中...")
        
        # 特徴量データを結合
        morph_features = self.dataset_morph.features
        patho2_features = self.dataset_patho2.features
        combined_features = np.vstack([morph_features, patho2_features])
        
        # バッチ変数作成（データセット識別子）
        n_morph = len(self.dataset_morph.feature_names)
        n_patho2 = len(self.dataset_patho2.feature_names)
        batch = np.array([0] * n_morph + [1] * n_patho2)
        
        # NaNを含む特徴量を除外
        valid_features = ~np.isnan(combined_features).any(axis=0)
        if not valid_features.any():
            print("警告: すべての特徴量にNaNが含まれています")
            return
            
        features_clean = combined_features[:, valid_features]
        
        # pycombatはDataFrameを期待するため変換
        features_df = pd.DataFrame(features_clean.T)
        corrected_df = pycombat(features_df, batch)
        corrected_clean = corrected_df.values.T
        
        # 元の形状に戻す
        corrected_features = combined_features.copy()
        corrected_features[:, valid_features] = corrected_clean
        
        print(f"Combat補正成功: {corrected_features.shape[1]}特徴量中{valid_features.sum()}特徴量を補正")
        
        # 補正後の特徴量を各データセットに書き戻し
        self.dataset_morph.features = corrected_features[:n_morph]
        self.dataset_patho2.features = corrected_features[n_morph:]
        
        # 各データセットのデータフレームを再構築
        self.dataset_morph.__post_init__()
        self.dataset_patho2.__post_init__()
        
        # mergedデータセットも再作成
        self.dataset_merged = merge_dataset(self.dataset_morph, self.dataset_patho2)
        
    def _set_target_dataset(self, target: str):
        """解析対象データセットを設定"""
        if target == 'morph':
            self.dataset = self.dataset_morph
        elif target == 'patho2':
            self.dataset = self.dataset_patho2
        elif target == 'merged':
            self.dataset = self.dataset_merged
        else:
            raise ValueError(f"Unknown target: {target}")
            
        # 特徴量カラムを取得
        self.feature_cols = [col for col in self.dataset.merged_data.columns if col.startswith('feature_')]
        
        print(f"解析対象データセット: {target} ({len(self.dataset.merged_data)} samples)")

