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
import umap
import matplotlib.pyplot as plt

from .seed import fix_global_seed, get_global_seed
from .dataset import Dataset, load_dataset, merge_dataset



class ExperimentCLI(AutoCLI):
    class CommonArgs(BaseModel):
        use_combat: bool = param(False, description="Combat補正を使用するか")
        seed: int = get_global_seed()
        dataset: str = param('morph', choices=['morph', 'patho2', 'merged'])

    def prepare(self, a: CommonArgs):
        fix_global_seed(a.seed)

        # 両方のデータセットを読み込み
        self.dataset_morph = load_dataset('morph')
        self.dataset_patho2 = load_dataset('patho2')

        # mergedデータセットを作成
        self.dataset_merged = merge_dataset(self.dataset_morph, self.dataset_patho2)

        # Combat補正を適用
        subdir = 'raw'
        if a.use_combat:
            self._apply_combat_correction()
            subdir = 'combat'

        """解析対象データセットを設定"""
        if a.dataset == 'morph':
            self.dataset = self.dataset_morph
        elif a.dataset == 'patho2':
            self.dataset = self.dataset_patho2
        elif a.dataset == 'merged':
            self.dataset = self.dataset_merged
        else:
            raise ValueError(f"Unknown dataset: {a.dataset}")

        self.output_dir = Path(f'out/{subdir}/{a.dataset}')
        self.output_dir.mkdir(parents=True, exist_ok=True)


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



class CLI(ExperimentCLI):
    class UmapArgs(ExperimentCLI.CommonArgs):
        # foo: str = param('defaut', l='--foooo', s='-f') # l for long param, s for short
        foo: list[str] = ['aa', 'bb']

    def run_umap(self, a: UmapArgs):
        print(a.foo)
        """UMAP可視化を実行"""
        features = self.dataset.features

        # UMAP実行
        reducer = umap.UMAP(random_state=a.seed, n_components=2)
        embedding = reducer.fit_transform(features)

        # プロット作成
        plt.figure(figsize=(10, 8))

        # データセット別に色分け（mergedの場合）
        if a.dataset == 'merged' and 'dataset' in self.dataset.merged_data.columns:
            datasets = self.dataset.merged_data['dataset'].values
            # embeddingとdatasetsのサイズを合わせる
            min_size = min(len(embedding), len(datasets))
            embedding_subset = embedding[:min_size]
            datasets_subset = datasets[:min_size]

            for ds in np.unique(datasets_subset):
                mask = datasets_subset == ds
                plt.scatter(embedding_subset[mask, 0], embedding_subset[mask, 1],
                           label=f'{ds}', alpha=0.7, s=50)
            plt.legend()
        else:
            plt.scatter(embedding[:, 0], embedding[:, 1], alpha=0.7, s=50)

        plt.title(f'UMAP - {a.dataset} (Combat: {a.use_combat})')
        plt.xlabel('UMAP 1')
        plt.ylabel('UMAP 2')
        plt.show()

        print(f"UMAP完了: {len(features)} samples, {features.shape[1]} features")
        return True

if __name__ == '__main__':
    cli = CLI()
    cli.run()

