import os
import sys
import re
from string import capwords
import inspect
import asyncio
from typing import Callable, Type
import argparse
from pathlib import Path

from pydantic import BaseModel, Field
from pydantic_autocli import AutoCLI, param

from .seed import fix_global_seed, get_global_seed
from .data_loader import load_common_data


class BaseMLArgs(BaseModel):
    seed: int = get_global_seed()
    dataset: str = param('morph', choices=['morph', 'patho2'])

class BaseMLCLI(AutoCLI):
    class CommonArgs(BaseMLArgs):
        device: str = 'cuda'

    def _pre_common(self, a: BaseMLArgs):
        fix_global_seed(a.seed)
        
        # データセットディレクトリ設定
        if a.dataset == 'morph':
            self.dataset_dir = Path('./data/DLBCL-Morph/')
        elif a.dataset == 'patho2':
            self.dataset_dir = Path('./data/DLBCL-Patho2/')
        else:
            raise ValueError('Invalid dataset', a.dataset)

        # 統計解析・特徴量解析で使用する共通データの読み込み
        data_dict = load_common_data(a.dataset)
        if data_dict:
            self.clinical_data = data_dict['clinical_data']
            self.features = data_dict['features']
            self.feature_names = data_dict['feature_names']
            self.patient_ids = data_dict['patient_ids']
            self.feature_df = data_dict['feature_df']
            self.merged_data = data_dict['merged_data']
        
        super()._pre_common(a)
