import os
import re
import warnings
from pathlib import Path

from glob import glob
from tqdm import tqdm
from pydantic import Field
from PIL import Image, ImageDraw
import cv2
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt, colors as mcolors
from matplotlib.colors import LinearSegmentedColormap, Normalize
import seaborn as sns
import h5py
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from scipy import stats
from scipy.stats import pearsonr, spearmanr, chi2_contingency, mannwhitneyu, kruskal, f_oneway, ttest_ind
from scipy.cluster.hierarchy import dendrogram, linkage, leaves_list
from scipy.spatial.distance import pdist
from pydantic_autocli import param, AutoCLI
import hdbscan
import torch
import timm
from umap import UMAP
from gigapath import slide_encoder
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
import matplotlib.patches as patches
import scanpy as sc
import anndata as ad



class CLI(AutoCLI):
    class CommonArgs(BaseModel):
        pass

    def run_gather_features(self, a):
        data = []
        slide_features = []
        center_features = []
        medoid_features = []
        dirs = sorted(glob(str(self.dataset_dir / 'dataset/*')))
        assert len(dirs) > 0
        for dir in tqdm(dirs):
            name = os.path.basename(dir)
            for i, h5_path in enumerate(sorted(glob(f'{dir}/*.h5'))):
                h5_filename = Path(h5_path).name
                # Support both Morph (12345_0.h5) and Patho2 (20-0494_1_1.h5) patterns
                if a.dataset == 'morph':
                    m = re.match(r'^\d\d\d\d\d_\d\.h5$', h5_filename)
                else:  # patho2
                    m = re.match(r'^[\d\-]+_\d+_\d+\.h5$', h5_filename)

                if not m:
                    print('skip', h5_path)
                    continue
                with h5py.File(h5_path, 'r') as f:
                    slide_features.append(f['gigapath/slide_feature'][:])
                    features = f['gigapath/features'][:]
                    # 中心
                    center_feature = np.mean(features, axis=0)
                    # 中心最近傍
                    medoid_idx = np.argmin(np.linalg.norm(features - center_feature, axis=1))
                    medoid_feature = features[medoid_idx]
                    center_features.append(center_feature)
                    medoid_features.append(medoid_feature)

                data.append({
                    'name': name,
                    'order': i,
                    'filename': os.path.basename(h5_path),
                })

        df = pd.DataFrame(data)
        slide_features = np.array(slide_features)
        center_features = np.array(center_features)
        medoid_features = np.array(medoid_features)
        print('slide_features', slide_features.shape)
        print('center_features', center_features.shape)
        print('medoid_features', medoid_features.shape)

        o = str(self.dataset_dir / 'global_features.h5')
        with h5py.File(o, 'w') as f:
            f.create_dataset('gigapath/slide_features', data=slide_features)
            f.create_dataset('gigapath/center_features', data=center_features)
            f.create_dataset('gigapath/medoid_features', data=medoid_features)
            f.create_dataset('names', data=df['name'].values)
            f.create_dataset('orders', data=df['order'].values)
            f.create_dataset('filenames', data=df['filename'].values)
        print(f'wrote {o}')


if __name__ == '__main__':
    cli = CLI()
    cli.run()
