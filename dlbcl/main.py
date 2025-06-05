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
from pydantic_autocli import param
import hdbscan
import torch
import timm
from umap import UMAP
from gigapath import slide_encoder

from .utils import BaseMLCLI, BaseMLArgs

warnings.filterwarnings('ignore', category=FutureWarning, message='.*force_all_finite.*')



class CLI(BaseMLCLI):
    class CommonArgs(BaseMLArgs):
        # This includes `--seed` param
        device: str = 'cuda'
        dataset: str = param('morph', choices=['morph', 'patho2'])

    def pre_common(self, a:CommonArgs):
        if a.dataset == 'morph':
            self.dataset_dir = Path('./data/DLBCL-Morph/')
        elif a.dataset == 'pathp2':
            self.dataset_dir = Path('./data/DLBCL-Patho2/')
        else:
            raise ValueError('Invalid dataset', a.dataset)

    class ExtractGlobalFeaturesArgs(CommonArgs):
        noshow: bool = False

    def run_gather_features(self, a):
        featuress = []
        lengths = []
        for dir in sorted(glob('data/dataset/*')):
            name = os.path.basename(dir)
            for i, h5_path in enumerate(sorted(glob(f'{dir}/*.h5'))):
                with h5py.File(h5_path, 'r') as f:
                    features = f['features'][:]
                    lengths.append(len(features))
                    featuress.append(features)

        features = np.concatenate(featuress)

        with h5py.File('data/global_features.h5', 'w') as f:
            f.create_dataset('global_features', data=features)
            f.create_dataset('lengths', data=np.array(lengths))

    def run_gather_slide_features(self, a):
        data = []
        features = []
        dirs = sorted(glob(str(self.dataset_dir / 'dataset/*')))
        assert len(dirs) > 0
        for dir in tqdm(dirs):
            name = os.path.basename(dir)
            for i, h5_path in enumerate(sorted(glob(f'{dir}/*.h5'))):
                m = re.match(r'^\d\d\d\d\d_\d\.h5$', Path(h5_path).name)
                if not m:
                    print('skip', h5_path)
                    continue
                print('loading', h5_path)
                with h5py.File(h5_path, 'r') as f:
                    features.append(f['gigapath/slide_feature'][:])
                data.append({
                    'name': name,
                    'order': i,
                    'filename': os.path.basename(h5_path),
                })

        df = pd.DataFrame(data)
        features = np.array(features)
        print('features', features.shape)

        o = str(self.dataset_dir / 'slide_features.h5')
        with h5py.File(o, 'w') as f:
            f.create_dataset('features', data=features)
            f.create_dataset('names', data=df['name'].values)
            f.create_dataset('orders', data=df['order'].values)
            f.create_dataset('filenames', data=df['filename'].values)
        print(f'wrote {o}')

    class UMAPArgs(CommonArgs):
        keys: list[str] = param(['DBSCAN cluster'])

    def run_umap(self, a:UMAPArgs):
        df  = pd.read_csv('./data/DLBCL-Morph/DBSCAN_clusters_with_ids.csv', index_col=0)
        keys = a.keys
        if 'ALL' in a.keys:
            keys = list(df.columns)
        with h5py.File('./data/DLBCL-Morph/slide_features.h5', 'r') as f:
            features = f['features'][()]
            names = f['names'][()]
            orders = f['orders'][()]
            data = []
            for name, order in zip(names, orders):
                label = f'{name.decode("utf-8")}_{order}'
                v = {}
                for key in keys:
                    v[key] = df.loc[label, key]
                data.append(v)

        df_labels = pd.DataFrame(data)

        for key in keys:
            plt.close()
            labels = df_labels[key]
            umap = UMAP()
            embs = umap.fit_transform(features)
            unique_labels = sorted(np.unique(labels))
            cmap = plt.get_cmap('tab20')
            for i, label in enumerate(unique_labels):
                mask = labels == label
                if isinstance(label, (int, np.integer)):
                    c = 'gray' if label < 0 else cmap(label)
                else:
                    c = cmap(i)
                plt.scatter(embs[mask,0], embs[mask,1], c=c, label=f'{key} {label}')
            plt.legend()
            plt.savefig(f'./out/umap_{key.replace(" ", "_")}.png')
        plt.show()



    class ClusterArgs(CommonArgs):
        target: str = Field('cluster', s='-T')
        noshow: bool = False

    def run_cluster(self, a):
        with h5py.File('./data/DLBCL-Morph/slide_features.h5', 'r') as f:
            features = f['features'][:]
            df = pd.DataFrame({
                'name': [int((v.decode('utf-8'))) for v in f['names'][:]],
                'filename': [v.decode('utf-8') for v in f['filenames'][:]],
                'order': f['orders'][:],
            })

        df_clinical = pd.read_excel('./data/clinical_data_cleaned.xlsx', index_col=0)
        df = pd.merge(
            df,
            df_clinical,
            left_on='name',
            right_index=True,
            how='left'
        )

        print('Loaded features', features.shape)
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(features)
        # scaled_features = features

        print('UMAP fitting...')
        reducer = umap.UMAP(
                n_neighbors=10,
                min_dist=0.05,
                n_components=2,
                metric='cosine',
                random_state=a.seed,
                n_jobs=1,
            )
        embedding = reducer.fit_transform(scaled_features)
        print('Loaded features', features.shape)

        if a.target in [
                'HDBSCAN',
                'CD10 IHC', 'MUM1 IHC', 'HANS', 'BCL6 FISH', 'MYC FISH', 'BCL2 FISH',
                'ECOG PS', 'LDH', 'EN', 'Stage', 'IPI Score',
                'IPI Risk Group (4 Class)', 'RIPI Risk Group', 'Follow-up Status',
                ]:
            mode = 'categorical'
        elif a.target in ['MYC IHC', 'BCL2 IHC', 'BCL6 IHC', 'Age', 'OS', 'PFS']:
            mode = 'numeric'
        else:
            raise RuntimeError('invalid target', a.target)


        plt.figure(figsize=(10, 8))
        marker_size = 15

        if mode == 'categorical':
            labels = df[a.target].fillna(-1)
            n_labels = len(set(labels))
            cmap = plt.cm.viridis

            noise_mask = labels == -1
            valid_labels = sorted(list(set(labels[~noise_mask])))
            norm = plt.Normalize(min(valid_labels or [0]), max(valid_labels or [1]))
            for label in valid_labels:
                mask = labels == label
                color = cmap(norm(label))
                plt.scatter(
                    embedding[mask, 0], embedding[mask, 1], c=[color],
                    s=marker_size, label=f'{a.target} {label}'
                )

            if np.any(noise_mask):
                plt.scatter(
                    embedding[noise_mask, 0], embedding[noise_mask, 1], c='gray',
                    s=marker_size, marker='x', label='Noise/NaN',
                )

        else:
            values = df[a.target]
            norm = Normalize(vmin=values.min(), vmax=values.max())
            values = values.fillna(-1)
            has_value = values > 0
            cmap = plt.cm.viridis
            scatter = plt.scatter(embedding[has_value, 0], embedding[has_value, 1], c=values[has_value],
                                  s=marker_size, cmap=cmap, norm=norm, label=a.target,)
            if np.any(has_value):
                plt.scatter(embedding[~has_value, 0], embedding[~has_value, 1], c='gray',
                            s=marker_size, marker='x', label='NaN')
            cbar = plt.colorbar(scatter)
            cbar.set_label(a.target)

        plt.title(f'UMAP + {a.target}')
        plt.xlabel('UMAP Dimension 1')
        plt.ylabel('UMAP Dimension 2')
        # plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.legend()
        plt.tight_layout()
        os.makedirs('out/morph', exist_ok=True)
        name = a.target.replace(' ', '_')
        plt.savefig(f'out/morph/umap_{name}.png')
        if not a.noshow:
            plt.show()


    class GlobalClusterArgs(CommonArgs):
        noshow: bool = False
        n_samples: int = Field(100, s='-N')

    def run_global_cluster(self, a):
        features = []
        images = []
        dfs = []
        for dir in sorted(glob('data/dataset/*')):
            name = os.path.basename(dir)
            for i, h5_path in enumerate(sorted(glob(f'{dir}/*.h5'))):
                with h5py.File(h5_path, 'r') as f:
                    patch_count = f['metadata/patch_count'][()]
                    ii = np.random.choice(patch_count, size=a.n_samples, replace=False)
                    ii = np.sort(ii)
                    features.append(f['features'][ii])
                    images.append(f['patches'][ii])
                    df_wsi = pd.DataFrame({'index': ii})
                df_wsi['name'] = int(os.path.basename(dir))
                df_wsi['order'] = i
                df_wsi['filename'] = os.path.basename(h5_path)
                dfs.append(df_wsi)

        df = pd.concat(dfs)
        df_clinical = pd.read_excel('./data/clinical_data_cleaned.xlsx', index_col=0)
        df = pd.merge(
            df,
            df_clinical,
            left_on='name',
            right_index=True,
            how='left'
        )

        features = np.concatenate(features)
        images = np.concatenate(images)
        # images = [Image.fromarray(i) for i in images]

        print('Loaded features', features.dtype, features.shape)
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(features)

        print('UMAP fitting...')
        reducer = umap.UMAP(
                # n_neighbors=80,
                # min_dist=0.3,
                n_components=2,
                metric='cosine',
                min_dist=0.5,
                spread=2.0
                # random_state=a.seed
            )
        embedding = reducer.fit_transform(scaled_features)
        print('UMAP ok')

        # scatter = plt.scatter(embedding[:, 0], embedding[:, 1], s=5, c=df['LDH'].values)
        # hover_images_on_scatters([scatter], [images])

        target = 'HANS'

        fig, ax = plt.subplots(figsize=(10, 8))
        plt.scatter(embedding[:, 0], embedding[:, 1], alpha=0)
        for (x, y), image, (_idx, row) in zip(embedding, images, df.iterrows()):
            img = OffsetImage(image, zoom=.125)
            value = row[target]

            text = TextArea(row['name'], textprops=dict(color='#000', ha='center'))
            vpack = VPacker(children=[text, img], align='center', pad=1)

            cmap = plt.cm.viridis
            color = '#333' if value < 0 else cmap(value)
            bbox_props = dict(boxstyle='square,pad=0.1', edgecolor=color, linewidth=1, facecolor='none')

            ab = AnnotationBbox(vpack, (x, y), frameon=True, bboxprops=bbox_props)
            ax.add_artist(ab)

        plt.title(f'UMAP')
        plt.xlabel('UMAP Dimension 1')
        plt.ylabel('UMAP Dimension 2')
        # plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    cli = CLI()
    cli.run()

