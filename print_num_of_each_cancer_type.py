import argparse
import numpy as np
import os
import pandas as pd
import pickle
import pyreadr
import scanpy as sc
from tqdm import tqdm

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.patches import Patch

from sklearn.model_selection import cross_validate
from sklearn.linear_model import LogisticRegression

data_dir = '/data1/chanj3/LUAS.multiome.results/epigenetic/TCGA_modeling/out'
tcga_data = pyreadr.read_r(f'{data_dir}/tcga_log2cpm_jointTMMwsp.rds')[None]
adata = sc.AnnData(tcga_data.T)
adata.obs['cancer_type'] = [s.split('_')[0] for s in adata.obs_names]
cancer_types = ['LUAD', 'LUSC']
adata = adata[adata.obs['cancer_type'].isin(cancer_types)]
print('adata.shape', adata.shape)
print(adata.obs['cancer_type'].value_counts())
