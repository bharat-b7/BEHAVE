"""
Code to create train and test splits for BEHAVE dataset.
Author: Bharat Lal Bhatnagar
Cite: BEHAVE: Dataset and Method for Tracking Human Object Interactions, CVPR'22
"""

import numpy as np
from glob import glob
from tqdm import tqdm
import pickle as pkl
import os
from os.path import split, join, exists
import json
import yaml
import sys
sys.path.append('../')
with open("PATHS.yml", 'r') as stream:
    paths = yaml.safe_load(stream)
PROCESSED_PATH, BEHAVE_PATH = paths['PROCESSED_PATH'], paths['BEHAVE_PATH']
EXT = '02'
SUFFIX = '01'


def make_split(train, test, save_path):
    tr_lis, te_lis, skipped = [], [], 0
    for dataset in tqdm(train):
        for scan in glob(join(PROCESSED_PATH, dataset, '*')):
            name = split(scan)[1]
            if not exists(join(scan, 'voxelized',
                               name + '_voxelized_point_cloud_res_128_points_5000_{}.npz'.format(SUFFIX))) or \
                    not exists(join(scan, 'boundary_sampling', name + '_sigma_0.015_{}.npz'.format(EXT))) or \
                    not exists(join(scan, 'boundary_sampling', name + '_sigma_0.15_{}.npz'.format(EXT))):
                skipped += 1
                continue
            tr_lis.append(scan)
    print('train', len(tr_lis), 'skipped', skipped)

    skipped = 0
    for dataset in tqdm(test):
        for scan in glob(join(PROCESSED_PATH, dataset, '*')):
            name = split(scan)[1]
            if not exists(join(scan, 'voxelized',
                               name + '_voxelized_point_cloud_res_128_points_5000_{}.npz'.format(SUFFIX))) or \
                    not exists(join(scan, 'boundary_sampling', name + '_sigma_0.015_{}.npz'.format(EXT))) or \
                    not exists(join(scan, 'boundary_sampling', name + '_sigma_0.15_{}.npz'.format(EXT))):
                skipped += 1
                continue
            te_lis.append(scan)
    print('test', len(te_lis), 'skipped', skipped)
    pkl.dump({'train': tr_lis, 'val': te_lis},
             open(save_path, 'wb'))


if __name__ == "__main__":
    with open('assets/split.json', 'r') as f:
        data = json.load(f)
    make_split(data['train'], data['test'], 'assets/data_split_01.pkl')