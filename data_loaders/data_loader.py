"""
Code to load data for training and testing.
Author: Bharat Lal Bhatnagar
Cite: BEHAVE: Dataset and Method for Tracking Human Object Interactions, CVPR'22
"""

import os
from os.path import join, split, exists
import pickle as pkl
import numpy as np
from glob import glob
import trimesh
import codecs
import torch

import sys
sys.path.append('..')
import yaml
with open("PATHS.yml", 'r') as stream:
    paths = yaml.safe_load(stream)
sys.path.append(paths['CODE'])
PROCESSED_PATH, BEHAVE_PATH = paths['PROCESSED_PATH'], paths['BEHAVE_PATH']

from utils.preprocess_pointcloud import preprocess
from utils.voxelize_ho import clean_pc
from utils.compute_df_ho import parse_object


class DataLoader(object):
    def __init__(self, mode, res=32, pointcloud_samples=3000, data_path=PROCESSED_PATH,
                 split_file='assets/data_split_01.pkl', suffix='',
                 batch_size=64, num_sample_points=1024, num_workers=12, sample_distribution=[1], sample_sigmas=[0.005],
                 ext=''):
        # sample distribution should contain the percentage of uniform samples at index [0]
        # and the percentage of N(0,sample_sigma[i-1]) samples at index [i] (>0).
        self.sample_distribution = np.array(sample_distribution)
        self.sample_sigmas = np.array(sample_sigmas)

        assert np.sum(self.sample_distribution) == 1
        assert np.any(self.sample_distribution < 0) == False
        assert len(self.sample_distribution) == len(self.sample_sigmas)

        self.mode = mode
        self.path = data_path
        with open(split_file, "rb") as f:
            self.split = pkl.load(f)

        self.data = self.split[mode]
        self.res = res
        self.num_sample_points = num_sample_points
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pointcloud_samples = pointcloud_samples
        self.ext = ext
        self.suffix = suffix

        # compute number of samples per sampling method
        self.num_samples = np.rint(self.sample_distribution * self.num_sample_points).astype(np.uint32)

        # get objects
        self.objects = {}
        for n, i in enumerate(glob('assets/objects/*')):
            nam = split(i)[1][:-4]
            self.objects[nam] = n

    def __len__(self):
        return len(self.data)

    def get_loader(self, shuffle=True):
        return torch.utils.data.DataLoader(
            self, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=shuffle,
            worker_init_fn=self.worker_init_fn)

    def worker_init_fn(self, worker_id):
        ''' Worker init function to ensure true randomness.
		'''
        # base_seed = int(os.urandom(4).encode('hex'), 16)
        base_seed = int(codecs.encode(os.urandom(4), 'hex'), 16)
        np.random.seed(base_seed + worker_id)

    def get_object_class(self, path):
        for n, k in enumerate(self.objects):
            if k in path:
                return n
        raise ValueError

    def load_sampling_points(self, file):
        points, df_h, df_o, coords, parts, corr, cent, pca = [], [], [], [], [], [], [], []
        occ_h = []
        for i, num in enumerate(self.num_samples):
            boundary_samples_path = file + '_{}_{}.npz'.format(self.sample_sigmas[i], self.ext)
            try:
                boundary_samples_npz = np.load(boundary_samples_path)
            except:
                print(boundary_samples_path)
            boundary_sample_coords = boundary_samples_npz['grid_coords']
            boundary_sample_points = boundary_samples_npz['points']
            boundary_sample_df_h = boundary_samples_npz['dist_h']
            boundary_sample_df_o = boundary_samples_npz['dist_o']
            boundary_sample_cent = boundary_samples_npz['cent']
            boundary_sample_pca = boundary_samples_npz['pca_axis']

            subsample_indices = np.random.randint(0, len(boundary_sample_points), num)
            points.extend(boundary_sample_points[subsample_indices])
            coords.extend(boundary_sample_coords[subsample_indices])
            df_h.extend(boundary_sample_df_h[subsample_indices])
            df_o.extend(boundary_sample_df_o[subsample_indices])
            pca.extend(np.repeat(boundary_sample_pca[None], num, axis=0))

            if 'parts' in boundary_samples_npz.keys():
                boundary_sample_parts = boundary_samples_npz['parts']
                parts.extend(boundary_sample_parts[subsample_indices])
                boundary_sample_correspondences = boundary_samples_npz['correspondences']
                corr.extend(boundary_sample_correspondences[subsample_indices])
                cent.append(boundary_sample_cent)
            else:
                boundary_sample_parts = None
                boundary_sample_correspondences = None

            if 'occ_h' in boundary_samples_npz.keys():
                boundary_sample_occupancies = boundary_samples_npz['occ_h']
                occ_h.extend(boundary_sample_occupancies[subsample_indices])
            else:
                occ_h = None

        assert len(points) == self.num_sample_points
        assert len(df_o) == self.num_sample_points
        assert len(df_h) == self.num_sample_points
        assert len(coords) == self.num_sample_points
        assert (len(parts) == self.num_sample_points) or (boundary_sample_parts is None)
        assert (len(occ_h) == self.num_sample_points) or (occ_h is None)

        return points, coords, df_h, df_o, parts, corr, cent, pca, occ_h

    def load_voxel_input(self, file):
        occupancies = np.unpackbits(np.load(file)['compressed_occupancies'])
        segmentation = np.unpackbits(np.load(file)['compressed_segmentation'])

        obj = self.get_object_class(file)
        ## human -> occ==1 && seg==1; object -> occ==1 && seg==0
        human = occupancies & segmentation
        object = (occupancies & ~segmentation) * (obj + 2)  # we want 0 to be bg, 1 to be human and then objects
        input = np.maximum(human, object)  # human -> 1; bg -> 0; object -> -1
        input = np.reshape(input, (self.res,) * 3)
        return input

    def __getitem__(self, idx):
        path = self.data[idx]
        name = split(path)[1]
        obj = parse_object(split(split(path)[0])[1])

        voxel_path_full = join(path, 'voxelized', name + '_voxelized_point_cloud_res_{}_points_{}_{}.npz'.format(
            self.res, self.pointcloud_samples, self.suffix))

        input_full = self.load_voxel_input(voxel_path_full)

        if self.mode == 'train':
            boundary_samples_path = join(path, 'boundary_sampling',
                                         name + '_sigma')
            points, coords, df_h, df_o, parts, corr, cent, pca, occ_h = self.load_sampling_points(boundary_samples_path)
            pc_h, pc_o, pose, betas, trans, J = None, None, None, None, None, None
        elif self.mode == 'val':
            boundary_samples_path = join(path, 'boundary_sampling',
                                         name + '_sigma')
            points, coords, df_h, df_o, parts, corr, cent, pca, occ_h = self.load_sampling_points(boundary_samples_path)
        else:  # for testing we might not have annotations
            points, coords, df_h, df_o, parts, corr, cent, pca, occ_h = None, None, None, None, None, [
                [None]], None, None, None

        if self.mode == 'val' or self.mode == 'test':
            # Load original pointclouds
            pc_h = trimesh.load(join(path.replace(PROCESSED_PATH, BEHAVE_PATH), 'person/person.ply'), process=False)
            pc_o = trimesh.load(join(path.replace(PROCESSED_PATH, BEHAVE_PATH), obj, '{}.ply'.format(obj)), process=False)
            if cent is None:
                cent = np.append(pc_h.vertices, pc_o.vertices, axis=0).mean(0)[None]
            pc_h.vertices, _ = preprocess(pc_h.vertices, cent=cent[0])
            pc_o.vertices, _ = preprocess(pc_o.vertices, cent=cent[0])

            # Clean PC using GT fit. Not required if dataset is cleaned.
            obj = trimesh.load(join(path.replace(PROCESSED_PATH, BEHAVE_PATH), obj, 'fit01', '{}_fit.ply'.format(obj)),
                               process=False)
            obj.vertices, _ = preprocess(obj.vertices, cent=cent[0])
            pc_o = trimesh.Trimesh(vertices=clean_pc(pc_o.vertices, obj.vertices))

            # Load SMPL joints
            smpl = trimesh.load(join(path.replace(PROCESSED_PATH, BEHAVE_PATH), 'person/fit02/person_fit.ply'), process=False)
            smpl.vertices, _ = preprocess(smpl.vertices, cent=cent[0])
            regressor = np.load('assets/SMPL_joint_regressor.npy')
            J = np.matmul(regressor, np.array(smpl.vertices))

            pose = np.zeros((72,))
            pose[0] = np.pi
            betas = np.zeros((10,))
            trans = np.zeros((3,))

        return {'grid_coords': np.array(coords, dtype=np.float32),
                'cent': np.array(cent[0], dtype=np.float32),
                'pc_h': 0 if pc_h is None else np.array(
                    pc_h.vertices[np.random.choice(np.arange(len(pc_h.vertices)), 8000)].astype('float32')),
                'pc_o': 0 if pc_o is None else np.array(
                    pc_o.vertices[np.random.choice(np.arange(len(pc_o.vertices)), 8000)].astype('float32')),
                'smpl_J': 0 if J is None else J.astype('float32'),
                'df_h': np.array(df_h, dtype=np.float32),
                'df_o': np.array(df_o, dtype=np.float32),
                'points': np.array(points, dtype=np.float32),
                'parts': np.array(parts, dtype=np.float32),
                'corr': np.array(corr, dtype=np.float32).transpose(1, 0),
                'inputs': np.array(input_full, dtype=np.float32),
                'path': path,
                'pose': 0 if pose is None else pose.astype('float32'),
                'betas': 0 if betas is None else betas.astype('float32'),
                'trans': 0 if trans is None else trans.astype('float32'),
                'pca_axis': np.array(pca).astype('float32').transpose(1, 2, 0),
                'occ_h': np.array(occ_h).astype('float32')
                }

if __name__ == "__main__":
	args = lambda: None
	args.pc_samples = 5000
	args.res = 128
	args.sample_distribution = [0.5, 0.5]
	args.sample_sigmas = [0.15, 0.015]
	args.num_sample_points = 40000
	args.batch_size = 48
	args.suffix = '02'
	args.ext = '01'

	args.split_file = 'assets/data_split_kinect.pkl'
	train_dataset = DataLoader('train', pointcloud_samples=args.pc_samples, res=args.res,
	                              sample_distribution=args.sample_distribution,
	                              sample_sigmas=args.sample_sigmas, num_sample_points=args.num_sample_points,
	                              batch_size=args.batch_size, num_workers=48,
	                              suffix=args.suffix, ext=args.ext,
	                              split_file=args.split_file).get_loader(shuffle=False)

	for n, b in enumerate(train_dataset):
		pass
		# break

	# import ipdb; ipdb.set_trace()
	print('Done')