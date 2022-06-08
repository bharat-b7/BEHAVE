"""
Code to train and test BEHAVE network.
Author: Bharat Lal Bhatnagar
Cite: BEHAVE: Dataset and Method for Tracking Human Object Interactions, CVPR'22
"""

from models.network import HOTNet
from models.trainer import Trainer, Fitter
from data_loaders.dataloader import DataLoader
import argparse
import torch
import numpy as np

import os
from os.path import join, split, exists

def main(args):
	net = torch.nn.DataParallel(HOTNet(hidden_dim=args.decoder_hidden_dim, num_parts=14))

	exp_name = '{}_exp_id{}'.format(
		args.ext,
		args.exp_id
	)

	if args.mode == 'train':
		train_dataset = DataLoader('train', pointcloud_samples=args.pc_samples, res=args.res,
		                              sample_distribution=args.sample_distribution,
		                              sample_sigmas=args.sample_sigmas, num_sample_points=args.num_sample_points,
		                              batch_size=args.batch_size, num_workers=48,
		                              suffix=args.suffix, ext=args.ext, split_file=args.split_file)

		trainer = Trainer(net, torch.device("cuda"), train_dataset, None, exp_name,
		                     optimizer=args.optimizer)

		trainer.train_model(args.epochs)
	elif args.mode == 'val':
		test_dataset = DataLoader('val', pointcloud_samples=args.pc_samples, res=args.res,
		                             sample_distribution=args.sample_distribution,
		                             sample_sigmas=args.sample_sigmas, num_sample_points=args.num_sample_points,
		                             batch_size=args.batch_size, num_workers=30,
		                             suffix=args.suffix, ext=args.ext, split_file=args.split_file).get_loader(shuffle=False)

		trainer = Fitter(net, torch.device("cuda"), None, test_dataset, exp_name,
		                                    optimizer=args.optimizer, opt_dict={'cache_folder': args.cache_suffix,
		                                                                        'iter_per_step': {1: 201, 2: 1, 3: 1}},
		                                    checkpoint_number=args.checkpoint)

		trainer.fit_test_sample(save_name=args.save_name, num_saves=args.num_samples)

	else:
		print('invalid mode', args.mode)


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Run Model')
	# number of points in input in case of pointcloud input
	parser.add_argument('-pc_samples', default=5000, type=int)
	# number of points to predict as output
	parser.add_argument('-num_sample_points', default=20000, type=int)
	# distribution of samples used constructed via different standard devations
	parser.add_argument('-dist', '--sample_distribution', default=[0.5, 0.5], nargs='+', type=float)
	# the standard deviations from the surface used to compute inside/outside samples
	parser.add_argument('-std_dev', '--sample_sigmas', default=[0.15, 0.015], nargs='+', type=float)
	# defines how much input data is unsed as a batch.
	parser.add_argument('-batch_size', default=1, type=int)
	# the resolution of the input
	parser.add_argument('-res', default=128, type=int)
	# keep this fixed
	parser.add_argument('-h_dim', '--decoder_hidden_dim', default=256, type=int)
	# keep this fixed
	parser.add_argument('-o', '--optimizer', default='Adam', type=str)
	# data suffix
	parser.add_argument('-suffix', '--suffix', default='', type=str)
	# ext for data suffix
	parser.add_argument('-ext', '--ext', default='', type=str)
	# experiment id for folder suffix
	parser.add_argument('-exp_id', '--exp_id', default='', type=str)
	# Select singleView mode
	parser.add_argument('-split_file', '--split_file',
	                    default='/BS/bharat-4/work/HOTracking/assets/data_split_kinect_gt_obj_10.2.pkl', type=str)
	# Epochs
	parser.add_argument('-epochs', default=250, type=int)

	# modes
	parser.add_argument('-mode', default='train', choices=['train', 'val', 'eval'])
	# number of test samples
	parser.add_argument('-num_samples', default=-1, type=int)
	# number of points queried for to produce the result
	parser.add_argument('-retrieval_res', default=256, type=int)
	# which checkpoint of the experiment should be used?
	parser.add_argument('-checkpoint', default=None, type=int)
	# number of points from the querey grid which are put into the batch at once
	parser.add_argument('-batch_points', default=500000, type=int)

	args = parser.parse_args()

	main(args)