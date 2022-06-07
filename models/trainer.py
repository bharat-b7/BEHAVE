"""
Code to train the BEHAVE network.
Author: Bharat Lal Bhatnagar
Cite: BEHAVE: Dataset and Method for Tracking Human Object Interactions, CVPR'22
"""

from __future__ import division
import torch
import torch.optim as optim
from torch.nn import functional as F
import os
from os.path import join, split, exists
from torch.utils.tensorboard import SummaryWriter
from torch import pca_lowrank
from pytorch3d.ops import knn_points
from pytorch3d.loss import chamfer_distance
from glob import glob
import numpy as np
from collections import Counter
import trimesh
import pickle as pkl
from psbody.mesh import Mesh, MeshViewer
import pickle as pkl
from sklearn.decomposition import PCA

from models.volumetric_SMPL import VolumetricSMPL
from lib.smpl_paths import SmplPaths
from lib.th_smpl_prior import get_prior
from lib.smpl_layer import SMPL_Layer

NUM_POINTS = 30000
DEBUG = False


class Trainer(object):
	def __init__(self, model, device, train_dataset, val_dataset, exp_name, optimizer='Adam'):
		self.model = model.to(device)
		self.device = device
		if optimizer == 'Adam':
			self.optimizer = optim.Adam(self.model.parameters(), lr=1e-4)
		if optimizer == 'Adadelta':
			self.optimizer = optim.Adadelta(self.model.parameters())
		if optimizer == 'RMSprop':
			self.optimizer = optim.RMSprop(self.model.parameters(), momentum=0.9)

		self.train_dataset = train_dataset
		self.val_dataset = val_dataset
		self.exp_path = os.path.dirname(__file__) + '/../experiments/{}/'.format(exp_name)
		self.checkpoint_path = self.exp_path + 'checkpoints/'.format(exp_name)
		if not os.path.exists(self.checkpoint_path):
			print(self.checkpoint_path)
			os.makedirs(self.checkpoint_path)
		self.writer = SummaryWriter(self.exp_path + 'summary'.format(exp_name))
		self.val_min = None

	@staticmethod
	def sum_dict(los):
		temp = 0
		for l in los:
			temp += los[l]
		return temp

	def save_checkpoint(self, epoch):
		path = self.checkpoint_path + 'checkpoint_epoch_{}.tar'.format(epoch)
		if not os.path.exists(path):
			torch.save({'epoch': epoch, 'model_state_dict': self.model.state_dict(),
			            'optimizer_state_dict': self.optimizer.state_dict()}, path)

	def load_checkpoint(self, number=None):
		checkpoints = glob(self.checkpoint_path + '/*')
		if len(checkpoints) == 0:
			print('No checkpoints found at {}'.format(self.checkpoint_path))
			return 0

		if number is None:
			checkpoints = [os.path.splitext(os.path.basename(path))[0][17:] for path in checkpoints]
			checkpoints = np.array(checkpoints, dtype=int)
			checkpoints = np.sort(checkpoints)

			if checkpoints[-1] == 0:
				print('Not loading model as this is the first epoch')
				return 0

			path = join(self.checkpoint_path, 'checkpoint_epoch_{}.tar'.format(checkpoints[-1]))
		else:
			path = join(self.checkpoint_path, 'checkpoint_epoch_{}.tar'.format(number))

		print('Loaded checkpoint from: {}'.format(path))
		checkpoint = torch.load(path)
		self.model.load_state_dict(checkpoint['model_state_dict'])
		self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
		epoch = checkpoint['epoch']
		return epoch

	def train_step(self, batch):
		self.model.train()
		self.optimizer.zero_grad()
		loss_ = self.compute_loss(batch)
		loss = self.sum_dict(loss_)
		# import ipdb; ipdb.set_trace()
		loss.backward()
		self.optimizer.step()
		# import ipdb; ipdb.set_trace()
		return {k: loss_[k].item() for k in loss_}

	def train_model(self, epochs):
		start = self.load_checkpoint()

		for epoch in range(start, epochs):
			print('Start epoch {}'.format(epoch))
			train_data_loader = self.train_dataset.get_loader()

			if epoch % 1 == 0:
				self.save_checkpoint(epoch)
			sum_loss = None
			for n, batch in enumerate(train_data_loader):
				loss = self.train_step(batch)
				if sum_loss is None:
					sum_loss = Counter(loss)
				else:
					sum_loss.update(Counter(loss))

			loss_str = ''
			# import ipdb; ipdb.set_trace()
			for l in sum_loss:
				# self.writer.add_scalar(l, loss[l], epoch)
				self.writer.add_scalar(l, sum_loss[l] / len(train_data_loader), epoch)
				loss_str += '{}: {}, '.format(l, sum_loss[l] / len(train_data_loader))
			print(loss_str)

	def compute_loss(self, batch):
		device = self.device

		p = batch.get('grid_coords').to(device)
		df_h = batch.get('df_h').to(device)
		df_o = batch.get('df_o').to(device)
		inputs = batch.get('inputs').to(device)
		parts = batch.get('parts').to(device)
		corr = batch.get('corr').to(device)
		pca_axis = batch.get('pca_axis').to(device)
		occ = {'df_h': df_h, 'df_o': df_o, 'parts': parts, 'corr': corr, 'pca_axis': pca_axis}

		# Surface, Parts, Correspondences
		logits = self.model(p, inputs)
		loss = {}
		for i in occ:
			if 'parts' in i:
				# import ipdb; ipdb.set_trace()
				loss_i = F.cross_entropy(logits[i], occ[i].long(), reduction='none') * 0.1
				loss[i] = loss_i.sum(-1).mean()
			elif 'pca_axis' in i:
				mask = (occ['df_o'] < 0.05).unsqueeze(1).unsqueeze(1)
				# import ipdb; ipdb.set_trace()
				loss_i = (F.mse_loss(logits[i], occ[i], reduction='none') * mask)
				loss_i = loss_i.sum(axis=-1) / mask.sum(axis=-1) * 10. ** 3
				loss[i] = loss_i.mean()
			elif 'corr' in i:
				loss_i = F.mse_loss(logits[i], occ[i], reduction='none') * 10.
				loss[i] = loss_i.sum(-1).mean()
			else:
				# import ipdb; ipdb.set_trace()
				loss_i = F.mse_loss(logits[i], occ[i], reduction='none')
				loss[i] = loss_i.sum(-1).mean()

		# import ipdb; ipdb.set_trace()
		return loss