"""
BEHAVE network.
Author: Bharat Lal Bhatnagar
Cite: BEHAVE: Dataset and Method for Tracking Human Object Interactions, CVPR'22
"""

import torch
from torch import nn
from torch.nn import functional as F


class HOTNet(nn.Module):
	def __init__(self, hidden_dim=256, num_parts=24, tex=False):
		super(HOTNet, self).__init__()

		self.num_parts = num_parts
		self.tex = tex

		self.make_encoder()

		if tex:
			feature_size = (3 + 32 + 64 + 64 + 128 + 128) * 7
		else:
			feature_size = (1 + 32 + 64 + 64 + 128 + 128) * 7

		# part predictor
		self.part_predictor = self.make_decoder(feature_size, num_parts, 1, hidden_dim)
		self.fc_parts_softmax = nn.Softmax(1)

		# Human + Object DF predictor
		self.df = self.make_decoder(feature_size, 2, 1, hidden_dim)

		# per-part correspondence predictor
		self.corr_predictor = self.make_decoder(feature_size, num_parts * 3, num_parts, hidden_dim)

		# object pca_axis predictor
		self.pca_predictor = self.make_decoder(feature_size, 9, 1, hidden_dim)

		self.actvn = nn.ReLU()
		self.displacments = self.make_displacements()

	def make_encoder(self):
		if self.tex:
			self.conv_00 = nn.Conv3d(3, 32, 3, padding=1)  # out: 128
		else:
			self.conv_00 = nn.Conv3d(1, 32, 3, padding=1)  # out: 128
		self.conv_01 = nn.Conv3d(32, 32, 3, padding=1)  # out: 128
		self.bn_01 = torch.nn.BatchNorm3d(32)

		self.conv_10 = nn.Conv3d(32, 64, 3, padding=1)  # out: 128
		self.conv_11 = nn.Conv3d(64, 64, 3, padding=1, stride=2)  # out: 64
		self.bn_11 = torch.nn.BatchNorm3d(64)

		self.conv_20 = nn.Conv3d(64, 64, 3, padding=1)  # out: 64
		self.conv_21 = nn.Conv3d(64, 64, 3, padding=1, stride=2)  # out: 32
		self.bn_21 = torch.nn.BatchNorm3d(64)

		self.conv_30 = nn.Conv3d(64, 128, 3, padding=1)  # out: 32
		self.conv_31 = nn.Conv3d(128, 128, 3, padding=1, stride=2)  # out: 16
		self.bn_31 = torch.nn.BatchNorm3d(128)

		self.conv_40 = nn.Conv3d(128, 128, 3, padding=1)  # out: 16
		self.conv_41 = nn.Conv3d(128, 128, 3, padding=1, stride=2)  # out: 8
		self.bn_41 = torch.nn.BatchNorm3d(128)

	def make_decoder(self, input_sz, output_sz, group_sz, hidden_sz):
		# per-part occupancy predictor
		predictor = [
			nn.Conv1d(input_sz, hidden_sz * group_sz, 1),
			nn.ReLU(),
			nn.Conv1d(hidden_sz * group_sz, hidden_sz * group_sz, 1, groups=group_sz),
			nn.ReLU(),
			nn.Conv1d(hidden_sz * group_sz, hidden_sz * group_sz, 1, groups=group_sz),
			nn.ReLU(),
			nn.Conv1d(hidden_sz * group_sz, output_sz, 1, groups=group_sz)
		]
		return nn.Sequential(*predictor)

	@staticmethod
	def make_displacements():
		displacment = 0.0722
		displacments = [[0, 0, 0]]
		for x in range(3):
			for y in [-1, 1]:
				input = [0, 0, 0]
				input[x] = y * displacment
				displacments.append(input)

		return torch.nn.Parameter(torch.tensor(displacments), requires_grad=False)

	def encode(self, p, x):
		if not self.tex:
			x = x.unsqueeze(1)
		p = p.unsqueeze(1).unsqueeze(1)
		p = torch.cat([p + d for d in self.displacments], dim=2)
		full_0 = F.grid_sample(x, p, align_corners=True)

		net = self.actvn(self.conv_00(x))
		net = self.actvn(self.conv_01(net))
		net = self.bn_01(net)
		full_1 = F.grid_sample(net, p, align_corners=True)
		# ipdb.set_trace()

		net = self.actvn(self.conv_10(net))
		net = self.actvn(self.conv_11(net))
		net = self.bn_11(net)
		full_2 = F.grid_sample(net, p, align_corners=True)

		net = self.actvn(self.conv_20(net))
		net = self.actvn(self.conv_21(net))
		net = self.bn_21(net)
		full_3 = F.grid_sample(net, p, align_corners=True)

		net = self.actvn(self.conv_30(net))
		net = self.actvn(self.conv_31(net))
		net = self.bn_31(net)
		full_4 = F.grid_sample(net, p, align_corners=True)
		# ipdb.set_trace()

		net = self.actvn(self.conv_40(net))
		net = self.actvn(self.conv_41(net))
		net = self.bn_41(net)
		full_5 = F.grid_sample(net, p, align_corners=True)

		full = torch.cat((full_0, full_1, full_2, full_3, full_4, full_5), dim=1)
		shape = full.shape
		full = torch.reshape(full, (shape[0], shape[1] * shape[3], shape[4]))

		return full

	def forward(self, p, x):
		batch_sz = x.shape[0]
		full = self.encode(p, x)  # feature grid

		out_parts = self.part_predictor(full)
		parts_softmax = self.fc_parts_softmax(out_parts)  # part probability

		out_df = self.df(full)
		out_pca = self.pca_predictor(full)

		corr = self.corr_predictor(full).view(batch_sz, 3, self.num_parts, -1)
		corr *= parts_softmax.view(batch_sz, 1, self.num_parts, -1)
		out_corr = corr.mean(2)

		out = {'df_h': out_df[:, 0], 'df_o': out_df[:, 1], 'parts': out_parts,
		       'corr': out_corr, 'pca_axis': out_pca.view(batch_sz, 3, 3, -1)}
		return out
