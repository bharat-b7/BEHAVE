"""
Code to normalise PCs.
Author: Bharat Lal Bhatnagar
Cite: BEHAVE: Dataset and Method for Tracking Human Object Interactions, CVPR'22
"""
bb_min = -1.
bb_max = 1.
new_cent = (bb_max + bb_min) / 2


def preprocess(vertices, cent=None, scale=1):
	"""
	Function to normalize the scans.
	"""

	if cent is None:
		cent = (vertices.max(axis=0) + vertices.min(axis=0)) / 2
	vertices -= (cent - new_cent)

	vertices /= scale

	return vertices, cent


def revert_preprocess(vertices, cent, scale=1.):
	vertices *= scale
	vertices += cent
	return vertices


if __name__ == "__main__":
	import numpy as np
	import trimesh

	mesh = trimesh.load(
		'/BS/bharat-4/static00/kinect_data/May06_xianghui_suitcase_pull_righthand/t0003.000/t0003.000.ply')
	preprocess(mesh.vertices)
