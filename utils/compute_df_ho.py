"""
Compute DT, correspondences for the human and the object. This is used to supervise the BEHAVE model.
Author: Bharat Lal Bhatnagar
Cite: BEHAVE: Dataset and Method for Tracking Human Object Interactions, CVPR'22
"""

import sys
sys.path.append('..')
import yaml
with open("PATHS.yml", 'r') as stream:
    paths = yaml.safe_load(stream)
sys.path.append(paths['CODE'])
import numpy as np
import glob
import os
from os.path import join, split, exists
import argparse
import pickle as pkl
import trimesh
from sklearn.decomposition import PCA
from utils.preprocess_pointcloud import preprocess, bb_max, bb_min
from utils.voxelize_ho import create_grid_points_from_bounds, clean_pc


def parse_object(name):
    temp = name.split('_')[2]
    if temp == 'chairwood' or temp == 'chairblack':
        return 'chair'
    elif temp == 'basketball' or temp == 'yogaball':
        return 'sports ball'
    else:
        return temp


def boundary_sampling(pc_human, pc_obj, smpl_path, obj_path, name, out_path, sigma=0.05, sample_num=100000,
                      bounds=(-1., 1.), ext=''):
	"""
	center SMPL according to the original PC, sample points on SMPL, compute DT.
	"""

	out_file = join(out_path, name + '_sigma_{}_{}.npz'.format(sigma, ext))

	if exists(out_file) and REDO == False:
		print('File already exists, ', out_file)
		return out_file

	if not exists(smpl_path):
		print('Mesh not found, ', smpl_path)
		return False

	pc_h = trimesh.load(pc_human, process=False)
	pc_o = trimesh.load(pc_obj, process=False)
	pc = trimesh.PointCloud(vertices=list(pc_h.vertices) + list(pc_o.vertices))
	_, cent = preprocess(pc.vertices)
	pc_h.vertices, _ = preprocess(pc_h.vertices, cent=cent)
	pc_o.vertices, _ = preprocess(pc_o.vertices, cent=cent)

	smpl = trimesh.load(smpl_path, process=False)
	obj = trimesh.load(obj_path, process=False)
	smpl.vertices, _ = preprocess(smpl.vertices, cent=cent)
	obj.vertices, _ = preprocess(obj.vertices, cent=cent)

	# pc_o = trimesh.Trimesh(vertices=clean_pc(pc_o.vertices, obj.vertices))

	# Get object PCA
	pca = PCA(n_components=3)
	pca.fit(obj.vertices)
	pca_axis = pca.components_

	comb = trimesh.util.concatenate([smpl, obj])

	points_full = comb.sample(sample_num)

	boundary_points = points_full + sigma * np.random.randn(sample_num, 3)

	# coordinates transformation for torch.nn.functional.grid_sample grid interpolation method
	# for indexing of grid_sample function: swaps x and z coordinates
	grid_coords = boundary_points.copy()
	grid_coords[:, 0], grid_coords[:, 2] = grid_coords[:, 2], grid_coords[:, 0].copy()

	ce, sc = bounds[0] + bounds[1], bounds[1] - bounds[0]
	grid_coords = 2 * grid_coords - ce
	grid_coords = grid_coords / sc

	## Also add uniform points
	# n_samps = 20
	# uniform_points = np.array(create_grid_points_from_bounds(bounds[0], bounds[1], n_samps))
	# uniform_points_scaled = (2. * uniform_points.copy() - ce) / sc
	# uniform_points_scaled[:, 0], uniform_points_scaled[:, 2] = uniform_points_scaled[:, 2], \
	#                                                            uniform_points_scaled[:, 0].copy()

	# grid_coords = np.append(grid_coords, uniform_points_scaled, axis=0)
	# boundary_points = np.append(boundary_points, uniform_points, axis=0)

	# Get distance from smpl
	temp_h = trimesh.proximity.ProximityQuery(smpl)
	_, d_h, _ = temp_h.on_surface(boundary_points)
	occ_h = temp_h.signed_distance(boundary_points) < 0

	# Get distance from object
	temp_o = trimesh.proximity.ProximityQuery(obj)
	_, d_o, _ = temp_o.on_surface(boundary_points)

	# Get correspondences and part labels
	# Parts
	part_labels = pkl.load(open('assets/smpl_parts_dense.pkl', 'rb'))
	labels = np.zeros((6890,), dtype='int32')
	for n, k in enumerate(part_labels):
		labels[part_labels[k]] = n

	_, vert_ids = temp_h.vertex(boundary_points)
	jidx = labels[vert_ids]

	# Correspondences
	closest_point, _, tri_id = trimesh.proximity.closest_point(smpl, boundary_points)
	bary = trimesh.triangles.points_to_barycentric(smpl.triangles[tri_id], closest_point)
	ref = trimesh.load_mesh('assets/ref_mesh.obj', process=False)  # reference mesh
	correspondences = trimesh.triangles.barycentric_to_points(ref.triangles[tri_id], bary)

	if not exists(out_path):
		os.makedirs(out_path)

	np.savez(out_file, points=boundary_points, dist_h=d_h, dist_o=d_o, grid_coords=grid_coords, cent=cent, parts=jidx,
	         correspondences=correspondences, pca_axis=pca_axis, occ_h=occ_h)
	print('Done part labels and correspondences, ', out_file)

	return out_file


REDO = False
if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Run boundary sampling')
	parser.add_argument('pc_h', type=str)
	parser.add_argument('pc_o', type=str)
	parser.add_argument('smpl_path', type=str)
	parser.add_argument('obj_path', type=str)
	parser.add_argument('out_path', type=str)
	parser.add_argument('--ext_out', type=str, default='')
	parser.add_argument(
		'--REDO', dest='REDO', action='store_true')
	parser.add_argument('--sigma', default=0.15, type=float)
	parser.add_argument('--sample_num', default=100000, type=np.int32)
	parser.set_defaults(parts=False)
	parser.add_argument('--parts', dest='parts', action='store_true')
	args = parser.parse_args()

	# from generate_scripts.voxelize_pointcloud import parse_object
	# args = lambda: None
	# pat = '/BS/xxie-4/work/kindata/Sep29_bharat_chairwood_sit/t0003.000'
	# object_name = parse_object(split(split(pat)[0])[1])
	# args.pc_h = join(pat, 'person/person.ply')
	# args.pc_o = join(pat, object_name, '{}.ply'.format(object_name))
	# args.smpl_path = join(pat, 'person/fit02/person_fit.ply')
	# args.obj_path = join(pat, object_name, 'fit01/{}_fit.ply'.format(object_name))
	# args.out_path = join('/BS/bharat-5/static00/HOTracking_data', split(pat)[1], 'boundary_sampling')
	# args.sigma = 0.15
	# args.sample_num = 100000
	# args.ext_out = '02'
	# args.REDO = True

	REDO = args.REDO
	# name = split(args.pc_h)[1][:-4]
	name = split(split(split(args.pc_h)[0])[0])[1]
	if not exists(args.out_path):
		os.makedirs(args.out_path)

	out_file = boundary_sampling(args.pc_h, args.pc_o, args.smpl_path, args.obj_path, name, args.out_path,
	                             sigma=args.sigma,
	                             sample_num=args.sample_num,
	                             bounds=(bb_min, bb_max), ext=args.ext_out)

	# add_pca(out_file, args.obj_path)
