"""
Code to voxelize PC.
Author: Bharat Lal Bhatnagar
Cite: BEHAVE: Dataset and Method for Tracking Human Object Interactions, CVPR'22
"""

from scipy.spatial import cKDTree as KDTree
import numpy as np
import trimesh
import os
from os.path import join, split, exists
import argparse
from utils.preprocess_pointcloud import bb_max, bb_min, preprocess
from lib.torch_functions import np2tensor, tensor2np


def create_grid_points_from_bounds(minimun, maximum, res):
    x = np.linspace(minimun, maximum, res)
    X, Y, Z = np.meshgrid(x, x, x, indexing='ij')
    X = X.reshape((np.prod(X.shape),))
    Y = Y.reshape((np.prod(Y.shape),))
    Z = Z.reshape((np.prod(Z.shape),))

    points_list = np.column_stack((X, Y, Z))
    del X, Y, Z, x
    return points_list


def clean_pc(pc, fit, thresh=0.05):
    'Clean noisy pointclouds using GT mesh. Points further than thresh are removed.'
    from pytorch3d.ops import knn_points

    closest_dist_in_fit = knn_points(np2tensor(pc.astype('float32'))[None],
                                     np2tensor(fit.astype('float32'))[None], K=1)
    closest_dist_in_fit = closest_dist_in_fit.dists ** 0.5

    return pc[closest_dist_in_fit.squeeze() < thresh]

def voxelized_pointcloud(pc_h_, pc_o_, name, out_path, res, num_points, bounds=(-1, 1), ext=''):
    t1, t2 = split(pc_o_)
    if not exists(pc_h_) or not exists(pc_o_) or not exists(join(t1, 'fit01', t2.replace('.ply', '_fit.ply'))):
        print('human/object not found, ', pc_h_)
        return

    out_file = join(out_path, name + '_voxelized_point_cloud_res_{}_points_{}_{}.npz'.format(res, num_points, ext))
    if exists(out_file) and REDO == False:
        print('Already exists, ', split(out_file)[1])
        return

    pc_h = trimesh.load(pc_h_)
    pc_o = trimesh.load(pc_o_)
    pc = trimesh.PointCloud(vertices=list(pc_h.vertices) + list(pc_o.vertices))
    point_cloud, cent = preprocess(pc.vertices)
    pc_h.vertices, _ = preprocess(pc_h.vertices, cent=cent)
    pc_o.vertices, _ = preprocess(pc_o.vertices, cent=cent)

    # Load GT SMPL and object fit for cleaning
    # smpl = trimesh.load(pc_h_.replace('person.ply', 'fit02/person_fit.ply'), process=False)
    t1, t2 = split(pc_o_)
    obj = trimesh.load(join(t1, 'fit01', t2.replace('.ply', '_fit.ply')), process=False)
    # smpl.vertices, _ = preprocess(smpl.vertices, cent=cent)
    obj.vertices, _ = preprocess(obj.vertices, cent=cent)
    pc_o = trimesh.Trimesh(vertices=clean_pc(pc_o.vertices, obj.vertices))

    grid_points = create_grid_points_from_bounds(bounds[0], bounds[1], res)
    occupancies = np.zeros(len(grid_points), dtype=np.int8)
    segmentation = np.zeros(len(grid_points), dtype=np.int8)
    kdtree = KDTree(grid_points)
    # Occ wrt. human
    _, idx = kdtree.query(pc_h.vertices)
    occupancies[idx] = 1
    segmentation[idx] = 1
    # Occ. wrt. object
    _, idx = kdtree.query(pc_o.vertices)
    occupancies[idx] = 1
    ## human -> occ==1 && seg==1; object -> occ==1 && seg==0

    compressed_occupancies = np.packbits(occupancies)
    if not exists(out_path):
        os.makedirs(out_path)
    np.savez(out_file, point_cloud=point_cloud, compressed_occupancies=compressed_occupancies,
             compressed_segmentation=np.packbits(segmentation),
             bb_min=bounds[0], bb_max=bounds[1], res=res)
    print('Done, ', split(out_file)[1])


REDO = False
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Run voxelized sampling'
    )
    parser.add_argument('pc_h', type=str)
    parser.add_argument('pc_o', type=str)
    parser.add_argument('out_path', type=str)
    parser.add_argument('--ext', type=str, default='')
    parser.add_argument('--REDO', dest='REDO', action='store_true')
    parser.add_argument('--res', type=int, default=128)
    parser.add_argument('--num_points', type=int, default=5000)
    args = parser.parse_args()

    # args = lambda: None
    # pat = '/BS/xxie-4/work/kindata/Sep29_bharat_keyboard_move/t0004.000'
    # object_name = parse_object(split(split(pat)[0])[1])
    # args.pc_h = join(pat, 'person/person.ply')
    # args.pc_o = join(pat, object_name, '{}.ply'.format(object_name))
    # args.out_path = pat
    # args.res = 40
    # args.num_points = 5000
    # args.ext = ''
    # args.REDO=True

    REDO = args.REDO
    # name = split(args.pc_h)[1][:-4]
    name = split(split(split(args.pc_h)[0])[0])[1]
    voxelized_pointcloud(args.pc_h, args.pc_o, name, args.out_path, args.res, args.num_points, bounds=(bb_min, bb_max),
                         ext=args.ext)