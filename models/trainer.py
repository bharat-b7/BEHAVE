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
import time

from models.volumetric_SMPL import VolumetricSMPL
from lib.smpl_paths import SmplPaths
from lib.th_smpl_prior import get_prior
from lib.smpl_layer import SMPL_Layer
import yaml
with open("PATHS.yml", 'r') as stream:
    paths = yaml.safe_load(stream)
PROCESSED_PATH, BEHAVE_PATH, OBJECT_PATH = paths['PROCESSED_PATH'], paths['BEHAVE_PATH'], paths['OBJECT_TEMPLATE']

NUM_POINTS = 30000
DEBUG = False


class Trainer(object):
    def __init__(self, model, device, train_dataset, val_dataset, exp_name, optimizer='Adam'):
        self.model = model.to(device)
        self.device = device
        self.optimizer_type = optimizer
        self.optimizer = self.init_optimizer(optimizer, self.model.parameters(), learning_rate=0.001)

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
    def init_optimizer(optimizer, params, learning_rate=1e-4):
        if optimizer == 'Adam':
            optimizer = optim.Adam(params, lr=learning_rate, betas=(0.9, 0.999))
        if optimizer == 'Adadelta':
            optimizer = optim.Adadelta(params)
        if optimizer == 'RMSprop':
            optimizer = optim.RMSprop(params, momentum=0.9)
        return optimizer

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


class Fitter(object):
    def __init__(self, model, device, train_dataset, val_dataset, exp_name, opt_dict={},
                 checkpoint_number=-1):
        self.model = model.to(device)
        self.device = device
        self.opt_dict = self.parse_opt_dict(opt_dict)

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.checkpoint_number = checkpoint_number
        self.exp_path = os.path.dirname(__file__) + '/../experiments/{}/'.format(exp_name)
        self.checkpoint_path = self.exp_path + 'checkpoints/'.format(exp_name)
        if not os.path.exists(self.checkpoint_path):
            print(self.checkpoint_path)
            os.makedirs(self.checkpoint_path)
        self.writer = SummaryWriter(self.exp_path + 'summary'.format(exp_name))
        self.val_min = None

        # Load vsmpl
        self.vsmpl = VolumetricSMPL('assets/volumetric_smpl_function_64', device, 'male')
        self.smpl_layer = SMPL_Layer(center_idx=0, gender='male', num_betas=10,
                                     model_root='smplpytorch/smplpytorch/native/models').to(
            device)

        sp = SmplPaths(gender='male')
        self.ref_smpl = sp.get_smpl()
        self.template_points = torch.tensor(
            trimesh.Trimesh(vertices=self.ref_smpl.r, faces=self.ref_smpl.f).sample(NUM_POINTS).astype('float32'),
            requires_grad=False).unsqueeze(0)

        self.pose_prior = get_prior('male', precomputed=True)

        self.object_mesh = {}
        for i in glob(join(OBJECT_PATH, '*')):
            nam = split(i)[1]
            self.object_mesh[nam] = trimesh.load_mesh(join(i, nam + '.obj') , process=False)

        # normalise the objects to zero mean
        self.objects, self.pca_axis = {}, {}
        pca = PCA(n_components=3)
        for i in self.object_mesh:
            self.object_mesh[i].vertices -= self.object_mesh[i].vertices.mean(0)

            pca.fit(self.object_mesh[i].vertices)
            self.pca_axis[i] = torch.tensor(pca.components_, device=self.device, requires_grad=False,
                                            dtype=torch.float32)

            x = self.object_mesh[i].sample(3000)
            self.objects[i] = torch.tensor(x, device=self.device, requires_grad=False, dtype=torch.float32)

    @staticmethod
    def parse_opt_dict(opt_dict):
        # timestamp = int(time.time())
        parsed_dict = {'iter_per_step': {1: 200, 2: 200, 3: 1},
                       'epochs_phase_01': 0, 'epochs_phase_02': 0}
        """ 
        Phase_01: Initialised SMPL are far off from the solution. Optimize SMPL based on correspondences.
        Phase_02: SMPL models are close to solution. Fit SMPL based on ICP.
        Phase_03: Jointly update SMPL and correspondences.
        """
        for k in parsed_dict:
            if k in opt_dict:
                parsed_dict[k] = opt_dict[k]
        # print('Cache folder: ', parsed_dict['cache_folder'])
        return parsed_dict

    @staticmethod
    def get_optimization_weights(phase):
        """
        Phase_01: Initialised SMPL are far off from the solution. Optimize SMPL based on correspondences.
        Phase_02: SMPL models are close to solution. Fit SMPL based on ICP.
        Phase_03: Jointly update SMPL and correspondences.
        """
        if phase == 1:
            return {'corr': 10. ** 1, 'templ': 2 * 10. ** 2, 's2m': 10. ** 2, 'm2s': 10. ** 1, 'pose_pr': 10. ** -2,
                    'shape_pr': 10. ** 0, 'object': 1., 'smpl_J': 10. ** 1, 'pca_axis': 10. ** 3, 's2m_o': 2 * 10. ** 1,
                    'contacts': 10. ** 0}
        elif phase == 2:
            return {'corr': 10. ** -1, 'templ': 2 * 10. ** 2, 's2m': 2 * 10. ** 3, 'm2s': 10. ** 3,
                    'pose_pr': 10. ** -4, 'shape_pr': 10. ** 0, 'object': 1., 'smpl_J': 10. ** 1, 'pca_axis': 10. ** 3,
                    's2m_o': 10. ** 2, 'contacts': 10. ** 1}
        else:
            return {'corr': 2 * 10. ** 2, 'templ': 2 * 10. ** 2, 's2m': 10. ** 4, 'm2s': 10. ** 4, 'pose_pr': 10. ** -4,
                    'shape_pr': 10. ** 0, 'object': 1., 'smpl_J': 10. ** 0, 'pca_axis': 10. ** 3, 's2m_o': 10. ** 2,
                    'contacts': 10. ** 1}

    @staticmethod
    def init_object_orientation(src_axis, tgt_axis):
        # pseudo inverse
        assert len(src_axis.shape) == 3
        tr = torch.bmm(src_axis.transpose(2, 1), src_axis)
        tr_inv = torch.inverse(tr)
        pseudo = torch.bmm(tr_inv, src_axis.transpose(2, 1))

        rot = torch.bmm(pseudo, tgt_axis)  # + 1e-4 * torch.rand(pseudo.shape[0], 3, 3).to(pseudo.device)
        # return rot
        U, S, V = torch.svd(rot)
        R = torch.bmm(U, V.transpose(2, 1))
        return R

    @staticmethod
    def sum_dict(los):
        temp = 0
        for l in los:
            temp += los[l]
        return temp

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
        epoch = checkpoint['epoch']
        return epoch

    def get_object_class(self, paths):
        lis = []
        for path in paths:
            fl = -1
            for k in self.objects:
                if k in path:
                    lis.append(k)
                    fl = 1
                    break
            assert fl == 1

        return lis

    def save_output(self, names, pose_, betas_, trans_, corr_, obj_R, obj_t, cent, object_class, save_name, epoch, it):
        from lib.smpl_paths import SmplPaths

        sp = SmplPaths(gender='male')
        smpl = sp.get_smpl()
        for nam, p, b, t, c, ro, to, co, ce in zip(names, pose_, betas_, trans_, corr_, obj_R, obj_t, object_class,
                                                   cent):
            name = split(nam)[1]
            dataset = split(split(nam)[0])[1]
            outfile = join(self.exp_path, save_name + '_ep_{}'.format(epoch), dataset)
            if not exists(outfile):
                os.makedirs(outfile)

            smpl.pose[:] = p
            smpl.betas[:10] = b
            smpl.trans[:] = t + ce

            # save registration
            trimesh.Trimesh(vertices=smpl.r, faces=smpl.f).export(
                join(outfile, name + '_{}_reg.ply'.format(it)))

            trimesh.Trimesh(vertices=np.matmul(self.object_mesh[co].vertices, ro) + to + ce,
                            faces=self.object_mesh[co].faces).export(
                join(outfile, name + '_{}_obj.ply'.format(it)))

            # save SMPL params
            with open(join(outfile, name + '_{}_reg.pkl'.format(it)), 'wb') as f:
                pkl.dump({'pose': p, 'betas': b, 'trans': t + ce}, f)
            print('Saved,', join(outfile, name + '_{}_reg.pkl'.format(it)))

    def fit_test_sample(self, save_name, num_saves=None):
        device = self.device

        epoch = self.load_checkpoint(number=self.checkpoint_number)
        print('Testing with epoch {}'.format(epoch))
        if not exists(join(self.exp_path, save_name + '_ep_{}'.format(epoch))):
            os.makedirs(join(self.exp_path, save_name + '_ep_{}'.format(epoch)))
        # self.model.train(False)

        count = 0
        for batch in self.val_dataset:
            names = batch.get('path')
            p = batch.get('grid_coords').to(device)
            inputs = batch.get('inputs').to(device)
            pose = batch.get('pose').to(device).requires_grad_(True)
            betas = batch.get('betas').to(device).requires_grad_(True)
            trans = batch.get('trans').to(device).requires_grad_(True)
            paths = batch.get('path')

            obj_t = torch.zeros((3,), dtype=torch.float, device=device)[None].repeat(trans.shape[0], 1).requires_grad_(
                True)

            # load object
            object_class = self.get_object_class(paths)
            object_init = torch.cat([self.objects[x][None] for x in object_class], dim=0)
            pca_axis_init = torch.cat([self.pca_axis[x][None] for x in object_class], dim=0)

            # predict initial correspondences and SDF
            out = self.model(p, inputs)
            corr_init = out['corr'].permute(0, 2, 1).detach()
            _, part_label = torch.max(out['parts'].data, 1)
            # import ipdb; ipdb.set_trace()
            corr = corr_init.clone().requires_grad_(True)
            df_h = out['df_h'].detach()
            df_o = out['df_o'].detach()

            # Select points for object surface
            mask = (df_o < 0.02).unsqueeze(1).unsqueeze(1)
            pca_axis = (out['pca_axis'] * mask).sum(axis=-1) / mask.sum(axis=-1)  # predicted PCA axis
            r = self.init_object_orientation(pca_axis_init, pca_axis)
            r += 1e-5 * torch.rand(r.shape[0], 3, 3).to(device)
            obj_R = torch.tensor(r, device=device, requires_grad=True)

            instance_params = {'pose': pose, 'betas': betas, 'trans': trans}

            # initialize optimizer for instance specific SMPL params, object params
            obj_optimizer = optim.Adam([obj_t, obj_R], lr=0.005)
            corr_optimizer = optim.Adam([corr], lr=0.02)
            smpl_optimizer = optim.Adam(instance_params.values(), lr=0.02)
            instance_params['corr'] = corr
            instance_params['df_h'] = df_h
            instance_params['obj_R'] = obj_R  # + 1e-4 * torch.rand(obj_R.shape[0], 3, 3).cuda()
            instance_params['obj_t'] = obj_t
            instance_params['object_init'] = object_init

            for it in range(self.opt_dict['iter_per_step'][1] + self.opt_dict['iter_per_step'][2]):
                smpl_optimizer.zero_grad()
                corr_optimizer.zero_grad()
                obj_optimizer.zero_grad()
                # import ipdb; ipdb.set_trace()
                if it == 0:
                    phase = 1
                    wts = self.get_optimization_weights(phase=1)
                    print('Optimizing phase 1')
                elif it == self.opt_dict['iter_per_step'][1]:
                    # import ipdb; ipdb.set_trace()
                    phase = 2
                    wts = self.get_optimization_weights(phase=2)
                    print('Optimizing phase 2')

                loss_ = self.iteration_step(batch, instance_params, weight_dict=wts)
                loss = self.sum_dict(loss_)

                # back propagate
                loss.backward()

                if phase == 1:
                    smpl_optimizer.step()
                    obj_optimizer.step()
                elif phase == 2:
                    smpl_optimizer.step()
                    corr_optimizer.step()
                    obj_optimizer.step()

                if it % 100 == 99:
                    l_str = 'Iter: {}'.format(it)
                    for l in loss_:
                        l_str += ', {}: {:0.5f}'.format(l, loss_[l].item())
                    print(l_str)

                if it % 300 == 299 or it == self.opt_dict['iter_per_step'][1] + self.opt_dict['iter_per_step'][2] - 1:
                    pose_ = pose.detach().cpu().numpy()
                    betas_ = betas.detach().cpu().numpy()
                    trans_ = trans.detach().cpu().numpy()
                    corr_ = corr.detach().cpu().numpy()

                    U, S, V = torch.svd(obj_R)
                    R = torch.bmm(U, V.transpose(2, 1))
                    R_ = R.detach().cpu().numpy()
                    obj_t_ = obj_t.detach().cpu().numpy()

                    self.save_output(names, pose_, betas_, trans_, corr_, R_, obj_t_, batch['cent'].numpy(),
                                     object_class, save_name, epoch,
                                     it)

            count += len(names)

            if (num_saves is not None) and (count >= num_saves):
                break

    def iteration_step(self, batch, instance_params, weight_dict={}):
        """
        Computes losses for a single step of optimization.
        Entries in loss/weight dict should have the following entries (always edit loss_keys to modify loss terms):
        corr, templ, s2m, m2s, pose_pr, shape_pr
        """
        loss_keys = ['corr', 'templ', 's2m_h', 's2m_o', 'm2s', 'pose_pr', 'shape_pr', 'smpl_J', 'pca_axis']
        for k in loss_keys:
            if k not in weight_dict.keys():
                weight_dict[k] = 1.

        device = self.device
        loss = {}
        inputs = batch.get('inputs').to(device)
        p = batch.get('grid_coords').to(device)
        points = batch.get('points').to(device)
        joints = batch.get('smpl_J').to(device)
        pc_h = batch.get('pc_h').to(device)
        pc_o = batch.get('pc_o').to(device)
        batch_sz = inputs.shape[0]

        poses, betas, trans = instance_params['pose'], instance_params['betas'], instance_params['trans']
        corr, df_h = instance_params['corr'], instance_params['df_h']
        # import ipdb; ipdb.set_trace()
        ## FIT OBJECT
        object_init = instance_params['object_init']
        rot = instance_params['obj_R']

        U, S, V = torch.svd(rot + 1e-3 * torch.rand(batch_sz, 3, 3).to(device))
        R = torch.bmm(U, V.transpose(2, 1))
        object = torch.bmm(object_init, R) + instance_params['obj_t'].unsqueeze(1)

        # evaluate object DF at object points
        out_ = self.model(object, inputs)
        loss['object'] = out_['df_o'].mean() * weight_dict['object']

        # S2M on human and object
        loss['s2m_o'], _ = chamfer_distance(pc_o, object)
        loss['s2m_o'] *= weight_dict['s2m_o']

        posed_scan_correspondences = self.vsmpl(corr, poses, betas, trans)

        # SMPL joints
        _, smpl_J, _, _ = self.smpl_layer(poses, betas, trans)
        # loss on 3D joints lifted from the images.
        loss['smpl_J'] = F.l1_loss(smpl_J, joints) * weight_dict['smpl_J']

        # correspondence loss
        loss['corr'] = F.l1_loss(torch.linalg.norm(points - posed_scan_correspondences, dim=-1), df_h,
                                 reduction='none').mean() * weight_dict['corr']

        # pose prior
        loss['pose_pr'] = self.pose_prior(poses).mean() * weight_dict['pose_pr']

        # shape prior
        loss['shape_pr'] = (betas ** 2).mean() * weight_dict['shape_pr']

        return loss
