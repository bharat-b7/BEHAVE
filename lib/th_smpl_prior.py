"""
If code works:
    Author: Bharat
else:
    Author: Anonymous
"""
import numpy as np
import torch
import ipdb
import os
import sys
sys.path.append(os.getcwd())
import pickle as pkl
from PATHS import SMPL_MODEL_ROOT, SMPL_ASSETS_ROOT


def get_prior(gender='male'):
    prior = Prior(gender)
    return prior['Generic']


class th_Mahalanobis(object):
    def __init__(self, mean, prec, prefix):
        self.mean = torch.tensor(mean.astype('float32'), requires_grad=False).unsqueeze(axis=0).cuda()
        self.prec = torch.tensor(prec.astype('float32'), requires_grad=False).cuda()
        self.prefix = prefix

    def __call__(self, pose, prior_weight=1.):
        '''
        :param pose: Batch x pose_dims
        :return: weighted L2 distance of the N pose parameters, where N = 72 - prefix for SMPL model
        '''
        # return (pose[:, self.prefix:] - self.mean)*self.prec
        temp = pose[:, self.prefix:] - self.mean
        temp2 = torch.matmul(temp, self.prec) * prior_weight
        return (temp2 * temp2).sum(dim=1)
        

class Prior(object):
    def __init__(self, gender, prefix=3):
        self.prefix = prefix
        dat = pkl.load(open(os.path.join(SMPL_ASSETS_ROOT, f'prior_{gender}.pkl'), 'rb'))
        self.priors = {'Generic': th_Mahalanobis(dat['mean'],
                       dat['precision'],
                       self.prefix)}

    def create_prior_from_samples(self, samples):
        from sklearn.covariance import GraphicalLassoCV
        from numpy import asarray, linalg
        model = GraphicalLassoCV()
        model.fit(asarray(samples))
        return th_Mahalanobis(asarray(samples).mean(axis=0),
                           linalg.cholesky(model.precision_),
                           self.prefix)

    def __getitem__(self, pid):
        if pid not in self.priors:
            samples = [p[self.prefix:] for qsub in self.pose_subjects
                       for name, p in zip(qsub['pose_fnames'], qsub['pose_parms'])
                       if pid in name.lower()]
            self.priors[pid] = self.priors['Generic'] if len(samples) < 3 \
                               else self.create_prior_from_samples(samples)

        return self.priors[pid]

def save_priors(prior:th_Mahalanobis, outfile):
    """
    save prior as a pkl file
    """
    import pickle as pkl
    pkl.dump({
        "mean":prior.mean.cpu().numpy(),
        'precision':prior.prec.cpu().numpy()
    }, open(outfile, 'wb'))
    print('file saved to', outfile)


if __name__ == '__main__':
    "test loading prior"
    prior = get_prior('male')
    print(prior.mean)
    print(prior.prec.shape)
    # save_priors(prior, 'assets/prior_male.pkl')
    prior = get_prior('female')
    # save_priors(prior, 'assets/prior_female.pkl')
    print(prior.mean)
    print(prior.prec.shape)