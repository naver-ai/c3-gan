import numpy as np
from config import cfg

from scipy import linalg
from scipy.optimize import linear_sum_assignment
from sklearn.preprocessing import normalize



## evaluate clustering performance
def get_stat(gen_label, num_gt_lab, gt_lab):
    c_stat = np.zeros([num_gt_lab,num_gt_lab])
    for i in range(len(gt_lab)):
        gt_idx = int(gt_lab[i])
        c_stat[gt_idx][gen_label[i]] += 1
    return c_stat

def get_match(stat):
    _, col_ind = linear_sum_assignment(stat.max()-stat)
    return col_ind

def get_acc(stat, col_ind, over):
    tot = 0
    for i in range(stat.shape[0]):
        tot += stat[i][col_ind[i]]
    return tot/(np.sum(stat)/over)

def get_nmi(stat):
    n,m = stat.shape
    pij = stat/np.sum(stat)
    pi = np.sum(pij, 1)
    pj = np.sum(pij, 0)
    enti = sum([-pi[i]*np.log2(pi[i]+1e-6) for i in range(n)])
    entj = sum([-pj[i]*np.log2(pj[i]+1e-6) for i in range(m)])
    mi = 0
    for i in range(n):
        for j in range(m):
            mi += pij[i][j]*(np.log2(pij[i][j]/(pi[i]*pj[j]+1e-6)+1e-6))
    return mi/max(enti, entj)


## evaluate the image generation performance
## This code has fetched from https://github.com/mseitzer/pytorch-fid.
def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)
    assert mu1.shape == mu2.shape
    assert sigma1.shape == sigma2.shape
    diff = mu1 - mu2
    
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real
    tr_covmean = np.trace(covmean)

    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2*tr_covmean
