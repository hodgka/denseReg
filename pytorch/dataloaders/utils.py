import matplotlib.pyplot as plt
import numpy as np
import torch
import scipy.stats as st
import cv2
from collections import namedtuple

CameraConfig = namedtuple('CameraConfig', 'fx,fy,cx,cy,w,h')

_pro = lambda pt3, cfg: [pt3[0]*cfg[0]/pt3[2]+cfg[2], pt3[1]*cfg[1]/pt3[2]+cfg[3], pt3[2]]
_bpro = lambda pt2, cfg: [(pt2[0]-cfg[2])*pt2[2]/cfg[0], (pt2[1]-cfg[3])*pt2[2]/cfg[1], pt2[2]] 


def xyz_to_uvd(xyz, camera_config):
    xyz = xyz.reshape((-1, 3))
    uvd = [_pro(pt3, camera_config) for pt3 in xyz]
    return np.array(uvd)


def uvd2xyz(uvd, cfg):
    '''uvd: list of uvd points
    cfg: camera configuration
    '''
    uvd = uvd.reshape((-1,3))
    # backprojection
    xyz = [_bpro(pt2, cfg) for pt2 in uvd]
    return np.array(xyz)


def _pytorch_gaussian_kern(filter_size=10, sigma=3):
    '''
        return an np array of a Gaussian kernel
    '''
    interval = (2*sigma+1.0)/(filter_size)
    x = torch.linspace(-sigma - interval / 2., sigma + interval / 2., filter_size + 1)
    rv = torch.distributions.normal.Normal(torch.tensor([0.0]), torch.tensor([1.0]))
    cdf = rv.cdf(x)
    kern1d = cdf[1:] - cdf[:-1]
    kernel_raw = torch.sqrt(torch.ger(kern1d, kern1d))    
    kernel = kernel_raw/kernel_raw.sum()
    return kernel

def gaussian_filter(filter_size=10, sigma=3):
    kernel = _pytorch_gaussian_kern(filter_size, sigma)
    with torch.no_grad(kernel):
        kernel = kernel.unsqueeze(-1).unsqueeze(-1)
    return kernel

def heatmap_from_uvd(uvd_points, camera_config, gaussian_filter):
    uvd_points = uvd_points.view(-1, 3)
    num_points = uvd_points.size(0)

    nn = torch.arange(num_points).view(-1, 1)
    xx = uvd_points[:, 0]
    xx = torch.clamp(xx, 0, camera_config.w - 1)
    xx = xx.view(-1, 1)

    yy = uvd_points[:, 1]
    yy = torch.clamp(yy, 0, camera_config.h - 1)
    yy = yy.view(-1, 1)

    indices = torch.cat([nn, yy, xx], axis=1)
    val = 1.0
    
    #TODO need to finish this


    