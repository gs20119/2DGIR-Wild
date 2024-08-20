#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import math
import numpy as np
import torch.nn.functional as F
from typing import NamedTuple
from .sh_utils import rotation_between_z


class BasicPointCloud(NamedTuple):
    points : np.array
    colors : np.array
    normals : np.array

def geom_transform_points(points, transf_matrix):
    P, _ = points.shape
    ones = torch.ones(P, 1, dtype=points.dtype, device=points.device)
    points_hom = torch.cat([points, ones], dim=1)
    points_out = torch.matmul(points_hom, transf_matrix.unsqueeze(0))

    denom = points_out[..., 3:] + 0.0000001
    return (points_out[..., :3] / denom).squeeze(dim=0)

def getWorld2View(R, t):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0
    return np.float32(Rt)

def getWorld2View2(R, t, translate=np.array([.0, .0, .0]), scale=1.0):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    C2W = np.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = np.linalg.inv(C2W)
    return np.float32(Rt)

def getProjectionMatrix(znear, zfar, fovX, fovY):
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros(4, 4)

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P

def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))

def focal2fov(focal, pixels):
    return 2*math.atan(pixels/(2*focal))


# From Relightable-3DGS
def fibonacci_sphere_sampling(normals, sample_num, random_rotate=True):
    pre_shape = normals.shape[:-1] # N
    if len(pre_shape) > 1:
        normals = normals.reshape(-1, 3) # [N,3]
    delta = np.pi * (3.0 - np.sqrt(5.0))

    # fibonacci sphere sample around z axis
    idx = torch.arange(sample_num, dtype=torch.float, device='cuda')[None] # [1,S]
    z = (1 - 2 * idx / (2 * sample_num - 1)).clamp_min(np.sin(10/180*np.pi)) # [1,S]
    rad = torch.sqrt(1 - z ** 2)
    theta = delta * idx
    if random_rotate:
        theta = torch.rand(*pre_shape, 1, device='cuda') * 2 * np.pi + theta # [N,S]
    y = torch.cos(theta) * rad # [N,S]
    x = torch.sin(theta) * rad # [N,S]
    z_samples = torch.stack([x, y, z.expand_as(y)], dim=-2) # [N,3,S]

    rotation_matrix = rotation_between_z(normals) # [N,3,3]
    incident_dirs = rotation_matrix @ z_samples # [N,3,S]
    incident_dirs = F.normalize(incident_dirs, dim=-2).transpose(-1, -2) # [N,S,3]
    incident_areas = torch.ones_like(incident_dirs)[..., 0:1] * 2 * np.pi
    if len(pre_shape) > 1:
        incident_dirs = incident_dirs.reshape(*pre_shape, sample_num, 3)
        incident_areas = incident_areas.reshape(*pre_shape, sample_num, 1)
    return incident_dirs, incident_areas

# (S1=res/4 x S2=res) spherical coordinate sampling
def full_hemisphere_sampling(normals, res=100): 
    pre_shape = normals.shape[:-1] # N
    if len(pre_shape) > 1:
        normals = normals.reshape(-1, 3) # [N,3]

    phi_ = torch.tensor(np.linspace(0.0,0.5*np.pi,res//4), dtype=torch.float, device='cuda') # [S1]
    theta_ = torch.tensor(np.linspace(0.0,2.0*np.pi,res), dtype=torch.float, device='cuda') # [S2]
    phi, theta = torch.meshgrid(phi_, theta_, indexing='ij') # [S1,S2]
    x = torch.sin(theta) * torch.cos(phi)
    y = torch.cos(theta) * torch.cos(phi)
    z = torch.sin(phi)
    samples_ = torch.stack([x, y, z], dim=0).flatten(start_dim=-2)[None] # [1,3,S1*S2]
    samples = samples_.expand(normals.shape[0],-1,-1) # [N,3,S1*S2]

    rotation_matrix = rotation_between_z(normals) # [N,3,3]
    incident_dirs = rotation_matrix @ samples # [N,3,S1*S2]
    incident_dirs = F.normalize(incident_dirs, dim=-2).transpose(-1, -2) # [N,S1*S2,3]
    
    if len(pre_shape) > 1:
        incident_dirs = incident_dirs.reshape(*pre_shape, sample_num, 3)
    return incident_dirs