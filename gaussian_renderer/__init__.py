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
from diff_surfel_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh
from utils.point_utils import depth_to_normal
import numpy as np

def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None,\
    other_viewpoint_camera=None, store_cache=False, use_cache=False, 
    is_training=False, dict_params=None, point_features=None):
    """
    Render the scene. 
    Background tensor (bg_color) must be on GPU!
    """
    if other_viewpoint_camera is not None: #render using other camera center
        viewpoint_camera.camera_center=other_viewpoint_camera.camera_center
    
    if use_cache: pc.forward_cache(viewpoint_camera)
    elif point_features is not None: pc.forward_interpolate(viewpoint_camera,point_features)
    else: pc.forward(viewpoint_camera,store_cache)

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0     #[Npoint,3]
    try: 
        screenspace_points.retain_grad()
    except: 
        pass 
    
    if other_viewpoint_camera is not None: #render using other camera center
        viewpoint_camera=other_viewpoint_camera

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,  #4*4
        projmatrix=viewpoint_camera.full_proj_transform,   #4*4
        sh_degree=pc.active_sh_degree,                    #0,
        campos=viewpoint_camera.camera_center,            #3
        prefiltered=False,
        debug=pipe.debug
    )
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz              
    means2D = screenspace_points      #[Npoint,3]
    opacity = pc.get_opacity          #[Npoint,1]  

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        splat2world = pc.get_covariance(scaling_modifier)
        W, H = viewpoint_camera.image_width, viewpoint_camera.image_height
        near, far = viewpoint_camera.znear, viewpoint_camera.zfar
        ndc2pix = torch.tensor([
            [W / 2, 0, 0, (W-1) / 2],
            [0, H / 2, 0, (H-1) / 2],
            [0, 0, far-near, near],
            [0, 0, 0, 1]]).float().cuda().T
        world2pix =  viewpoint_camera.full_proj_transform @ ndc2pix
        cov3D_precomp = (splat2world[:, [0,1,3]] @ world2pix[:,[0,1,3]]).permute(0,2,1).reshape(-1, 9) # column major
    else:
        scales = pc.get_scaling        #[Npoint,2] 
        rotations = pc.get_rotation    #[Npoint,4]   
        
    base_color = pc.get_base_color
    roughness = pc.get_roughness
    metallic = pc.get_metallic
    normal = pc.get_normal
    incidents = pc.get_incidents  # incident rgb - color net TODO
    viewdirs = F.normalize(viewpoint_camera.camera_center - means3D, dim=-1)

    dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_shs.shape[0], 1))
    dir_pp_normalized = F.normalize(dir_pp, dim=-1)

    brdf_colors, extra_results = rendering_equation(
        base_color, roughness, normal.detach(), viewdirs, incidents,
        incident_dirs_precompute=pc._incident_dirs, 
        incident_areas_precompute=pc._incident_areas)

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    rendered_image, radii, allmap = rasterizer(
        means3D = means3D,       # [Npoint,3]
        means2D = means2D,      #[Npoint,3]
        shs = shs,              #[Npoint,16,3]
        colors_precomp = brdf_colors,
        opacities = opacity,     #[Npoint,1]  
        scales = scales,            #[Npoint,3] 
        rotations = rotations,      #[Npoint,4]    
        cov3D_precomp = cov3D_precomp)

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    rets = {"render": rendered_image,
            "viewspace_points": means2D,
            "visibility_filter" : radii > 0,
            "radii": radii}        
    
    # 2DGS renderer part (additional)
    # additional regularizations
    render_alpha = allmap[1:2]

    # get normal map
    # transform normal from view space to world space
    render_normal = allmap[2:5]
    render_normal = (render_normal.permute(1,2,0) @ (viewpoint_camera.world_view_transform[:3,:3].T)).permute(2,0,1)
    
    # get median depth map
    render_depth_median = allmap[5:6]
    render_depth_median = torch.nan_to_num(render_depth_median, 0, 0)

    # get expected depth map
    render_depth_expected = allmap[0:1]
    render_depth_expected = (render_depth_expected / render_alpha)
    render_depth_expected = torch.nan_to_num(render_depth_expected, 0, 0)
    
    # get depth distortion map
    render_dist = allmap[6:7]

    # psedo surface attributes
    # surf depth is either median or expected by setting depth_ratio to 1 or 0
    # for bounded scene, use median depth, i.e., depth_ratio = 1; 
    # for unbounded scene, use expected depth, i.e., depth_ration = 0, to reduce disk anliasing.
    surf_depth = render_depth_expected * (1-pipe.depth_ratio) + (pipe.depth_ratio) * render_depth_median
    
    # assume the depth points form the 'surface' and generate psudo surface normal for regularizations.
    surf_normal = depth_to_normal(viewpoint_camera, surf_depth)
    surf_normal = surf_normal.permute(2,0,1)
    # remember to multiply with accum_alpha since render_normal is unnormalized.
    surf_normal = surf_normal * (render_alpha).detach()

    rets.update({
        'rend_alpha': render_alpha,
        'rend_normal': render_normal,
        'rend_dist': render_dist,
        'surf_depth': surf_depth,
        'surf_normal': surf_normal,
    })

    return rets



def rendering_equation(base_color, roughness, normals, viewdirs,
                              incident_lights, incident_dirs_precompute=None, incident_areas_precompute=None): # TODO
    incident_dirs, incident_areas = incident_dirs_precompute, incident_areas_precompute

    n_d_i = (normals[:, None] * incident_dirs).sum(-1, keepdim=True).clamp(min=0)
    f_d = base_color[:, None] / np.pi
    f_s = GGX_specular(normals, viewdirs, incident_dirs, roughness, fresnel=0.04)

    transport = incident_lights * incident_areas * n_d_i  # （num_pts, num_sample, 3)
    specular = ((f_s)*transport).mean(dim=-2)
    pbr = ((f_d+f_s)*transport).mean(dim=-2)
    diffuse_light = transport.mean(dim=-2)

    extra_results = {
        "incident_dirs": incident_dirs,
        "incident_lights": incident_lights,
        "diffuse_light": diffuse_light,
        "specular": specular,
    }

    return pbr, extra_results


def GGX_specular(normal, pts2c, pts2l, roughness, fresnel):
    L = F.normalize(pts2l, dim=-1)  # [nrays, nlights, 3]
    V = F.normalize(pts2c, dim=-1)  # [nrays, 3]
    H = F.normalize((L + V[:, None, :]) / 2.0, dim=-1)  # [nrays, nlights, 3]
    N = F.normalize(normal, dim=-1)  # [nrays, 3]

    NoV = torch.sum(V * N, dim=-1, keepdim=True)  # [nrays, 1]
    N = N * NoV.sign()  # [nrays, 3]

    NoL = torch.sum(N[:, None, :] * L, dim=-1, keepdim=True).clamp_(1e-6, 1)  # [nrays, nlights, 1] TODO check broadcast
    NoV = torch.sum(N * V, dim=-1, keepdim=True).clamp_(1e-6, 1)  # [nrays, 1]
    NoH = torch.sum(N[:, None, :] * H, dim=-1, keepdim=True).clamp_(1e-6, 1)  # [nrays, nlights, 1]
    VoH = torch.sum(V[:, None, :] * H, dim=-1, keepdim=True).clamp_(1e-6, 1)  # [nrays, nlights, 1]

    alpha = roughness * roughness  # [nrays, 3]
    alpha2 = alpha * alpha  # [nrays, 3]
    k = (alpha + 2 * roughness + 1.0) / 8.0
    FMi = ((-5.55473) * VoH - 6.98316) * VoH
    frac0 = fresnel + (1 - fresnel) * torch.pow(2.0, FMi)  # [nrays, nlights, 3]
    
    frac = frac0 * alpha2[:, None, :]  # [nrays, 1]
    nom0 = NoH * NoH * (alpha2[:, None, :] - 1) + 1

    nom1 = NoV * (1 - k) + k
    nom2 = NoL * (1 - k[:, None, :]) + k[:, None, :]
    nom = (4 * np.pi * nom0 * nom0 * nom1[:, None, :] * nom2).clamp_(1e-6, 4 * np.pi)
    spec = frac / nom
    return spec