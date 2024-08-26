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
from utils.graphics_utils import fibonacci_sphere_sampling, full_hemisphere_sampling
import torch.nn.functional as F
import numpy as np

def render(scene_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, 
        scaling_modifier = 1.0, viewpoint_camera=None, is_training=False):
    """
    Render the scene. 
    Background tensor (bg_color) must be on GPU!
    """

    if viewpoint_camera is None: viewpoint_camera = scene_camera
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0     #[Npoint,3]
    try: screenspace_points.retain_grad()
    except: pass 

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

    # we will use rgb color from rendering_equation without shs
    base_color = pc.get_base_color
    roughness = pc.get_roughness
    metallic = pc.get_metallic
    normal = pc.get_normal.detach() # [N,3] (why do we need detach here? TODO)
    view = pc.get_viewdirs(viewpoint_camera).detach() # outgoing light [N,3]
    nov = torch.sum(normal*view, dim=-1, keepdim=True)
    normal *= nov.sign() # align normal with view_dirs

    sample_num = pipe.sampling_ray_num # S
    incident_dirs, incident_areas = sample_incident_rays(normal, is_training, sample_num) # incident lights [N,S,3], [N,S,1]
    incidents = pc.get_incidents(scene_camera, incident_dirs)  # incident rgb color from color net. [N,S,3]
    
    brdf_color, extra_results = rendering_equation(
        base_color, roughness, metallic, normal, view, 
        incidents, incident_dirs, incident_areas)
    colors = torch.cat([brdf_color, base_color, roughness, metallic], dim=-1)

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    rendered_images, radii, allmap = rasterizer(
        means3D = means3D,            # [N,3]
        means2D = means2D,            # [N,3]
        shs = None,                   # [N,16,3], not used
        colors_precomp = colors,      # [N,3+3+1+1]
        opacities = opacity,          # [N,1]  
        scales = scales,              # [N,2] 
        rotations = rotations,        # [N,4] (quarternion)    
        cov3D_precomp = cov3D_precomp)
    gamma = pc.get_gamma
    rendered_brdf = rendered_images[:3].clamp(min=1e-9) ** gamma # HDR -> SDR tone mapping

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    rets = {
        "render": rendered_brdf, # [3,H,W]
        "render_base": rendered_images[3:6], # [3,H,W]
        "render_rough": rendered_images[6:7], # [1,H,W]
        "render_metal": rendered_images[7:8], # [1,H,W]
        "viewspace_points": means2D,
        "visibility_filter": radii > 0,
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


# From Relightable 3DGS
def sample_incident_rays(normals, is_training=False, sample_num=24): 
    incident_dirs, incident_areas = fibonacci_sphere_sampling(normals, sample_num, random_rotate=is_training) 
    return incident_dirs, incident_areas  # [N,S,3], [N,S,1]

def sample_incident_rays_all(normals, res=100): 
    incident_dirs = full_hemisphere_sampling(normals, res=res)
    return incident_dirs  # [N,(res/4)*(res),3], [N,(res/4)*(res),1]


def rendering_equation(base_color, roughness, metallic, normals, 
            viewdirs, incident_lights, incident_dirs, incident_areas): 

    n_d_i = (normals[:, None, :] * incident_dirs).sum(-1, keepdim=True).clamp(min=0) 
    f_d = (1 - metallic[:, None, :]) * base_color[:, None, :] / np.pi
    f_s = GGX_specular(normals, viewdirs, incident_dirs, base_color, roughness, metallic)

    transport = incident_lights * incident_areas * n_d_i  # [num_pts, num_sample, 3]
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


def GGX_specular(normal, pts2c, pts2l, base_color, roughness, metallic): # Unreal Engine
    L = F.normalize(pts2l, dim=-1)  # [nrays, nlights, 3]
    V = F.normalize(pts2c, dim=-1)  # [nrays, 3]
    H = F.normalize((L + V[:, None, :]) / 2.0, dim=-1)  # [nrays, nlights, 3]
    N = F.normalize(normal, dim=-1)  # [nrays, 3]

    NoL = torch.sum(N[:, None, :] * L, dim=-1, keepdim=True).clamp_(1e-6, 1)  # [nrays, nlights, 1] 
    NoV = torch.sum(N * V, dim=-1, keepdim=True).clamp_(1e-6, 1)              # [nrays, 1]
    NoH = torch.sum(N[:, None, :] * H, dim=-1, keepdim=True).clamp_(1e-6, 1)  # [nrays, nlights, 1]
    VoH = torch.sum(V[:, None, :] * H, dim=-1, keepdim=True).clamp_(1e-6, 1)  # [nrays, nlights, 1]

    # D: Trowbridge-Reitz GGX
    alpha = roughness * roughness  # or roughness
    alpha2 = alpha * alpha  
    nom0 = NoH * NoH * (alpha2[:, None, :] - 1) + 1
    D = alpha2[:, None, :] / (np.pi * nom0 * nom0) 

    # F: Fresnel-Schlick approximation
    F0 = 0.04 * (1-metallic) + metallic * base_color # [nrays, 3]
    Fr = F0[:, None, :] + (1 - F0[:, None, :]) * torch.pow(1 - VoH, 5.0)

    # G: Smith's joint approximation
    k = alpha[:, None, :] # or (alpha + 2 * roughness + 1.0) / 8.0
    nom1 = NoV[:, None, :] * (1 - k) + k
    nom2 = NoL * (1 - k) + k
    G = 0.5 / (NoL * nom1 + NoV[:, None, :] * nom2)

    spec = D * Fr * G / (4.0 * NoV[:, None, :] * NoL)
    return spec