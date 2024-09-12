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
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state, PILtoTorch
from argparse import ArgumentParser,Namespace
from arguments import ModelParams, PipelineParams, get_combined_args,args_init
from gaussian_renderer import GaussianModel
from gaussian_renderer import sample_incident_rays_all
import copy,pickle,time
from utils.general_utils import *
import imageio

from utils.mesh_utils import GaussianExtractor, to_cam_open3d, post_process_mesh
from utils.render_utils import generate_path, create_videos
import open3d as o3d


@torch.no_grad()
def render_interpolate(path, views, gaussians, pipeline, background): # select small portion of views
    inter_path = os.path.join(path, "intrinsic_dynamic_interpolate")
    makedirs(inter_path, exist_ok=True)
    inter_weights=[i*0.1 for i in range(0,21)]
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        for inter_weight in inter_weights:
            gaussians.colornet_inter_weight=inter_weight
            rendering = render(view, gaussians, pipeline, background)["render"]
            torchvision.utils.save_image(rendering, os.path.join(inter_path, f"{idx}_{inter_weight:.2f}.png"))
    gaussians.colornet_inter_weight=1.0

@torch.no_grad()
def render_multiview(path, views, gaussians, pipeline, background, scene_idx=0): # select small portion of views
    origin_views = copy.deepcopy(views)
    multiview_path = os.path.join(path, "multiview")
    view = views[scene_idx]
    sub_multiview_path=os.path.join(multiview_path,f"{scene_idx}")
    makedirs(sub_multiview_path, exist_ok=True)
    for o_idx, o_view in enumerate(tqdm(origin_views, desc="Rendering progress")):
        rendering = render(view, gaussians, pipeline, background, viewpoint_camera=o_view)["render"]
        torchvision.utils.save_image(rendering, os.path.join(sub_multiview_path, f"view_{o_idx}" + ".png"))

@torch.no_grad()
def render_intrinsic(path, views, gaussians, pipeline, background):
    for attr in ['base', 'rough', 'metal']:
        intrinsic_path = os.path.join(path, attr)
        makedirs(intrinsic_path, exist_ok=True)
        for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
            rendering = render(view, gaussians, pipeline, background)["render_"+attr]       
            torchvision.utils.save_image(rendering, os.path.join(intrinsic_path, '{0:05d}'.format(idx) + ".png"))

@torch.no_grad()
def generate_lightmap(path, views, gaussians, pipeline, background, indices=[0]):
    lightmaps_path = os.path.join(path, "incidents")
    res_lightmap = pipe.sampling_ray_res
    normal = gaussians.get_normal
    for idx_v, view in enumerate(tqdm(views, desc="Solving incident light maps")):
        incident_dirs_full = sample_incident_rays_all(normal[indices], res=res_lightmap)
        incidents_full = gaussians.get_incidents(view, incident_dirs_full, indices)
        incident_maps = incidents_full.reshape(-1, res_lightmap//4, res_lightmap, 3).permute(0,3,1,2)
        incident_maps = incident_maps.clamp(min=1e-9) ** gaussians.get_gamma # tone mapping
        lightmap_path = os.path.join(lightmaps_path, f"scene_{idx_v:02d}")
        makedirs(lightmap_path, exist_ok=True)
        for idx_l, lightmap in enumerate(incident_maps): #zip(indices, incident_maps):
            torchvision.utils.save_image(lightmap, os.path.join(lightmap_path, f"point_{idx_l}.png"))

@torch.no_grad()
def test_rendering_speed(views, gaussians, pipeline,background,use_cache=False): # don't use
    views=copy.deepcopy(views)
    length=min(1000,len(views))
    for idx in range(length):
        view=views[idx] 
        view.original_image=torch.nn.functional.interpolate(view.original_image.unsqueeze(0),size=(800,800)).squeeze()
        view.image_height,view.image_width=800,800
    if not use_cache:
        rendering = render(views[0], gaussians, pipeline, background)["render"]
        start_time=time.time()
        for idx in tqdm(range(length), desc="Rendering progress"):
            view=views[idx]
            rendering = render(view, gaussians, pipeline, background)["render"]
        end_time=time.time()
        
        avg_rendering_speed=(end_time-start_time)/length
        print(f"rendering speed:{avg_rendering_speed}s/image")
        return avg_rendering_speed
    else:
        for i in range(100):
            views[i+1].image_height,views[i+1].image_width=view.image_height,view.image_width
        rendering = render(views[0], gaussians, pipeline, background,store_cache=True)["render"]
        start_time=time.time()
        rendering = render(view, gaussians, pipeline, background,store_cache=True)["render"]
        for idx in tqdm(range(length), desc="Rendering progress"):
            view=views[idx+1]
            rendering = render(view, gaussians, pipeline, background,use_cache=True)["render"]       
        end_time=time.time()
        avg_rendering_speed=(end_time-start_time)/length
        print(f"rendering speed using cache:{avg_rendering_speed}s/image")
        return avg_rendering_speed
    
        
if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--skip_incident", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--render_interpolate", action="store_true",default=False)
    parser.add_argument("--render_multiview_video", action="store_true",default=False)
    parser.add_argument("--bake_mesh", action="store_true",default=False)
    parser.add_argument("--voxel_size", default=-1.0, type=float, help='Mesh: voxel size for TSDF') # From 2DGS
    parser.add_argument("--depth_trunc", default=-1.0, type=float, help='Mesh: Max depth range for TSDF')
    parser.add_argument("--sdf_trunc", default=-1.0, type=float, help='Mesh: truncation value for TSDF')
    parser.add_argument("--num_cluster", default=50, type=int, help='Mesh: number of connected clusters to export')
    parser.add_argument("--unbounded", action="store_true", help='Mesh: using unbounded mode for meshing')
    parser.add_argument("--mesh_res", default=1024, type=int, help='Mesh: resolution for unbounded mesh extraction')
    
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)
    
    dataset, iteration, pipe = model.extract(args), args.iteration, pipeline.extract(args)
    safe_state(args.quiet) # ??

    gaussians = GaussianModel(dataset.sh_degree,args) 
    scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False) 
    bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    
    train_dir = os.path.join(args.model_path, 'train', "ours_{}".format(scene.loaded_iter))
    test_dir = os.path.join(args.model_path, 'test', "ours_{}".format(scene.loaded_iter))
    gaussExtractor = GaussianExtractor(gaussians, render, pipe, bg_color=bg_color)    
    gaussians.set_eval(True)

    train_cameras=scene.getTrainCameras()
    test_cameras=scene.getTestCameras()

    if not args.skip_train:
        print("export training images ...")
        os.makedirs(train_dir, exist_ok=True)
        gaussExtractor.reconstruction(train_cameras)
        gaussExtractor.export_image(train_dir)  
        render_intrinsic(train_dir, train_cameras, gaussians, pipe, background) 

    if not args.skip_test and (len(test_cameras) > 0):
        print("export rendered testing images ...")
        os.makedirs(test_dir, exist_ok=True)
        gaussExtractor.reconstruction(test_cameras)
        gaussExtractor.export_image(test_dir)
        render_intrinsic(test_dir, test_cameras, gaussians, pipe, background) 
    
    indices = [20929, 27612, 8411, 3992, 8383, 8274, 39056, 1178, 78261, 2452, 2994, 1274, 19360, 18964]
    if not args.skip_incident and (len(indices)>0):
        print("export incident light maps ...")
        os.makedirs(train_dir, exist_ok=True)
        generate_lightmap(train_dir, train_cameras, gaussians, pipe, background, indices) 

    if args.render_multiview_video: 
        render_multiview(train_dir, train_cameras, gaussians, pipe, background, scene_idx=19)

    if args.render_interpolate: 
        render_interpolate(test_dir, train_cameras, gaussians, pipe, background)

    if args.bake_mesh:
        print("export mesh ...")
        os.makedirs(train_dir, exist_ok=True)
        gaussExtractor.reconstruction(train_cameras, mode='render_base')

        # extract the mesh and save
        if args.unbounded:
            name = 'fuse_unbounded.ply'
            mesh = gaussExtractor.extract_mesh_unbounded(resolution=args.mesh_res)
        else:
            name = 'fuse.ply'
            depth_trunc = (gaussExtractor.radius * 2.0) if args.depth_trunc < 0  else args.depth_trunc
            voxel_size = (depth_trunc / args.mesh_res) if args.voxel_size < 0 else args.voxel_size
            sdf_trunc = 5.0 * voxel_size if args.sdf_trunc < 0 else args.sdf_trunc
            mesh = gaussExtractor.extract_mesh_bounded(voxel_size=voxel_size, sdf_trunc=sdf_trunc, depth_trunc=depth_trunc)
        
        #gaussExtractor.gaussians.colornet_inter_weight = 1.0
        o3d.io.write_triangle_mesh(os.path.join(train_dir, name), mesh)
        print("mesh saved at {}".format(os.path.join(train_dir, name)))
        
        # post-process the mesh and save, saving the largest N clusters
        mesh_post = post_process_mesh(mesh, cluster_to_keep=args.num_cluster)
        o3d.io.write_triangle_mesh(os.path.join(train_dir, name.replace('.ply', '_post.ply')), mesh_post)
        print("mesh post processed saved at {}".format(os.path.join(train_dir, name.replace('.ply', '_post.ply'))))

    # All done
    gaussians.set_eval(False)
    print("\nRendering complete.")


