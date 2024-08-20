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

import os
import torch
import numpy as np
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
from gaussian_renderer import sample_incident_rays_all
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams,args_init
import logging,time,shutil,torchvision
import matplotlib.pyplot as plt
from render import *
from metrics import evaluate as evaluate_metrics
from metrics_half import evaluate as evaluate_metrics_half
from kornia.filters import laplacian, spatial_gradient
import pickle

import lpips
from arguments import *
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

def training(dataset, opt, pipe, testing_iterations, saving_iterations, debug_from, args):

    '''
    dataset:Groupparams  ModelParams
    opt:Groupparams  OptimizationParams
    pipe:Groupparams  PipelineParams

    '''

    # create and save logs 
    saving_iterations+=[opt.iterations]
    dataset_name=args.scene_name
    dataset.data_perturb=args.data_perturb
    log_file_path=os.path.join(dataset.model_path,"logs",f"({time.strftime('%Y-%m-%d_%H-%M-%S')})_Iteration({opt.iterations})_({dataset_name}).log")
    os.makedirs(os.path.dirname(log_file_path),exist_ok=True)
    logging.basicConfig(filename=log_file_path, level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info(f"Experiment Configuration: {args}")
    logging.info(f"Model initialization and Data reading....")

    # save configurations
    with open(os.path.join(args.model_path,'cfg_arg.pkl'), 'wb') as file:
        pickle.dump(args, file)
    
    # rendered images in training progress (train_temp_something)
    # TODO gather these paths in a dictionary 
    render_temp_path=os.path.join(dataset.model_path,"train_temp_rendering")
    gt_temp_path=os.path.join(render_temp_path, "gt")
    base_temp_path=os.path.join(render_temp_path, "base")
    rough_temp_path=os.path.join(render_temp_path, "rough")
    metal_temp_path=os.path.join(render_temp_path, "metal")
    feature_maps_temp_path=os.path.join(render_temp_path, "feature_maps")
    incident_map_temp_path=os.path.join(render_temp_path, "incident_light_maps")
    render_temp_path=os.path.join(render_temp_path, "render")
    
    if os.path.exists(render_temp_path): shutil.rmtree(render_temp_path)
    if os.path.exists(gt_temp_path): shutil.rmtree(gt_temp_path)
    if os.path.exists(base_temp_path): shutil.rmtree(base_temp_path)
    if os.path.exists(rough_temp_path): shutil.rmtree(rough_temp_path)
    if os.path.exists(metal_temp_path): shutil.rmtree(metal_temp_path)
    if os.path.exists(feature_maps_temp_path): shutil.rmtree(feature_maps_temp_path)
    if os.path.exists(incident_map_temp_path): shutil.rmtree(incident_map_temp_path)

    os.makedirs(render_temp_path,exist_ok=True)
    os.makedirs(gt_temp_path,exist_ok=True)
    os.makedirs(base_temp_path,exist_ok=True)
    os.makedirs(rough_temp_path,exist_ok=True)
    os.makedirs(metal_temp_path,exist_ok=True)
    os.makedirs(feature_maps_temp_path,exist_ok=True)
    os.makedirs(incident_map_temp_path,exist_ok=True)

    if args.use_features_mask:
        render_temp_mask_path=os.path.join(dataset.model_path,"train_mask_temp_rendering")
        if os.path.exists(render_temp_mask_path):
            shutil.rmtree(render_temp_mask_path)
        os.makedirs(render_temp_mask_path,exist_ok=True)


    # prepare training
    first_iter = 0
    tb_writer = prepare_output_and_logger(args)        
    gaussians = GaussianModel(dataset.sh_degree,args)          
    scene = Scene(dataset, gaussians,shuffle=False)                     
    gaussians.training_setup(opt)      

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)
    
    viewpoint_stack = None
    ema_loss_for_log = 0.0
    ema_psnr_for_log = 0.0
    ema_dist_for_log = 0.0
    ema_normal_for_log = 0.0

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1

    len_timedata=len(scene.getTrainCameras())
    record_loss=[]
    ech_loss=0
    if args.use_lpips_loss:
        lpips_criteria = lpips.LPIPS(net='vgg').to("cuda:0")
    logging.info(f"Start trainning....")

    if args.warm_up_iter>0: 
        gaussians.set_learning_rate("box_coord",0.0) 
    

    for iteration in range(first_iter, opt.iterations + 1):  
        iter_start.record()
        gaussians.update_learning_rate(iteration, args.warm_up_iter) 

        # Every 1000 its we increase the levels of SH up to a maximum degree
        # if iteration % 1000 == 0:
        #    gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()          #
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))  #

        # Render
        if (iteration-1) == debug_from: pipe.debug = True
        bg = torch.rand((3), device="cuda") if opt.random_background else background

        render_pkg = render(viewpoint_cam, gaussians, pipe, bg, is_training=True)     
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        gt_image = viewpoint_cam.original_image.cuda()

        # Color Loss
        if args.use_features_mask and iteration>args.features_mask_iters: #2500
            mask=gaussians.features_mask
            mask=torch.nn.functional.interpolate(mask,size=(image.shape[-2:]))
            Ll1 = l1_loss(image*mask, gt_image*mask)                        
            loss = (1.0-opt.lambda_dssim)*Ll1 + opt.lambda_dssim*(1.0-ssim(image*mask, gt_image*mask))
            loss += (torch.square(1-gaussians.features_mask)).mean()*args.features_mask_loss_coef # L_vm (masking)
        else:
            Ll1 = l1_loss(image, gt_image)                     
            loss = (1.0-opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0-ssim(image, gt_image))

        if args.use_lpips_loss: 
            loss += lpips_criteria(image,gt_image).mean()*args.lpips_loss_coef

        # Gaussian-Wild regularization
        if args.use_scaling_loss:
            loss += torch.abs(gaussians.get_scaling).mean()*args.scaling_loss_coef 

        if args.use_box_coord_loss:
            loss += torch.relu(torch.abs(gaussians.coordinates[:,:-1,:])-1).mean()*args.box_coord_loss_coef # L_sc

        # 2DGS regularization 
        lambda_normal = opt.lambda_normal if iteration > 7000 else 0.0
        lambda_dist = opt.lambda_dist if iteration > 3000 else 0.0

        rend_dist = render_pkg["rend_dist"]
        rend_normal  = render_pkg['rend_normal']
        surf_normal = render_pkg['surf_normal']
        normal_error = (1-(rend_normal*surf_normal).sum(dim=0))[None]
        normal_loss = lambda_normal*(normal_error).mean()
        dist_loss = lambda_dist*(rend_dist).mean()

        # Inverse Rendering Regularization
        #data, img = [C, H, W]
        image_base = render_pkg["render_base"]
        image_rough = render_pkg["render_rough"] 
        image_metal = render_pkg["render_metal"]

        grad_rough = spatial_gradient(image_rough[None], order=1)[0] # [1,2,H,W]
        grad_metal = spatial_gradient(image_metal[None], order=1)[0] # [1,2,H,W]
        grad_base = spatial_gradient(image_base[None], order=1)[0] # [3,2,H,W]
        grad_gt_color = spatial_gradient(gt_image[None], order=2)[0,:,[0,2]] # [3,2,H,W]

        rough_smooth = opt.lambda_smooth_r * (grad_rough.abs() * torch.exp(-grad_base.norm(dim=0, keepdim=True))).mean() 
        metal_smooth = opt.lambda_smooth_m * (grad_metal.abs() * torch.exp(-grad_base.norm(dim=0, keepdim=True))).mean()        
        base_smooth = opt.lambda_smooth_b * (grad_base.abs() * torch.exp(-grad_gt_color.abs())).mean()
        
        #if args.use_features_mask and iteration>args.features_mask_iters: # TODO
        #    smooth_loss = (data_grad.abs() * torch.exp(-rgb.grad.abs()) * mask).sum(1).mean() 
        #    smooth_loss = (data_grad.abs() * torch.exp(-rgb.grad.norm(dim=1, keepdim=True)) * mask).sum(1).mean() #2
        
        smooth_loss = base_smooth + rough_smooth + metal_smooth 

        # Total Loss
        psnr_ = psnr(image,gt_image).mean().double()    
        total_loss = loss + normal_loss + dist_loss + smooth_loss
        total_loss.backward()
        iter_end.record()
        

        with torch.no_grad():
            # show progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_psnr_for_log = 0.4 * psnr_ + 0.6 * ema_psnr_for_log
            ema_dist_for_log = 0.4 * dist_loss.item() + 0.6 * ema_dist_for_log 
            ema_normal_for_log = 0.4 * normal_loss.item() + 0.6 * ema_normal_for_log

            if (iteration%1000==0) or (iteration==1):
                imgname = f"iter{iteration}_"+viewpoint_cam.image_name + ".png"
                torchvision.utils.save_image(image, os.path.join(render_temp_path, imgname))
                torchvision.utils.save_image(gt_image, os.path.join(gt_temp_path, imgname))
                
                torchvision.utils.save_image(image_base, os.path.join(base_temp_path, imgname))
                torchvision.utils.save_image(image_rough, os.path.join(rough_temp_path, imgname))
                torchvision.utils.save_image(image_metal, os.path.join(metal_temp_path, imgname))

                if args.use_features_mask:
                    torchvision.utils.save_image(
                        gaussians.features_mask.repeat(1,3,1,1), 
                        os.path.join(render_temp_mask_path, imgname))

            # save feature maps into image
            if iteration%2000==0:
                map_num = gaussians.map_num 
                for i in range(10): # range(gaussians._feature_maps.shape[1]):
                    feature_map_temp_path = os.path.join(feature_maps_temp_path, f"{(i//map_num):02d}_{(i%map_num):02d}")
                    os.makedirs(feature_map_temp_path,exist_ok=True)
                    feature_image = gaussians._feature_maps[0,i,:,:].detach()
                    # print(feature_image.min().item(), feature_image.max().item())
                    feature_image -= feature_image.min()
                    feature_image /= feature_image.max()
                    torchvision.utils.save_image(feature_image, os.path.join(feature_map_temp_path, imgname))

            # save incident light maps into image    
            if iteration%2000==0 or iteration==10:
                print("Extracting incident light maps")
                mapname = f"iter{iteration}_"+viewpoint_cam.image_name
                res_lightmap = pipe.sampling_ray_res
                normal = gaussians.get_normal
                indices = np.arange(normal.shape[0])%5000==0

                incident_dirs_full = sample_incident_rays_all(normal[indices], res=res_lightmap).detach()
                incidents_full = gaussians.get_incidents(viewpoint_cam, incident_dirs_full, indices)
                incident_maps = incidents_full.reshape(-1, res_lightmap//4, res_lightmap, 3).permute(0,3,1,2)
                
                for idx, lightmap in enumerate(incident_maps):
                    torchvision.utils.save_image(lightmap, os.path.join(incident_map_temp_path, mapname + f"({idx:02d}).png"))


            if iteration % 10 == 0:
                loss_dict = {
                    "Loss": f"{ema_loss_for_log:.{5}f}",
                    "psnr": f"{psnr_:.{2}f}",
                    "distort": f"{ema_dist_for_log:.{5}f}", # TODO
                    "normal": f"{ema_normal_for_log:.{5}f}",
                    "points": f"{len(gaussians.get_xyz)}"
                }            
                progress_bar.set_postfix(loss_dict)
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            ech_loss += loss.item()
            if (iteration)%len_timedata==0:
                logging.info(f'Iteration {iteration}/{opt.iterations}, Loss: {ech_loss/len_timedata}')
                logging.info(f"Iteration {iteration}:Guassian points' number:{gaussians._xyz.shape[0]}")
                record_loss.append(ech_loss/len_timedata)
                ech_loss=0            

            if tb_writer is not None:
                tb_writer.add_scalar('train_loss_patches/dist_loss', ema_dist_for_log, iteration)
                tb_writer.add_scalar('train_loss_patches/normal_loss', ema_normal_for_log, iteration) # TODO
            gaussians.set_eval(True)
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background))
            gaussians.set_eval(False)

            if (iteration in saving_iterations):
                logging.info("[ITER {}] Saving Gaussians".format(iteration))
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)
            
            # send current metrics to network_gui
            if network_gui.conn == None:
                network_gui.try_connect()
            while network_gui.conn != None:
                try:
                    net_image_bytes = None
                    custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                    if custom_cam != None:
                        render_pkg = render(custom_cam, gaussians, pipe, background, scaling_modifer)
                        net_image = render_pkg["render"] 
                        net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                    metrics_dict = {
                        "#": gaussians.get_opacity.shape[0],
                        "loss": ema_loss_for_log
                        # Add more metrics as needed
                    }
                    network_gui.send(net_image_bytes, dataset.source_path, metrics_dict)
                    if do_training and ((iteration < int(opt.iterations)) or not keep_alive): break
                except Exception as e: network_gui.conn = None


        with torch.no_grad():
            # Densification
            if iteration < opt.densify_until_iter:
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])#
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)       

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold,args.opacity_threshold, scene.cameras_extent, size_threshold)
                
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)


    # training finished. drawing loss curve
    fig = plt.figure()
    logging.info("Drawing Training loss curve")
    print("\nDrawing Training loss curve")
    plt.plot(record_loss[150:])
    plt.xlabel('Epochs')
    plt.ylabel('Training loss')
    plt.title('Training error curve')
    os.makedirs(os.path.join(scene.model_path,"train", "ours_{}".format(iteration)),exist_ok=True)
    fig.savefig(os.path.join(scene.model_path,"train", "ours_{}".format(iteration),"training_loss.png"))


    # render result and evaluate metrics
    with torch.no_grad():
        if args.render_after_train:
            gaussians.set_eval(True)
            gaussExtractor = GaussianExtractor(gaussians, render, pipe, bg_color=bg_color)
            train_dir = os.path.join(args.model_path, 'train', "ours_{}".format(scene.loaded_iter))
            test_dir = os.path.join(args.model_path, 'test', "ours_{}".format(scene.loaded_iter))

            torch.cuda.empty_cache()
            logging.info(f"Rendering testing set [{len(scene.getTestCameras())}]...")
            gaussExtractor.reconstruction(scene.getTrainCameras())
            gaussExtractor.export_image(test_dir) 
            
            torch.cuda.empty_cache()
            logging.info(f"Rendering training set [{len(scene.getTrainCameras())}]...")
            gaussExtractor.reconstruction(scene.getTrainCameras())
            gaussExtractor.export_image(train_dir) 

            logging.info(f"Rendering training set's intrinsic image [{len(scene.getTrainCameras())}]...")
            render_intrinsic(train_dir, scene.getTrainCameras(), gaussians, pipe, background)
                
            if args.metrics_after_train:
                logging.info("Evaluating metrics on testing set...")
                evaluate_metrics([dataset.model_path],use_logs=True)
                logging.info("Evaluating metrics half image on testing set...")
                evaluate_metrics_half([dataset.model_path],use_logs=True)
            gaussians.set_eval(False)

            
def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

@torch.no_grad()
def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)
        tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
        tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)

    # Report test and samples of training set
    if iteration in testing_iterations: 
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 500, 25)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    render_pkg = renderFunc(viewpoint, scene.gaussians, *renderArgs) 
                    image = torch.clamp(render_pkg["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)

                    if tb_writer and (idx < 5):
                        from utils.general_utils import colormap 
                        depth = render_pkg["surf_depth"]
                        norm = depth.max()
                        depth = depth / norm
                        depth = colormap(depth.cpu().numpy()[0], cmap='turbo')
                        tb_writer.add_images(config['name'] + "_view_{}/depth".format(viewpoint.image_name), depth[None], global_step=iteration)
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        
                        try: 
                            rend_alpha = render_pkg['rend_alpha']
                            rend_normal = render_pkg["rend_normal"]*0.5 + 0.5
                            surf_normal = render_pkg["surf_normal"]*0.5 + 0.5
                            tb_writer.add_images(config['name'] + "_view_{}/rend_normal".format(viewpoint.image_name), rend_normal[None], global_step=iteration)
                            tb_writer.add_images(config['name'] + "_view_{}/surf_normal".format(viewpoint.image_name), surf_normal[None], global_step=iteration)
                            tb_writer.add_images(config['name'] + "_view_{}/rend_alpha".format(viewpoint.image_name), rend_alpha[None], global_step=iteration)
                            
                            rend_dist = render_pkg["rend_dist"]
                            rend_dist = colormap(rend_dist.cpu().numpy()[0])
                            tb_writer.add_images(config['name'] + "_view_{}/rend_dist".format(viewpoint.image_name), rend_dist[None], global_step=iteration)
                        except: pass 

                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)

                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()

                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                logging.info("[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))

                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()
    

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[i*5000 for i in range(1,20)])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--render_after_train",  action='store_true', default=False)
    parser.add_argument("--metrics_after_train",  action='store_true', default=False)
    #parser.add_argument("--data_perturb", nargs="+", type=str, default=[]) #for lego ["color","occ"]
    
    args = parser.parse_args(sys.argv[1:])         
    args.save_iterations.append(args.iterations)     
    args = args_init.argument_init(args)
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    # network_gui.init(args.ip, args.port) # why?
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    op.position_lr_max_steps=op.iterations
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.debug_from, args)
    
    # All done
    print("\nTraining complete.")
