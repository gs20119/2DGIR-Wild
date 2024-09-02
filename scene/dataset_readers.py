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
import sys
from PIL import Image,ImageDraw
from typing import NamedTuple
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import json
from pathlib import Path
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB
from scene.gaussian_model import BasicPointCloud
import pandas as pd
import glob
import yaml

class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    CenterX: np.array
    CenterY: np.array
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int

class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str

def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)     
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}

def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder):
    cam_infos = []
  
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        # Assume dataset images are undistorted
        if intr.model in ["SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL"]: 
            focal = intr.params[0]
            center_x, center_y = intr.params[1], intr.params[2]
            FovY = focal2fov(focal, height)
            FovX = focal2fov(focal, width)
        else:
            focal_x, focal_y = intr.params[0], intr.params[1]
            center_x, center_y = intr.params[2], intr.params[3]
            FovX = focal2fov(focal_x, width)
            FovY = focal2fov(focal_y, height)

        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]
        image = Image.open(image_path)        

        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, CenterX=center_x, CenterY=center_y,
            image=image, image_path=image_path, image_name=image_name, width=width, height=height)     
        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos

def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)

def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

def readColmapSceneInfo(path, images, eval, llffhold=8):
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse", "cameras.bin")   
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)     
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)    
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    reading_dir = "images" if images == None else images

    cam_infos_unsorted = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, images_folder=os.path.join(path, reading_dir))
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)

    if eval:
        root_dir=os.path.dirname(path)
        tsv = glob.glob(os.path.join(root_dir, '*.tsv'))[0]
        scene_name = os.path.basename(tsv)[:-4]  
        files = pd.read_csv(tsv, sep='\t')                          
        files = files[~files['id'].isnull()]   
        files.reset_index(inplace=True, drop=True)

        img_path_to_id = {}
        for v in cam_extrinsics.values():
            img_path_to_id[v.name] = v.id
        img_ids = []
        image_paths = {} # {id: filename}
        for filename in list(files['filename']):
            if filename in img_path_to_id:
                id_ = img_path_to_id[filename]
                image_paths[id_] = filename              
                img_ids += [id_]               

        img_ids_train = [id_ for i, id_ in enumerate(img_ids) 
                                    if files.loc[i, 'split']=='train']
        img_ids_test = [id_ for i, id_ in enumerate(img_ids)
                                    if files.loc[i, 'split']=='test']
        
        train_cam_infos =[ c for c in cam_infos if c.uid in img_ids_train]
        test_cam_infos =[ c for c in cam_infos if c.uid in img_ids_test]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "sparse/points3D.ply")
    bin_path = os.path.join(path, "sparse/points3D.bin")
    txt_path = os.path.join(path, "sparse/points3D.txt")
    if not os.path.exists(ply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)
        storePly(ply_path, xyz, rgb)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

def readCamerasFromTransforms(path, transformsfile, white_background, extension=".png",data_perturb=None,split="train"):
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        FovX = contents["camera_angle_x"]

        frames = contents["frames"]      
        for idx, frame in enumerate(frames):
            cam_name = os.path.join(path, frame["file_path"] + extension)     

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])         
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            image = Image.open(image_path)
            if idx != 0 and split=="train":
                image=add_perturbation(image,data_perturb,idx)
                   
            im_data = np.array(image.convert("RGBA"))  
            bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])

            norm_data = im_data / 255.0
            arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])      
            image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")
            
            focal = fov2focal(FovX, image.size[0])
            FovY = focal2fov(focal, image.size[1])
            center_x = image.size[0]/2
            center_y = image.size[1]/2

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, CenterX=center_x, CenterY=center_y,
                image=image, image_path=image_path, image_name=image_name, width=image.size[0], height=image.size[1]))
            
    return cam_infos

def readNerfSyntheticInfo(path, white_background, eval, extension=".png",data_perturb=None):
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", white_background, extension,data_perturb=data_perturb,split="train")  #[CameraInfo(id，fov，R，T，图片路径，图片，高，宽)...100长度]
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(path, "transforms_test.json", white_background, extension,data_perturb=None,split="test")
    
    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)          

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")
        
        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3         
        shs = np.random.random((num_pts, 3)) / 255.0              
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)            
    try:
        pcd = fetchPly(ply_path)                
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

def readCamerasFromReNeTransforms(path, white_background, num_cameras, lid): # DataLoader, not completed TODO
    cam_infos = []
    path = os.path.join(path, f"lset{lid:03d}")

    cam_file = open(os.path.join(path, "camera.yaml"))
    cam_intrinsics = yaml.load(cam_file, Loader=yaml.FullLoader)["intrinsics"]
    FovX = focal2fov(cam_intrinsics["camera_matrix"][0][0])
    FovY = focal2fov(cam_intrinsics["camera_matrix"][1][1])
    cam_file.close()

    for cid in range(num_cameras):
        pose_path = os.path.join(path, "data", f"{cid:02d}_pose.txt")
        image_path = os.path.join(path, "data", f"{cid:02d}_image.png")

        ext_file = open(pose_path, 'r')
        c2w = np.asarray([[float(num) for num in line.split()] for line in ext_file])
        c2w[:3, 1:3] *= -1
        w2c = np.linalg.inv(c2w)
        R = np.transpose(w2c[:3,:3])
        T = w2c[:3, 3]
        ext_file.close()

        image_name = f"image{lid:03d}_{cid:02d}.png"
        image = Image.open(image_path)
        image_norm = np.array(image.convert("RGBA")) / 255.0
        bg = np.array([1,1,1]) if white_background else np.array([0,0,0])
        image_mix = image_norm[:,:,:3] * image_norm[:, :, 3:4] + bg * (1 - image_norm[:, :, 3:4])      
        image = Image.fromarray(np.array(image_mix*255.0, dtype=np.byte), "RGB")

        idx = num_cameras*lid+cid
        cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
            image_path=image_path, image_name=image_name, width=image.size[0], height=image.size[1]))

    return cam_infos

def readReNeSyntheticInfo(path, white_background, eval, extension=".png", num_cameras=50, num_lights=40): # TODO
    pass

sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Blender" : readNerfSyntheticInfo,
    "ReNe": readReNeSyntheticInfo
}

def add_perturbation(img, perturbation, seed):
    if 'occ' in perturbation:
        draw = ImageDraw.Draw(img)
        np.random.seed(seed)
        left = np.random.randint(200, 400)
        top = np.random.randint(200, 400)
        for i in range(10):
            np.random.seed(10*seed+i)
            random_color = tuple(np.random.choice(range(256), 3))
            draw.rectangle(((left+20*i, top), (left+20*(i+1), top+200)),
                            fill=random_color)

    if 'color' in perturbation:
        np.random.seed(seed)
        img_np = np.array(img)/255.0     #H,W,4
        s = np.random.uniform(0.8, 1.2, size=3)   #
        b = np.random.uniform(-0.2, 0.2, size=3)   #
        img_np[..., :3] = np.clip(s*img_np[..., :3]+b, 0, 1)
        img = Image.fromarray((255*img_np).astype(np.uint8))

    return img