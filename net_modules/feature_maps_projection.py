
import torch
import torch.nn.functional as F


def project2d(pointcloud, image, world2camera,camera2image): # removed box_coord

    xyz_world_h=torch.cat([pointcloud,torch.ones(size=(pointcloud.shape[0],1), dtype=pointcloud.dtype, device="cuda")],dim=1)
    xyz_camera=torch.mm(xyz_world_h,world2camera)[:,:3]       #N,3   
    xy_image0=torch.mm(xyz_camera,camera2image.transpose(0, 1))  #N,3  
    xy_image=torch.zeros((xy_image0.shape[0],2),device=xy_image0.device)
    xy_image[:,0]=xy_image0[:,0]/(xy_image0[:,2]+1e-2)
    xy_image[:,1]=xy_image0[:,1]/(xy_image0[:,2]+1e-2)    #N,2

    width, height = image.shape[-1], image.shape[-2]
    mask_in_width=torch.logical_and(0<xy_image[:,0],xy_image[:,0]<width)
    mask_in_height=torch.logical_and(0<xy_image[:,1],xy_image[:,1]<height)
    mask_in_image=torch.logical_and(mask_in_width, mask_in_height)
    mask_front_point=xy_image0[:,2]>0
    valid_point_mask=torch.logical_and(mask_in_image,mask_front_point) # [N]
    valid_pixel=xy_image[valid_point_mask][:,:2]     

    # valid_pixel_normal works like a float coordinate of the image    
    valid_pixelx_normalized = valid_pixel[:,0]/(width/2)-1 # [N_val] -1.0 ~ 1.0
    valid_pixely_normalized = valid_pixel[:,1]/(height/2)-1 # [N_val] -1.0 ~ 1.0
    valid_pixel_normal=torch.stack((valid_pixelx_normalized,valid_pixely_normalized),dim=1) # [N_val,2]
    valid_pixel_normal=torch.unsqueeze(valid_pixel_normal,0)
    valid_pixel_normal=torch.unsqueeze(valid_pixel_normal,0) # [1, 1, N_val, 2]

    coordinates=torch.zeros(size=(1,1,pointcloud.shape[0],2), 
        dtype=pointcloud.dtype, device=pointcloud.device)
    coordinates[:,:,valid_point_mask,:]=valid_pixel_normal

    return coordinates, valid_point_mask
