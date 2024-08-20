import torch
import torch.nn.functional as F

def normalize_pts(pts, box):
    if box.shape[0]==1:
        box=box.squeeze()
        return (pts - box[1,:].unsqueeze(0)) * (2.0 / (box[0,:].unsqueeze(0) - box[1,:].unsqueeze(0))) - 1.0
    elif box.shape[0]>1:
        return (pts - box[:,1,:]) * (2.0 / (box[:,0,:] - box[:,1,:])) - 1.0
    else:
        return None


def sample_from_feature_maps(feature_maps, pts, box_coord, coord_scale=1):

    n_maps, C, H, W = feature_maps.shape
    coordinates=box_coord.permute(1,0,2)/coord_scale # [M=2, N, 2] (M = num_maps without proj.)
    coordinates=coordinates.unsqueeze(1) # [M, 1, N, 2]

    output_features = F.grid_sample(
        feature_maps, coordinates.float(), 
        mode='bilinear', padding_mode='border').permute(2,3,0,1).squeeze()
    output_features = output_features.reshape((output_features.shape[0],-1)) 

    return output_features, coordinates
