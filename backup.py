
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os

@torch.jit.script_if_tracing
def farthest_point_sampling(input_pc_with_xyz, desired_num_of_centroids):
    """ This method samples out the centroids (sampled_points_indices) given original input point cloud

    Args:
        input_pc_with_xyz ( ): input points position data, [B, N, d]        d -> d-dim coordinates
        desired_num_of_centroidstorch.tensor (int): number which represents how many centroids are desired

    Returns:
        sampled_centroids (torch.tensor): tensor returning the indices of the sampled centroids, [B, N']
    """
    device = input_pc_with_xyz.device
    B, N, C = input_pc_with_xyz.shape
    sampled_centroids = torch.zeros(B, desired_num_of_centroids, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10  # distance values initialized with large values which we keep updating later
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)   # randomly selecting the index of a point within our point cloud
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(desired_num_of_centroids):
        sampled_centroids[:, i] = farthest
        centroid = input_pc_with_xyz[batch_indices, farthest % N, :].view(B, 1, 3)
        #dist = torch.sum((input_pc_with_xyz[batch_indices, :, :2] - centroid) ** 2, -1)  # computig the distance between the last sampled point with each points in our point set
        dist = torch.sum((input_pc_with_xyz - centroid) ** 2, -1)  # computig the distance between the last sampled point with each points in our point set
        dist = dist.to(torch.float32)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return sampled_centroids


def point_from_index(input_pc_with_xyz, sampled_centroids):
    """

    Input:
        input_pc_with_xyz: input points data, [B, N, d]                 d -> d-dim coordinates
        sampled_centroids: sample index data, [B, N']                   N' -> N' subsampled points
    Return:
        sampled_centroids_points:, indexed points data, [B, N', d]      N' -> N' subsampled points, d -> d-dim coordinates
    """
    #device = input_pc_with_xyz.device
    view_shape = list(sampled_centroids.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(sampled_centroids.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(input_pc_with_xyz.shape[0], dtype=torch.long).view(view_shape).repeat(repeat_shape)
    sampled_centroids_points = input_pc_with_xyz[batch_indices, sampled_centroids, :]
    return sampled_centroids_points


def pairwise_euclidean_distance(input_pc_with_xyz, sampled_centroids_points):
    """
    Calculate the pairwise distance between point in the original point cloud and each sampled centroid point.

    src^T * dst = xn * xm + yn * ym + zn * zm;
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst

    Input:
        input_pc_with_xyz: input points data, [B, N, d]                 d -> d-dim coordinates
        sampled_centroids_points:, indexed points data, [B, N', d]      N' -> N' subsampled points, d -> d-dim coordinates
    Output:
        euclidean_distance: per-point square distance, [B, N, N']
    """
    #B, N_dash, C = sampled_centroids_points.shape                        # N_dash -> N'
    #B, N, C = input_pc_with_xyz.shape
    euclidean_distance = -2 * torch.matmul(input_pc_with_xyz, sampled_centroids_points.permute(0, 2, 1))
    euclidean_distance += torch.sum(input_pc_with_xyz ** 2, -1).view(input_pc_with_xyz.shape[0], input_pc_with_xyz.shape[1], 1)
    euclidean_distance += torch.sum(sampled_centroids_points ** 2, -1).view(sampled_centroids_points.shape[0] , 1, sampled_centroids_points.shape[1])
    return euclidean_distance


def ball_query(radius, num_points_local, input_pc_with_xyz, sampled_centroids_points):
    """ this function finds(say K) neighboring points around each sampled centroid point.
            This method of finding neighboring points around a centroid to build a local region of points is more effective than KNN due to sparsity of point cloud

    Args:
        radius (_type_): _description_
        num_points_local (_type_): _description_
        input_pc_with_xyz (_type_): _description_
        sampled_centroids_points (_type_): _description_

    Returns:
        _type_: _description_
    """
    device = input_pc_with_xyz.device
    #B,N,C = input_pc_with_xyz.shape
    #B,S,C = sampled_centroids_points.shape

    local_group_idx = torch.arange(input_pc_with_xyz.shape[1], dtype=torch.long).to(device).view(
                                1, 1, input_pc_with_xyz.shape[1]).repeat([sampled_centroids_points.shape[0], sampled_centroids_points.shape[1], 1])
    sqrdists = pairwise_euclidean_distance(sampled_centroids_points, input_pc_with_xyz)
    local_group_idx[sqrdists > radius ** 2] = input_pc_with_xyz.shape[1]
    local_group_idx = local_group_idx.sort(dim=-1)[0][:, :, :num_points_local]
    local_group_first = local_group_idx[:, :, 0].view(sampled_centroids_points.shape[0], sampled_centroids_points.shape[1], 1).repeat([1, 1, num_points_local])
    mask = local_group_idx == input_pc_with_xyz.shape[1]  # mask is used here to handle the points in original point cloud that are outside the specified radius from any sampled centroid during grouping
    local_group_idx[mask] = local_group_first[mask]
    local_group_idx = local_group_idx.clamp(0, input_pc_with_xyz.shape[1] - 1)

    return local_group_idx


class SetAbstractionMSG(nn.Module):
    def __init__(self, desired_num_of_centroids, radius_list, nsample_list, in_channel, mlp_list):
        """" PointNet++ SetAbstraction module with MSG (multi-scale grouping)"

        Args:
            desired_num_of_centroids (int): desired num of points sampled in FPS
            radius_list (list(float32)): search radius in local region
            nsample_list (list(int32)): num of desired in each local region (around centroids)
            in_channel (int): no of total input features in data (d+C)
            mlp_list (list(list(int32))): output size for MLP on each point
        """
        super(SetAbstractionMSG, self).__init__()
        self.desired_num_of_centroids = desired_num_of_centroids
        self.radius_list = radius_list
        self.nsample_list = nsample_list
        self.in_channel = in_channel
        self.mlp_list = mlp_list
        self.conv_blocks = nn.ModuleList()
        self.bn_blocks = nn.ModuleList()
        for i in range(len(self.mlp_list)):
            convs = nn.ModuleList()
            bns = nn.ModuleList()
            last_channel = self.in_channel + 3
            for out_channel in self.mlp_list[i]:
                convs.append(nn.Conv2d(last_channel, out_channel, 1))
                bns.append(nn.BatchNorm2d(out_channel))
                last_channel = out_channel
            self.conv_blocks.append(convs)
            self.bn_blocks.append(bns)

    def forward(self, input_pc_with_xyz, input_points_without_xyz):
        """
        Input:
            input_pc_with_xyz: input points position data, [B, N, d]        d -> d-dim coordinates
            input_points_without_xyz: input points data, [B, C, N]          C -> C-dim point features
        Return:
            sampled_centroids_points_xyz: sampled points position data, [B, d , N']               d -> d-dim coordinates,  N' -> N' subsampled points
            new_points_concat: sample points feature data, [B, C', N']       C' -> new C'-dim feature vectors summarizing local context
        """
        #input_pc_with_xyz = input_pc_with_xyz.permute(0, 2, 1)
        if input_points_without_xyz is not None:
            #input_points_without_xyz = input_points_without_xyz.permute(0, 2, 1)
            input_points_without_xyz = input_points_without_xyz

        B, N, d = input_pc_with_xyz.shape
        desired_num_of_centroids = self.desired_num_of_centroids
        sampled_centroids_points_xyz = point_from_index(input_pc_with_xyz, farthest_point_sampling(input_pc_with_xyz, desired_num_of_centroids))
        sampled_centroids_points_list = []
        for i, radius in enumerate(self.radius_list):
            K = self.nsample_list[i]
            local_group_idx = ball_query(radius, K, input_pc_with_xyz, sampled_centroids_points_xyz)
            sampled_and_grouped_points_xyz = point_from_index(input_pc_with_xyz, local_group_idx)
            sampled_and_grouped_points_xyz -= sampled_centroids_points_xyz.view(B, desired_num_of_centroids, 1, d)
            if input_points_without_xyz is not None:
                grouped_points = point_from_index(input_points_without_xyz, local_group_idx)
                grouped_points = torch.cat([grouped_points, sampled_and_grouped_points_xyz], dim=-1) # B*N'*K*C + B*N'*K*d = B*N'*K*(d+C) 
            else:
                grouped_points = sampled_and_grouped_points_xyz

            grouped_points = grouped_points.permute(0, 3, 2, 1)  # [B, d+C, K, N']
            for j in range(len(self.conv_blocks[i])):
                conv = self.conv_blocks[i][j]
                bn = self.bn_blocks[i][j]
                grouped_points =  F.relu(bn(conv(grouped_points)))
            new_points = torch.max(grouped_points, 2)[0]  # [B, d', N']
            sampled_centroids_points_list.append(new_points)

        sampled_centroids_points_xyz = sampled_centroids_points_xyz
        new_points_concat = torch.cat(sampled_centroids_points_list, dim=1)
        new_points_concat = new_points_concat.permute(0, 2, 1)
        return sampled_centroids_points_xyz, new_points_concat


class FeaturePropagation(nn.Module):
    def __init__(self, in_channel, mlp):
        super(FeaturePropagation, self).__init__()
        self.in_channel = in_channel
        self.mlp = mlp
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = self.in_channel
        for out_channel in self.mlp:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel

    def forward(self, xyz1, xyz2, points1, points2):
        """
        Input:
            xyz1: input points position data, [B, N, d]
            xyz2: sampled input points position data, [B, N', d]
            points1: input points data, [B, N, C]
            points2: input points data, [B, N', C]
        Return:
            new_points: upsampled points data, [B, N, C']
        """
        #xyz1 = xyz1.permute(0, 2, 1)
        #xyz2 = xyz2.permute(0, 2, 1)

        #points2 = points2.permute(0, 2, 1)
        B, N, d = xyz1.shape
        _, N_dash, _ = xyz2.shape

        if N_dash == 1:
            interpolated_points = points2.repeat(1, N, 1)
        else:
            dists = pairwise_euclidean_distance(xyz1, xyz2)
            dists, idx = dists.sort(dim=-1)
            dists, idx = dists[:, :, :3], idx[:, :, :3]  # [B, N, 3]

            dist_recip = 1.0 / (dists + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm
            interpolated_points = torch.sum(point_from_index(points2, idx) * weight.view(B, N, 3, 1), dim=2)

        if points1 is not None:
            #points1 = points1.permute(0, 2, 1)
            new_points = torch.cat([points1, interpolated_points], dim=-1)
        else:
            new_points = interpolated_points

        new_points = new_points.permute(0, 2, 1)
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))
        new_points = new_points.permute(0, 2, 1)
        return new_points



if __name__ == '__main__':

    #xyz = torch.rand(2, 2048, 6)
    #sa_msg = SetAbstractionMSG(1024, [0.05, 0.1], [16, 32], 3, [[16, 16, 32], [32, 32, 64]])
    #sampled_centroids_points_xyz, new_points_concat = sa_msg(xyz[:, :, :3], xyz[:, :, 3:])
    #print(sampled_centroids_points_xyz.shape, new_points_concat.shape)
    
    #test_samp_2 = torch.tensor([[0,0], [1,4], [2,1], [2,2], [3,2], [4,1], [6,3], [3,-4], [6,-4], [-2,-4], [-2,4]])
    #test_samp_3 = torch.tensor([[0,0], [1,1], [2,2], [3,3], [4,4], [5,5], [6,6], [7,7], [8,8], [9,9], [10,10]])
    #test_samp_1 = torch.randint(-30, 30, (2, 150, 4))
    #src = torch.tensor([[[-2,3], [3,1], [4,-2], [6,3], [5,1], [2,-3], [6, -2], [3, -5]],
    #                    [[-2,2], [2,-1], [6,-1], [6,-3], [5,2], [2,-5], [6, -6], [-3, -5]]])
    #print(test_samp_1)
    #sampled_points_indices = farthest_point_sampling(test_samp_1, 6)
    #print(f"Indices:",sampled_points_indices)
    #print(f"Corresponding Points:",point_from_index(test_samp_1, sampled_points_indices))
    #print(f"local_group_index", ball_query(0.5, 10, test_samp_1, point_from_index(test_samp_1, sampled_points_indices)))
    data_root = './data/train_keyframes/'
    frame = 'scene-0010_frame3.npy'
    frame_path = os.path.join(data_root, frame)
    data = np.load(frame_path, 'r')
    numpy_array_with_batch_dim = data[np.newaxis, :]
    # Convert the NumPy array to a PyTorch tensor
    tensor_object = torch.tensor(numpy_array_with_batch_dim).float()
    sa_msg = SetAbstractionMSG(128, [.1, .2], [16, 32], 3, [[16, 16, 32], [32, 32, 64]])
    sampled_centroids_points_xyz, new_points_concat = sa_msg(tensor_object[:, :, :3], tensor_object[:, :, 3:6])
    print(sampled_centroids_points_xyz.shape, new_points_concat.shape)


