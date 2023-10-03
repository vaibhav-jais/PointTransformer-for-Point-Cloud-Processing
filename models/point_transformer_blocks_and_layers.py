import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os 
import sys
from point_transformer_modules import square_distance, index_points, PointNetSetAbstraction, PointNetFeaturePropagation


class PointTransformerBlock(nn.Module):
    def __init__(self, d_points, d_model, k):
        """_summary_

        Args:
            d_points (_type_): _description_
            d_model (_type_): _description_
            k (int): k nearest neighbors = 16 (as mentioned in the paper)
        """
        super().__init__()
        self.fc1 = nn.Linear(d_points, d_model)
        self.fc2 = nn.Linear(d_model, d_points)
        # positional encoding function of point transformer layer
        self.fc_delta = nn.Sequential(
            nn.Linear(3, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        # mapping function of point transformer layer (substraction as mentioned in the paper)
        self.fc_gamma = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        self.phi = nn.Linear(d_model, d_model, bias=False)      # φ (x_i)
        self.psi = nn.Linear(d_model, d_model, bias=False)      # Ψ (x_j)  x_j is a set of points in local neighborhood ( k nearest neighbors of x_i)
        self.alpha = nn.Linear(d_model, d_model, bias=False)    # α (x_j)  x_j is a set of points in local neighborhood ( k nearest neighbors of x_i)
        self.k = k
        
    # xyz: b x n x 3, features: b x n x f
    def forward(self, xyz, features):
        dists = square_distance(xyz, xyz)
        knn_idx = dists.argsort()[:, :, :self.k]  # b x n x k
        knn_xyz = index_points(xyz, knn_idx)      # p_j   (b x n x k * 3)
        
        pre = features
        x_i = self.fc1(features)        # transforms to 512 dims ecause of transformer
        q, k, v = self.phi(x_i), index_points(self.psi(x_i), knn_idx), index_points(self.alpha(x_i), knn_idx)

        positional_encoding = self.fc_delta(xyz[:, :, None] - knn_xyz)  # b x n x k x f
        
        attention_weights_vectors = self.fc_gamma(q[:, :, None] - k + positional_encoding)
        attention_weights_vectors = F.softmax(attention_weights_vectors / np.sqrt(k.size(-1)), dim=-2)  # b x n x k x f
        
        res = torch.einsum('bmnf,bmnf->bmf', attention_weights_vectors, v + positional_encoding)
        res = self.fc2(res) + pre   # skip connection
        return res, attention_weights_vectors
    

class TransitionDown(nn.Module):
    def __init__(self, num_points, nneighbor, channels):  # num_points is the cardinality of the points produced in each stage, nneighbor is the K in Knn
        super().__init__()
        self.sa = PointNetSetAbstraction(num_points, 0, nneighbor, channels[0], channels[1:], knn=True)  # channels[0] is input channel, channels[1:] is mlp_list
        
    def forward(self, xyz, points):
        return self.sa(xyz, points)
    

class TransitionUp(nn.Module):
    def __init__(self, dim1, dim2, dim_out):
        class SwapAxes(nn.Module):
            def __init__(self):
                super().__init__()
            
            def forward(self, x):
                return x.transpose(1, 2)

        super().__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(dim1, dim_out),
            SwapAxes(),
            nn.BatchNorm1d(dim_out),  # TODO
            SwapAxes(),
            nn.ReLU(),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(dim2, dim_out),
            SwapAxes(),
            nn.BatchNorm1d(dim_out),  # TODO
            SwapAxes(),
            nn.ReLU(),
        )
        self.fp = PointNetFeaturePropagation(-1, [])
    
    def forward(self, xyz1, points1, xyz2, points2):  #xyz1 = b_size*N/256*3,   xyz1 = b_size*N/64*3 at the beginning
        feats1 = self.fc1(points1)
        feats2 = self.fc2(points2)
        feats1 = self.fp(xyz2.transpose(1, 2), xyz1.transpose(1, 2), None, feats1.transpose(1, 2)).transpose(1, 2) #mapping via trilinear interpolation
        return feats1 + feats2
    
class Backbone(nn.Module):
    """contains 5 stages
        1) MLP of 2 linear layers (X input channels and 32 output channels) and a ReLU nonlinearity 
        2)TransitionDown block of N/4 points and PointTransformer block of 64 channels
        3)TransitionDown block of N/16 points and PointTransformer block of 128 channels
        4)TransitionDown block of N/64 points and PointTransformer block of 256 channels
        5)TransitionDown block of N/256 points and PointTransformer block of 512 channels

    Args:
        nn (_type_): _description_
    """
    def __init__(self, num_points, num_blocks, nneighbor, input_dim=5):  
        super().__init__()

        self.nneighbor = nneighbor
        # 1st MLP of the Point Transformer network that directly takes the input point cloud
        self.fc1 = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 32)
        )
        self.transformer1 = PointTransformerBlock(32, 512, nneighbor)
        self.transition_downs = nn.ModuleList()
        self.transformers = nn.ModuleList()
        for i in range(num_blocks):
            channel = 32 * 2 ** (i + 1)
            self.transition_downs.append(TransitionDown(num_points // 4 ** (i + 1), self.nneighbor, [channel // 2 + 3, channel, channel])) # 16 is the nneighbor
            self.transformers.append(PointTransformerBlock(channel, 512, self.nneighbor))
        self.nblocks = num_blocks
    
    def forward(self, input_point_cloud):           # x = 3 coordinated + 3 features
        xyz = input_point_cloud[..., :3]            # Batch_size*Num_points*3coordinates (X;Y;Z)
        points = self.transformer1(xyz, self.fc1(input_point_cloud))[0]    # points is res

        xyz_and_feats = [(xyz, points)]  # stores the inputs and transformed features of each stage of backbone until N/256
        for i in range(self.nblocks):
            xyz, points = self.transition_downs[i](xyz, points)
            points = self.transformers[i](xyz, points)[0]
            xyz_and_feats.append((xyz, points))
        return points, xyz_and_feats