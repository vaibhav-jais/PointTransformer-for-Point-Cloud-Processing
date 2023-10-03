import torch
import torch.nn as nn
import torch.nn.functional as F
import sys 
import os

# Add the root folder to the Python path
root_folder = os.path.dirname(os.path.abspath(__file__))
sys.path.append(root_folder)

from pointnet2_modules import SetAbstractionMSG, FeaturePropagation

class PointNet2Model(nn.Module):
    def __init__(self, num_classes):
        super(PointNet2Model, self).__init__()
        self.num_classes = num_classes
        #set abstraction layers
        self.sa1 = SetAbstractionMSG(1024, [1, 3], [8, 32], 3, [[32, 32, 64], [64, 64, 128]])
        self.sa2 = SetAbstractionMSG(512, [2, 4], [8, 32], 64+128, [[32, 32, 64], [64, 64, 128]])
        self.sa3 = SetAbstractionMSG(256, [3, 6], [16, 32], 64+128, [[64, 64, 128], [64, 64, 128]])
        self.sa4 = SetAbstractionMSG(128, [4, 8], [16, 32], 128+128, [[128, 256, 512], [128, 256, 512]])

        # feature propagation layers
        self.fp4 = FeaturePropagation(512+512+128+128, [256, 256])
        self.fp3 = FeaturePropagation(64+128+256, [256, 256])
        self.fp2 = FeaturePropagation(64+128+256, [256, 128])
        self.fp1 = FeaturePropagation(128, [128, 128, 128])

        #fully connected layers
        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 =  nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(128, self.num_classes, 1)

    def forward(self, input_pc):

        input_pc_with_xyz = input_pc[:, :, :3]
        input_pc_without_xyz = input_pc[:, :, 3:]
        
        l1_xyz, l1_points = self.sa1(input_pc_with_xyz, input_pc_without_xyz)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)

        l3_points = self.fp4(l3_xyz, l4_xyz, l3_points, l4_points)
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(input_pc_with_xyz, l1_xyz, None, l1_points)
        l0_points = l0_points.permute(0,2,1)

        x = self.drop1(F.relu(self.bn1(self.conv1(l0_points))))
        x = self.conv2(x)
        x = x.permute(0, 2, 1)
        return x

if __name__ == '__main__':
    import  torch
    import os
    import numpy as np
    """
    xyz = torch.rand(4, 229, 7)
    model = PointNet2Model(num_classes=6)
    #x, l4_points = model(xyz)
    x = model(xyz)
    print(x.shape)

    """
    data_root = './data/train_keyframes/'
    frame = 'scene-0871_frame26.npy'
    frame_path = os.path.join(data_root, frame)
    data = np.load(frame_path, 'r')
    numpy_array_with_batch_dim = data[np.newaxis, :]
    # Convert the NumPy array to a PyTorch tensor
    tensor_object = torch.tensor(numpy_array_with_batch_dim).float()
    model = PointNet2Model(num_classes=6)
    x = model(tensor_object[:, :, :6])
    print(x)
    
