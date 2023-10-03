import torch
import torch.nn as nn
from point_transformer_blocks_and_layers import TransitionUp, Backbone, PointTransformerBlock


class PointTransformerSeg(nn.Module):
    def __init__(self, num_points, nneighbors):
        super().__init__()
        self.num_points = num_points
        self.nneighbors = nneighbors
        self.num_blocks = 4
        self.num_classes = 6
        self.transformer_dim = 512
        self.backbone = Backbone(self.num_points, self.num_blocks, self.nneighbors)
        self.fc2 = nn.Sequential(
            nn.Linear(32 * 2 ** self.num_blocks, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 32 * 2 ** self.num_blocks)
        )
        self.transformer2 = PointTransformerBlock(32 * 2 ** self.num_blocks, self.transformer_dim, self.nneighbors)
    
        self.transition_ups = nn.ModuleList()
        self.transformers = nn.ModuleList()
        for i in reversed(range(self.num_blocks)):
            channel = 32 * 2 ** i
            self.transition_ups.append(TransitionUp(channel * 2, channel, channel))
            self.transformers.append(PointTransformerBlock(channel, self.transformer_dim, self.nneighbors))

        self.fc3 = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, self.num_classes)
        )
    
    def forward(self, x):
        points, xyz_and_feats = self.backbone(x)    # points (b_size*N/256*512) xyz_and_feats (list of output of each feature encoder stage)
        xyz = xyz_and_feats[-1][0]                  # b_size*N/256*3 (xyz output of last TD block)
        points = self.transformer2(xyz, self.fc2(points))[0]  # transformer layer attached with MLP layer after the end of Backbone but before the TU block

        for i in range(self.num_blocks):
            points = self.transition_ups[i](xyz, points, xyz_and_feats[- i - 2][0], xyz_and_feats[- i - 2][1]) #2nd last xyz and feature
            xyz = xyz_and_feats[- i - 2][0]
            points = self.transformers[i](xyz, points)[0]
            
        return self.fc3(points)
    

if __name__ == '__main__':
    input = torch.rand(4, 1024, 6)
    import numpy as np
    import os
    data_root = './data/train_keyframes/'
    frame = 'scene-0517_frame33.npy'
    frame_path = os.path.join(data_root, frame)
    data = np.load(frame_path, 'r')
    numpy_array_with_batch_dim = data[np.newaxis, :]
    # Convert the NumPy array to a PyTorch tensor
    tensor_data = torch.tensor(numpy_array_with_batch_dim).float()
    model = PointTransformerSeg()
    x = model(tensor_data[:, :, :6])
    print(x)