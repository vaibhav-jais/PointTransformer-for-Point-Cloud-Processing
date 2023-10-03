import os
import numpy as np
import torch
from torch.utils.data import Dataset
from natsort import natsorted

class Nuscenes_RadarPC_Dataset(Dataset):

    def __init__(self, data_root):
        """  method to initialize our variables
        Args:
            data_folder_path ( str ): contains the base path to our keyframes extracted from all the scenes
        """
        self.data_root = data_root
        self.frames = natsorted(os.listdir(self.data_root))

    def __len__(self):
        """ Torch internally uses this function to understand the size of the dataset in its dataloader, to call
            the  __getitem__() method with an index within this dataset size
        Returns:
            int : returns the size of our dataset ( number of frames in our case )
        """
        return len(self.frames)
    
    def __getitem__(self, index):
        """  function to retrieve the keyframes and point labels corresponding to the ids from self.frames list
        Args:
            index (int): corresponds to which frame to be returned from our self.frames list
        Returns:
            tuple : returns the point cloud data and its corresponding label as tuple
        """
        frame_file = self.frames[index]
        frame_file_path = os.path.join(self.data_root, frame_file)
        frame_data = np.load(frame_file_path)
        
        #frame_radar_point_data = torch.tensor(frame_data[:, [0, 1, 2, 7, 8, 15, 16]])
        frame_radar_point_data = torch.tensor(frame_data[:, [0, 1, 2, 7, 8]])
        frame_radar_point_label = torch.tensor(frame_data[:, -1])

        """  
        Training models with torch requires us to convert variables to the torch tensor format, that contain internal methods for calculating gradients, etc.
        """
        return frame_radar_point_data, frame_radar_point_label, frame_file
    

def class_weight(data_root, train):
    if train == True:
        train_data_root = os.path.join(data_root, "train_keyframes")
    else:
        train_data_root = os.path.join(data_root, "val_keyframes")
    all_frames = natsorted(os.listdir(train_data_root))
    all_targets = []
    for i in range(len(all_frames)):
        current_frame_name = all_frames[i]
        current_frame_file_path = os.path.join(train_data_root, current_frame_name)
        target = np.load(current_frame_file_path,'r')[:, -1]
        all_targets.append(target)
    results_targets = np.hstack(all_targets).astype(float)
    index_counts = torch.histc(torch.tensor(results_targets), bins=6, min=0, max=5)

    return index_counts
        

if __name__ == '__main__':

    data_root = './TrainingTooling_Data/nuScenes_dataset_in_AtCity_format/vjaiswalEnd2End/data/train_keyframes/'
    data = Nuscenes_RadarPC_Dataset(data_root=data_root)
    print('point data size:', data.__len__())
    print('point data 0 shape:', data.__getitem__(0)[0].shape)
    print('point label 0 shape:', data.__getitem__(0)[1].shape)
    print('frame 0 name:', data.__getitem__(0)[2])