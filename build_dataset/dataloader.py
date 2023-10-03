from torch.utils.data import DataLoader
from build_dataset.dataset import Nuscenes_RadarPC_Dataset
import os
import torch
from torch.nn.utils.rnn import pad_sequence

def custom_collate_fn(batch):
    # Pad sequences to make them of the same length
    data_batch = [item[0] for item in batch]
    labels_batch = [item[1] for item in batch]
    frame_name = [item[2] for item in batch]
    padding_label_value = -1

    #max_sequence_length for padding
    max_length = max(data.shape[0] for data in data_batch)
    # Pad data sequences
    data_padded = pad_sequence(data_batch, batch_first=True, padding_value=0)
    # Pad labels with -1 to distinguish from actual labels (0 to 5)
    labels_padded = [torch.cat([labels, torch.full((max_length - labels.shape[0],), padding_label_value, dtype=torch.long)]) for labels in labels_batch]
    labels_padded = torch.stack(labels_padded)
    # Create a mask to ignore padded labels during loss computation
    mask = (labels_padded != padding_label_value)
    return data_padded, labels_padded, mask, frame_name


#  to wrap an iterable around Dataset enabling easy access to the samples
def data_loader(data_root, batch_size, train, num_workers, pin_memory):
    
    if train == True:
        train_data_path = os.path.join(data_root, "train_keyframes", "")
        train_dataset = Nuscenes_RadarPC_Dataset(data_root=train_data_path)
        dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory, collate_fn=custom_collate_fn, drop_last=True)
    else:
        val_data_path = os.path.join(data_root, "val_keyframes", "")
        val_dataset = Nuscenes_RadarPC_Dataset(data_root=val_data_path)
        dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    return dataloader

if __name__ == "__main__":
    batch_size = 4
    root = './TrainingTooling_Data/nuScenes_dataset_in_AtCity_format/vjaiswalEnd2End/data/'
    dataloader = data_loader(data_root=root, batch_size=batch_size, train=True, num_workers=2, pin_memory=0)
    tb = 0
    for i, (data, labels, mask, frame_name) in enumerate(dataloader):
        print(data.shape, labels.shape, mask.shape)
        tb += 1
    print(tb)   