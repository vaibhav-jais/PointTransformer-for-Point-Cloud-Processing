import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Union

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2):
        super(FocalLoss, self).__init__()

        self.gamma = gamma
        self.alpha = alpha
        self.CE_loss = nn.CrossEntropyLoss(ignore_index=-1, reduction='none')

    def forward(self, prediction: torch.Tensor, target: torch.Tensor, class_counts: torch.tensor):
        """
        calculate focal loss with given prediction and true label
        alpha parameter was calculated using the inverse class frequency to address class imbalance (by assigning heigher weights to minority class and vice versa
                                                                                                        based on their frequency in the dataset)
        Args:
            prediction (torch.Tensor): model prediction which will be a raw output (logits)
            target (torch.Tensor): true label

        Returns:
            focal loss

        """
        class_frequencies = class_counts / (torch.sum(class_counts))
        # weight calculation for each class based on it's frequency in the dataset. Giving more weights to the minority class and less weight to the majority class during training.
        inverse_class_frequencies = 1.0 / (class_frequencies + 1e-6)
        ce_loss = self.CE_loss(prediction, target)
        p_t = torch.exp(-ce_loss)
        focal_loss = (inverse_class_frequencies[target]*((1- p_t)**self.gamma)*ce_loss).mean()
        return focal_loss
    
  
if __name__ == "__main__":
    loss = FocalLoss()
    input =torch.randn(10, 6)
    target =torch.tensor([[5], [5], [2], [5], [0], [1], [3], [5], [5], [4]]).T.squeeze().long()
    print(input)
    print(target)
    loss_value = loss(input, target)
    print("Loss", loss_value)