import torch.nn as nn

class DummyModel(nn.Module):
    """ define the dummy model to create the working training pipeline"""
    
    def __init__(self, num_features, num_classes):
        super(DummyModel, self).__init__()
        self.num_features = num_features
        self.num_classes = num_classes

        # Identity Layers
        self.hidden_layer_1 = nn.Linear(in_features=self.num_features, out_features=128)
        self.hidden_layer_2 = nn.Linear(in_features=128, out_features=64)
        self.hidden_layer_3 = nn.Linear(in_features=64, out_features=32)
        self.hidden_layer_4 = nn.Linear(in_features=32, out_features=16)
        self.output_layer = nn.Linear(in_features=16, out_features=self.num_classes)

    def forward(self, point_cloud):

        flatten_points = point_cloud[0]

        out_hidden_1 = self.hidden_layer_1(flatten_points)
        out_hidden_2 = self.hidden_layer_2(out_hidden_1)
        out_hidden_3 = self.hidden_layer_3(out_hidden_2)
        out_hidden_4 = self.hidden_layer_4(out_hidden_3)

        out_output = self.output_layer(out_hidden_4)

        return out_output


if __name__ == "__main__":
    pass
