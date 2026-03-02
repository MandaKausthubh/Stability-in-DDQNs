import torch.nn as nn

class FeatureExtractor(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super(FeatureExtractor, self).__init__()
        self.layers = nn.ModuleList()
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            self.layers.append(nn.Linear(prev_dim, hidden_dim))
            self.layers.append(nn.ReLU())
            prev_dim = hidden_dim
        self.layers.append(nn.Linear(prev_dim, output_dim))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

