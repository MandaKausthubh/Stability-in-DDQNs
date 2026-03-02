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


class DQN(nn.Module):
    def __init__(
            self,
            input_dim,
            hidden_dims,
            rep_dim,
            use_bias=True
    ):
        super(DQN, self).__init__()
        self.feature_extractor = FeatureExtractor(input_dim, hidden_dims, rep_dim)
        self.weights = nn.Linear(rep_dim, 1, bias=use_bias)

    def forward(self, x):
        features = self.feature_extractor(x)
        output = self.weights(features)
        return output.squeeze()

    def get_weights(self):
        return self.weights.weight.data, (
            self.weights.bias.data if self.weights.bias
            is not None else None
        )
