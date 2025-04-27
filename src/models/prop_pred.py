import torch.nn as nn

class PropertiesPredictionModel(nn.Module):

    def __init__(self, input_dim, hidden_size):
        super(PropertiesPredictionModel, self).__init__()
        self.feedforward = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, x):
        output = self.feedforward(x)
        return output
