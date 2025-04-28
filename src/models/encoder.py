import torch
import torch.nn as nn

from src.config.hyper_parameters import hyper_params
from src.grammar import D

# D = length of the grammar rules
num_in_channels = D
hyper_params
device = "cuda" if torch.cuda.is_available() else "cpu"


class Encoder(nn.Module):
    """
    Convolutional encoder for the Grammar VAE
    The implementation is equivalent to the original paper,
    only translated to pytorch
    """

    def __init__(self):
        """
        The network layers are defined in the __init__ function
        """
        super(Encoder, self).__init__()

        self.conv1 = nn.Conv1d(
            in_channels=num_in_channels, out_channels=9, kernel_size=9
        )
        self.conv2 = nn.Conv1d(in_channels=9, out_channels=9, kernel_size=9)
        self.conv3 = nn.Conv1d(in_channels=9, out_channels=10, kernel_size=11)
        self.linear = nn.Linear(740, 435)

        self.mu = nn.Linear(435, hyper_params["latent_dim"])
        self.sigma = nn.Linear(435, hyper_params["latent_dim"])

        self.relu = nn.LeakyReLU()

    def forward(self, x):
        """
        The operations of the layers defined in __init__ are done in the forward
        function
        """
        h = self.relu(self.conv1(x))
        h = self.relu(self.conv2(h))
        h = self.relu(self.conv3(h))
        h = torch.transpose(h, 1, 2)  # need to transpose to get the right output
        h = h.contiguous().view(h.size(0), -1)  # flatten
        h = self.relu(self.linear(h))

        mu = self.mu(h)
        sigma = self.sigma(h)

        return mu, sigma
