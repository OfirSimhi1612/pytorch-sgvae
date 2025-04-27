import torch
from src.config.hyper_parameters import hyper_params
import torch.nn as nn

from src.grammar import D

# D = length of the grammar rules
num_in_channels = D
device = "cuda" if torch.cuda.is_available() else "cpu"

class Decoder(nn.Module):
  """
  GRU decoder for the Grammar VAE

  The implementation is equivalent than the original paper, 
  only translated to pytorch
  """
  def __init__(self):
    """
    The network layers are defined in the __init__ function
    """
    super(Decoder, self).__init__()

    self.linear_in = nn.Linear(hyper_params['latent_dim'], hyper_params['latent_dim'])
    self.linear_out = nn.Linear(501, num_in_channels)

    self.rnn = nn.GRU(input_size = hyper_params['latent_dim'], hidden_size = 501, num_layers = hyper_params['n_layers'], batch_first=True)

    self.relu = nn.LeakyReLU()

  def forward(self, z):
    h = self.relu(self.linear_in(z))
    h = h.unsqueeze(1).expand(-1, hyper_params['input_dim'], -1)  #[batch, MAX_LENGHT, latent_dim] This does the same as the repeatvector on keras
    h, _ = self.rnn(h)
    h = self.relu(self.linear_out(h))

    return h