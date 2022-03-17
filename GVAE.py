import torch
import grammar
import parameters
import torch.nn as nn

from grammar import D

# D = length of the grammar rules
num_in_channels = D
params = parameters.load_params()
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
    
    self.conv1 = nn.Conv1d(in_channels=num_in_channels, out_channels=9, kernel_size=9)
    self.conv2 = nn.Conv1d(in_channels=9, out_channels=9, kernel_size=9)
    self.conv3 = nn.Conv1d(in_channels=9, out_channels=10, kernel_size=11)
    self.linear = nn.Linear(740, 435)

    self.mu = nn.Linear(435, params['latent_dim'])
    self.sigma = nn.Linear(435, params['latent_dim'])

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
    h = h.contiguous().view(h.size(0), -1) # flatten
    h = self.relu(self.linear(h))
    
    mu = self.mu(h)
    sigma = self.sigma(h)

    return mu, sigma
    
    
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

    self.linear_in = nn.Linear(params['latent_dim'], params['latent_dim'])
    self.linear_out = nn.Linear(501, num_in_channels)

    self.rnn = nn.GRU(input_size = params['latent_dim'], hidden_size = 501, num_layers = params['n_layers'], batch_first=True)

    self.relu = nn.LeakyReLU()

  def forward(self, z):
    h = self.relu(self.linear_in(z))
    h = h.unsqueeze(1).expand(-1, params['max_length'], -1)  #[batch, MAX_LENGHT, latent_dim] This does the same as the repeatvector on keras
    h, _ = self.rnn(h)
    h = self.relu(self.linear_out(h))

    return h
    
    
class GrammarVAE(nn.Module):
  """
  The grammarVAE model. Implementation follows the original paper with the
  addition of the 'conditional' function
  """

  def __init__(self):
    super(GrammarVAE, self).__init__()
    
    self.encoder = Encoder()
    self.decoder = Decoder()

  def sample(self, mu, sigma):
    """Reparametrization trick to sample z"""
    sigma = torch.exp(0.5 * sigma)
    epsilon = torch.randn(params['batch'], params['latent_dim']).to(device)     
    return mu + sigma * epsilon  

  def kl(self, mu, sigma):
    """KL divergence between the approximated posterior and the prior"""
    return - 0.5 * torch.mean(1 + sigma - mu.pow(2) - sigma.exp()) 

  def conditional(self, x_true, x_pred):
    most_likely = torch.argmax(x_true, dim=-1)
    most_likely = most_likely.view(-1) # flatten most_likely
    ix2 = torch.unsqueeze(grammar.ind_of_ind[most_likely], -1) # index ind_of_ind with res
    ix2 = ix2.type(torch.LongTensor)
    M2 = grammar.masks[list(ix2.T)]
    M3 = torch.reshape(M2, (params['batch'], params['max_length'], grammar.D))
    P2 = torch.mul(torch.exp(x_pred), M3.float()) # apply them to the exp-predictions
    P2 = torch.divide(P2, torch.sum(P2, dim=-1, keepdims=True)) # normalize predictions
    return P2

  def forward(self, x):
    mu, sigma = self.encoder(x)
    z = self.sample(mu, sigma)
    logits = self.decoder(z)
    return z, mu, sigma, logits