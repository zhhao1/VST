import numpy as np
import math
import torch
import torch.nn as nn
import torch.distributed as dist
from torch import Tensor
from torch.nn import functional as F
from torch.distributions import Distribution, uniform

class CouplingLayer(nn.Module):
  """
  Implementation of the additive coupling layer from section 3.2 of the NICE
  paper.
  """

  def __init__(self, data_dim, hidden_dim, mask, num_layers=4):
    super().__init__()

    assert data_dim % 2 == 0

    self.mask = mask

    modules = [nn.Linear(data_dim, hidden_dim), nn.LeakyReLU(0.2)]
    for i in range(num_layers - 2):
      modules.append(nn.Linear(hidden_dim, hidden_dim))
      modules.append(nn.LeakyReLU(0.2))
    modules.append(nn.Linear(hidden_dim, data_dim))

    self.m = nn.Sequential(*modules)

  def forward(self, x, logdet, invert=False):
    self.mask = self.mask.to(x.dtype)
    if not invert:
      x1, x2 = self.mask * x, (1. - self.mask) * x
      y1, y2 = x1, x2 + (self.m(x1) * (1. - self.mask))
      return y1 + y2, logdet

    # Inverse additive coupling layer
    y1, y2 = self.mask * x, (1. - self.mask) * x
    x1, x2 = y1, y2 - (self.m(y1) * (1. - self.mask))
    return x1 + x2, logdet


class ScalingLayer(nn.Module):
  """
  Implementation of the scaling layer from section 3.3 of the NICE paper.
  """
  def __init__(self, data_dim):
    super().__init__()
    self.log_scale_vector = nn.Parameter(torch.randn(1, data_dim, requires_grad=True))

  def forward(self, x, logdet, invert=False):
    log_det_jacobian = torch.sum(self.log_scale_vector)

    if invert:
        return torch.exp(- self.log_scale_vector) * x, logdet - log_det_jacobian

    return torch.exp(self.log_scale_vector) * x, logdet + log_det_jacobian


class LogisticDistribution(Distribution):
  def __init__(self):
    super().__init__()

  def log_prob(self, x):
    #return self.m.log_prob(x)
    return -(F.softplus(x) + F.softplus(-x))

  def sample(self, size):
    z = Uniform(torch.tensor([0.]), torch.tensor([1.])).sample(size)
    return torch.log(z) - torch.log(1. - z)

class GaussianDistribution(Distribution):
  def __init__(self, data_dim):
    super().__init__()
    self.m = torch.distributions.normal.Normal(torch.zeros((1,data_dim)), torch.ones((1,data_dim))) 
    
  def log_prob(self, x):
    return self.m.log_prob(x).sum(dim=-1) #B*1

  def sample(self, size):
    return self.m.sample(size)
    
class NICE(nn.Module):
  def __init__(self, data_dim, num_coupling_layers=2, num_coupling_hidden_layers=3, num_coupling_hidden_dim=256):
    super().__init__()

    self.data_dim = data_dim

    # alternating mask orientations for consecutive coupling layers
    masks = [self._get_mask(data_dim, orientation=(i % 2 == 0))
                                            for i in range(num_coupling_layers)]

    self.coupling_layers = nn.ModuleList([CouplingLayer(data_dim=data_dim,
                                hidden_dim=num_coupling_hidden_dim,
                                mask=masks[i], num_layers=num_coupling_hidden_layers)
                              for i in range(num_coupling_layers)])

    self.scaling_layer = ScalingLayer(data_dim=data_dim)

    self.prior = LogisticDistribution()

  def forward(self, x, invert=False):
    if not invert:
      z, log_det_jacobian = self.f(x)
      log_likelihood = torch.sum(self.prior.log_prob(z)) + log_det_jacobian
      return z, log_likelihood

    return self.f_inverse(x)

  def f(self, x):
    z = x
    log_det_jacobian = 0
    for i, coupling_layer in enumerate(self.coupling_layers):
      z, log_det_jacobian = coupling_layer(z, log_det_jacobian)
    z, log_det_jacobian = self.scaling_layer(z, log_det_jacobian)
    return z, log_det_jacobian

  def f_inverse(self, z):
    x = z
    x, _ = self.scaling_layer(x, 0, invert=True)
    for i, coupling_layer in reversed(list(enumerate(self.coupling_layers))):
      x, _ = coupling_layer(x, 0, invert=True)
    return x

  def sample(self, num_samples):
    z = self.prior.sample([num_samples, self.data_dim]).view(self.samples, self.data_dim)
    return self.f_inverse(z)

  def _get_mask(self, dim, orientation=True):
    mask = np.zeros(dim)
    mask[::2] = 1.
    if orientation:
      mask = 1. - mask     # flip mask orientation
    mask = torch.tensor(mask)
    return mask.cuda()

